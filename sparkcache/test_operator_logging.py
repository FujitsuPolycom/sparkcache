"""Operator logs preserve metrics and isolate verbose diagnostics at DEBUG."""

import types
from unittest import mock
import uuid

import pytest

from sparkcache.test_spark_context_cache_connector import (
    KVConnectorRole, _make_connector, connector_module,
)
from sparkcache.spark_context_cache_cuda_hybrid_restore import _integrity_label


def test_changed_counters_survive_reset_without_dropping_metrics():
    stats = connector_module.SparkCacheStats(data={"reports": []})
    report = {"rank": 0, "held": [], "counters": {"restore_hit": 2, "load_failed": 0}}
    stats.aggregate(connector_module.SparkCacheStats(data={"reports": [report]}))
    assert stats.format_log_lines() == ("sparkcache: counters restore_hit=2 ranks=1",)
    assert stats.reduce()["sparkcache_counter_load_failed"] == 0
    stats.reset()
    stats.aggregate(connector_module.SparkCacheStats(data={"reports": [report]}))
    assert stats.format_log_lines() == ()
    report["counters"]["restore_hit"] = 3
    assert stats.format_log_lines() == ("sparkcache: counters restore_hit=3",)
    report["counters"]["restore_hit"] = 0
    assert stats.format_log_lines() == ("sparkcache: counters restore_hit=0",)
    assert stats.reduce()["sparkcache_counter_restore_hit"] == 0


def test_r3_counter_history_survives_logger_accumulator_replacement():
    generation = uuid.uuid4().hex

    def interval(value=2, worker_generation=generation):
        return connector_module.SparkCacheStats(data={"reports": [{
            "rank": 0, "generation": worker_generation, "held": [],
            "counters": {"restore_hit": value},
        }]})

    assert interval().format_log_lines()
    # KVConnectorLogging resets its accumulator to None each interval and
    # reconstructs SparkCacheStats from the next observation's wire data.
    assert interval().format_log_lines() == ()
    assert interval().format_log_lines() == ()
    assert interval(3).format_log_lines() == ("sparkcache: counters restore_hit=3",)
    assert interval(0).format_log_lines() == ("sparkcache: counters restore_hit=0",)
    assert interval(0).reduce()["sparkcache_counter_restore_hit"] == 0
    assert interval(worker_generation=uuid.uuid4().hex).format_log_lines()


@pytest.mark.parametrize("committed,reason", [(True, "ok"), (False, "error")])
def test_save_outcome_is_one_short_line(tmp_path, committed, reason):
    connector = _make_connector(tmp_path, 0, 64)
    connector._store_pending_tokens = 128
    connector._store_pending_started_ns = 1_000_000
    with mock.patch.object(connector_module.time, "perf_counter_ns", return_value=3_000_000), \
            mock.patch.object(connector_module, "logger") as log:
        connector._finish_store("a" * 64, committed=committed)
    lines = [call.args[0] % call.args[1:] for call in log.info.call_args_list]
    outcome = "committed" if committed else "refused"
    assert lines == [f"sparkcache: save {outcome} tokens=128 ms=2.0 rank=0 reason={reason}"]
    connector.shutdown()


@pytest.mark.parametrize("invalid,line", [
    (set(), "sparkcache: restore hit tokens=1024 ms=2.0"),
    ({3}, "sparkcache: restore failed tokens=1024 ms=2.0; recomputing"),
])
def test_scheduler_logs_one_line_per_restore(tmp_path, invalid, line):
    scheduler = _make_connector(tmp_path, 0, dcp=1, role=KVConnectorRole.SCHEDULER)
    digest = "c" * 64
    scheduler._restore_flights[digest] = connector_module._RestoreFlight(
        digest=digest, span_tokens=1024, leader_request_id="reader",
        restored_block_ids=frozenset({3}), started_ns=1_000_000,
    )
    scheduler._restore_flight_leaders["reader"] = digest
    with mock.patch.object(connector_module.time, "perf_counter_ns", return_value=3_000_000),             mock.patch.object(connector_module, "logger") as log:
        scheduler.update_connector_output(types.SimpleNamespace(
            finished_recving={"reader"}, invalid_block_ids=invalid,
        ))
    assert [call.args[0] % call.args[1:] for call in log.info.call_args_list] == [line]
    scheduler.shutdown()


@pytest.mark.parametrize("mode,label", [("fast", "CRC32 fast"), ("sha256", "SHA-256"), ("none", "none")])
def test_integrity_labels_name_the_enabled_check(mode, label):
    assert _integrity_label(mode) == label
