"""Operator logs preserve metrics and isolate verbose diagnostics at DEBUG."""

from unittest import mock

import pytest

from sparkcache.test_spark_context_cache_connector import _make_connector, connector_module
from sparkcache.spark_context_cache_cuda_hybrid_restore import _integrity_label


def test_changed_counters_survive_reset_without_dropping_metrics():
    stats = connector_module.SparkCacheStats(data={"reports": []})
    report = {"rank": 0, "held": [], "counters": {"restore_hit": 2, "load_failed": 0}}
    stats.aggregate(connector_module.SparkCacheStats(data={"reports": [report]}))
    assert stats.format_log_lines() == ("sparkcache: counters counter_restore_hit=2 ranks=1",)
    assert stats.reduce()["sparkcache_counter_load_failed"] == 0
    stats.reset()
    stats.aggregate(connector_module.SparkCacheStats(data={"reports": [report]}))
    assert stats.format_log_lines() == ()
    report["counters"]["restore_hit"] = 3
    assert stats.format_log_lines() == ("sparkcache: counters counter_restore_hit=3",)
    report["counters"]["restore_hit"] = 0
    assert stats.format_log_lines() == ("sparkcache: counters counter_restore_hit=0",)
    assert stats.reduce()["sparkcache_counter_restore_hit"] == 0


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


@pytest.mark.parametrize("mode,label", [("fast", "CRC32 fast"), ("sha256", "SHA-256"), ("none", "none")])
def test_integrity_labels_name_the_enabled_check(mode, label):
    assert _integrity_label(mode) == label
