"""Publication age and maintenance activity remain observable without I/O."""

from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from sparkcache import test_publication_base_retention as retention_tests
from sparkcache import spark_context_cache_connector as connector_module
from sparkcache.persistent_context_cache.cache_manifest import MaintenanceReport


retained_base = retention_tests.retained_base


def work_report(connector):
    return connector.get_kv_connector_stats().data["reports"][0]["publication_work"]


def test_admission_age_includes_capture_and_clears_after_commit(retained_base, monkeypatch):
    connector, _runtime, plan, *_ = retained_base
    now = [1_000_000_000]
    monkeypatch.setattr(connector_module.time, "perf_counter_ns", lambda: now[0])
    connector.wait_for_save()
    now[0] += 125_000_000
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("metrics read the filesystem"))
    assert work_report(connector) == {"pending": 1, "oldest_pending_ms": 125.0, "maintenance_active": False}
    connector._finish_store(plan.digest, committed=True)
    now[0] += 1_000_000_000
    assert work_report(connector) == {"pending": 0, "oldest_pending_ms": 0.0, "maintenance_active": False}


@pytest.mark.parametrize("terminal", ["failure", "abort", "shutdown"])
def test_terminal_publication_clears_pending_age(retained_base, monkeypatch, terminal):
    connector, runtime, plan, *_ = retained_base
    now = [1_000_000_000]
    monkeypatch.setattr(connector_module.time, "perf_counter_ns", lambda: now[0])
    connector.wait_for_save()
    now[0] += 2_000_000_000
    assert work_report(connector)["oldest_pending_ms"] == 2000.0
    if terminal == "failure":
        connector._finish_store(plan.digest, committed=False, error=ValueError("publication failed"))
    elif terminal == "abort":
        connector._abort_async_page_capture(plan.digest, "capture aborted")
    else:
        runtime.quiesce = lambda: connector._abort_async_page_capture(plan.digest, "shutdown drain")
        connector.shutdown()
    assert connector._store_pending_started_ns is None
    assert work_report(connector)["pending"] == 0
    assert work_report(connector)["oldest_pending_ms"] == 0.0


@pytest.mark.parametrize("failed", [False, True])
def test_maintenance_activity_includes_reconciliation_and_clears(retained_base, monkeypatch, failed):
    connector, *_ = retained_base
    stages = []

    def maintain(*args, **kwargs):
        assert work_report(connector)["maintenance_active"] is True
        stages.append("scan")
        if failed:
            raise OSError("scan failed")
        return MaintenanceReport(capacity_satisfied=True)

    def reconcile():
        assert work_report(connector)["maintenance_active"] is True
        stages.append("reconcile")

    monkeypatch.setattr(connector._store, "maintain", maintain)
    monkeypatch.setattr(connector, "_reconcile_held_capacity", reconcile)
    connector._maintain_capacity(force=True)
    assert stages == ["scan", "reconcile"]
    assert work_report(connector)["maintenance_active"] is False


def test_reconciliation_exception_clears_maintenance_activity(retained_base, monkeypatch):
    connector, *_ = retained_base
    monkeypatch.setattr(connector._store, "maintain", lambda *args, **kw: MaintenanceReport())
    monkeypatch.setattr(connector, "_reconcile_held_capacity",
                        lambda: (_ for _ in ()).throw(RuntimeError("reconciliation failed")))
    with pytest.raises(RuntimeError, match="reconciliation failed"):
        connector._maintain_capacity(force=True)
    assert work_report(connector)["maintenance_active"] is False


def test_report_does_not_wait_for_a_blocked_capacity_scan(retained_base, monkeypatch):
    connector, *_ = retained_base
    entered, release, reported = threading.Event(), threading.Event(), threading.Event()
    observed = []

    def maintain(*args, **kwargs):
        entered.set()
        assert release.wait(timeout=5)
        return MaintenanceReport(capacity_satisfied=True)

    def report():
        observed.append(work_report(connector))
        reported.set()

    monkeypatch.setattr(connector._store, "maintain", maintain)
    maintenance = threading.Thread(target=lambda: connector._maintain_capacity(force=True))
    reader = threading.Thread(target=report)
    maintenance.start()
    try:
        assert entered.wait(timeout=2)
        reader.start()
        assert reported.wait(timeout=0.5), "metrics waited behind capacity maintenance"
        assert observed[0]["maintenance_active"] is True
    finally:
        release.set()
        maintenance.join(timeout=2)
        if reader.ident is not None:
            reader.join(timeout=2)


def test_shutdown_timeout_does_not_hide_pending_publication(retained_base, monkeypatch):
    connector, _runtime, _plan, *_ = retained_base
    now = [1_000_000_000]
    monkeypatch.setattr(connector_module.time, "perf_counter_ns", lambda: now[0])
    connector.wait_for_save()
    monkeypatch.setattr(connector, "wait_for_pending_stores", lambda timeout: False)
    connector._store_thread = SimpleNamespace(is_alive=lambda: True, join=lambda **kw: None)
    now[0] += 2_000_000_000
    connector.shutdown()
    assert work_report(connector)["pending"] == 1
    assert work_report(connector)["oldest_pending_ms"] == 2000.0


def test_existing_report_age_is_a_snapshot_until_worker_reports_again(retained_base, monkeypatch):
    connector, _runtime, _plan, *_ = retained_base
    now = [1_000_000_000]
    monkeypatch.setattr(connector_module.time, "perf_counter_ns", lambda: now[0])
    connector.wait_for_save()
    now[0] += 1_000_000_000
    snapshot = connector.get_kv_connector_stats()
    now[0] += 2_000_000_000
    assert snapshot.reduce()["sparkcache_publication_oldest_pending_ms"] == 1000.0
    assert work_report(connector)["oldest_pending_ms"] == 3000.0


def test_work_reduction_sums_rank_slots_takes_max_age_and_preserves_reports():
    incoming = {"rank": 1, "held_count": 2,
                "publication_work": {"pending": 1, "oldest_pending_ms": 2500, "maintenance_active": True}}
    stats = connector_module.SparkCacheStats(data={"reports": [
        {"rank": 0, "held_count": 3,
         "publication_work": {"pending": 1, "oldest_pending_ms": 1250, "maintenance_active": False}},
    ]})
    stats.aggregate(connector_module.SparkCacheStats(data={"reports": [incoming]}))
    incoming["publication_work"]["pending"] = 0
    reduced = stats.reduce()
    assert reduced["sparkcache_entries"] == 5
    assert reduced["sparkcache_publication_pending_rank_slots"] == 2
    assert reduced["sparkcache_publication_oldest_pending_ms"] == 2500
    assert reduced["sparkcache_maintenance_active_ranks"] == 1
    assert any("publication_work" in line for line in stats.format_log_lines())
    legacy = connector_module.SparkCacheStats(data={"reports": [{"rank": 2, "held": []}]})
    assert legacy.reduce() == {"sparkcache_ranks": 1, "sparkcache_entries": 0}


def test_prometheus_work_gauges_convert_age_to_seconds_and_reset():
    gauges = {}

    class Gauge:
        def __init__(self, *, name, **kwargs):
            gauges[name] = self

        def labels(self, *args):
            return self

        def set(self, value):
            self.value = value

    metrics = connector_module.SparkCachePromMetrics(SimpleNamespace(), {object: Gauge}, [], {0: []})
    metrics.observe({"reports": [{"rank": 0, "held": [], "publication_work": {
        "pending": 1, "oldest_pending_ms": 1250, "maintenance_active": True,
    }}]})
    assert gauges["vllm:sparkcache_publication_pending_rank_slots"].value == 1
    assert gauges["vllm:sparkcache_publication_oldest_pending_seconds"].value == 1.25
    assert gauges["vllm:sparkcache_maintenance_active_ranks"].value == 1
    metrics.observe({"reports": [{"rank": 0, "held": []}]})
    assert gauges["vllm:sparkcache_publication_pending_rank_slots"].value == 0
    assert gauges["vllm:sparkcache_publication_oldest_pending_seconds"].value == 0
    assert gauges["vllm:sparkcache_maintenance_active_ranks"].value == 0


def test_restore_decision_counters_reduce_across_ranks():
    stats = connector_module.SparkCacheStats(data={"reports": [
        {"rank": 0, "held": [],
         "counters": {"restore_hit": 2, "restore_skip_local_prefix": 5}},
    ]})
    stats.aggregate(connector_module.SparkCacheStats(data={"reports": [
        {"rank": 1, "held_count": 1,
         "counters": {"restore_hit": 3, "restore_skip_local_prefix": 1}},
    ]}))
    reduced = stats.reduce()
    assert reduced["sparkcache_counter_restore_hit"] == 5
    assert reduced["sparkcache_counter_restore_skip_local_prefix"] == 6
    # A report without a counters section reduces as zeros, not as an error.
    legacy = connector_module.SparkCacheStats(data={"reports": [{"rank": 2, "held": []}]})
    legacy.aggregate(stats)
    assert legacy.reduce()["sparkcache_counter_restore_hit"] == 5


def test_prometheus_decision_counter_gauges_sum_ranks_and_reset():
    gauges = {}

    class Gauge:
        def __init__(self, *, name, **kwargs):
            gauges[name] = self

        def labels(self, *args):
            return self

        def set(self, value):
            self.value = value

    metrics = connector_module.SparkCachePromMetrics(SimpleNamespace(), {object: Gauge}, [], {0: []})
    metrics.observe({"reports": [
        {"rank": 0, "held": [], "counters": {"restore_hit": 2,
                                             "restore_skip_local_prefix": 5}},
        {"rank": 1, "held": [], "counters": {"restore_hit": 3}},
    ]})
    assert gauges["vllm:sparkcache_restore_hits"].value == 5
    assert gauges["vllm:sparkcache_restore_skip_local_prefix"].value == 5
    # Ranks reporting without counters (or without the key) count as zero.
    metrics.observe({"reports": [{"rank": 0, "held": []}]})
    assert gauges["vllm:sparkcache_restore_hits"].value == 0
