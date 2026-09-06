"""Forced maintenance respects cooldown without repeating survivor scans."""

from unittest import mock

from sparkcache import test_spark_context_cache_connector as fixtures
from sparkcache.persistent_context_cache.cache_manifest import (
    CapacityPolicy,
    MaintenanceReport,
)


def test_forced_post_commit_cooldown_does_not_reconcile_inventory(tmp_path):
    connector = fixtures.AsyncRestoreTests()._cohort_connector(tmp_path)
    connector._capacity_policy = CapacityPolicy(
        max_bytes=10, low_watermark_bytes=8, maintenance_interval_ms=1000
    )
    connector._capacity_estimated_bytes = 11
    connector._capacity_status["capacity_satisfied"] = False
    try:
        with (
            mock.patch.object(
                connector._store,
                "maintain",
                return_value=MaintenanceReport(
                    capacity_satisfied=False, skipped_cooldown=True
                ),
            ),
            mock.patch.object(connector, "_reconcile_held_capacity") as reconcile,
        ):
            report = connector._maintain_capacity(force=True)
        assert report.skipped_cooldown
        assert not connector._capacity_status["capacity_satisfied"]
        assert connector._capacity_estimated_bytes == 11
        assert connector.counters["capacity_skipped_cooldown"] == 1
        reconcile.assert_not_called()
    finally:
        connector.shutdown()


def test_pacing_totals_aggregate_physical_ranks():
    from sparkcache.spark_context_cache_connector import SparkCacheStats

    stats = SparkCacheStats(
        data={
            "reports": [
                {
                    "rank": rank,
                    "capacity": {
                        "maintenance_deletion_attempts": 128,
                        "maintenance_budget_exhausted": rank,
                        "maintenance_skipped_cooldown": 2,
                    },
                }
                for rank in range(4)
            ]
        }
    )
    reduced = stats.reduce()
    assert reduced["sparkcache_maintenance_deletion_attempts"] == 512
    assert reduced["sparkcache_maintenance_budget_exhausted"] == 6
    assert reduced["sparkcache_maintenance_skipped_cooldown"] == 8
