"""Failed external restores must not invalidate another request's null padding."""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from sparkcache.runtime_patches.test_glm53_da4d7be_hma_recovery import (
    _patched_scheduler_class,
)
from sparkcache.test_spark_context_cache_connector import (
    SparkCacheConnectorMetadata,
    _ReqPlan,
    _drain,
    _make_connector,
    _make_pools,
)


@pytest.mark.parametrize("raise_failure", [False, True])
@pytest.mark.parametrize("evict_blocks", [False, True])
@pytest.mark.parametrize("stopping", [False, True])
def test_failed_restore_preserves_unrelated_hybrid_request(
    tmp_path, raise_failure, evict_blocks, stopping
):
    connector = _make_connector(tmp_path / "cache", 0, 64)
    connector.register_kv_caches(_make_pools(32, 64))
    failed_groups = ((0, 0, 11), (21, 22))
    unrelated_groups = ((0, 0, 12), (31, 32))
    connector.bind_connector_metadata(
        SparkCacheConnectorMetadata(
            plans=[
                _ReqPlan(
                    "failed",
                    "f" * 64,
                    8192,
                    failed_groups[0],
                    False,
                    block_ids_by_group=failed_groups,
                )
            ]
        )
    )
    loader = mock.Mock(
        side_effect=RuntimeError("payload checksum mismatch")
        if raise_failure
        else None,
        return_value=False,
    )
    try:
        connector._load_stop_requested = stopping
        with mock.patch.object(connector, "_load_one", loader):
            connector.start_load_kv(None)
            assert _drain(connector) == {"failed"}
        invalid = connector.get_block_ids_with_load_errors()
        assert connector.counters["load_failed"] == 1
        assert connector.get_block_ids_with_load_errors() == set()

        # Execute the shipped HMA recovery method, which compares complete
        # per-request block tables. Null block 0 occurs in unrelated tables
        # too, and therefore cannot identify ownership of failed payloads.
        scheduler = _patched_scheduler_class()()
        scheduler.block_size = 256
        tables = {"failed": failed_groups, "unrelated": unrelated_groups}
        scheduler.kv_cache_manager = SimpleNamespace(get_block_ids=tables.__getitem__)
        failed = SimpleNamespace(request_id="failed", num_computed_tokens=8192)
        unrelated = SimpleNamespace(request_id="unrelated", num_computed_tokens=4096)
        affected, recomputed, _ = scheduler._update_requests_with_invalid_blocks(
            [failed, unrelated], invalid, {}, evict_blocks
        )

        assert affected == {"failed"}
        assert recomputed == 8192
        assert failed.num_computed_tokens == 0
        assert unrelated.num_computed_tokens == 4096
        assert invalid == {11, 21, 22}
    finally:
        connector.shutdown()
