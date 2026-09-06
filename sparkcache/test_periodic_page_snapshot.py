"""Opt-in token-bucket selection of complete manager-page captures."""

from dataclasses import replace
from pathlib import Path

import pytest

from sparkcache import spark_context_cache_config as config
from sparkcache.test_spark_context_cache_config import _make_vllm_config
from sparkcache import test_publication_base_retention as retention_tests
from sparkcache.test_async_page_capture_connector import FakeSparseRing
from sparkcache.spark_context_cache_connector import SparkCacheConnectorMetadata
from sparkcache.streaming.manager_page_runtime import ManagerPageCaptureRuntime
from sparkcache.spark_context_cache_hybrid import PageGroup, PageLayer, PageLayout, encode_page_snapshot


KEY = "spark_cache_page_snapshot_interval_tokens"
ENV = "SPARK_CONTEXT_CACHE_PAGE_SNAPSHOT_INTERVAL_TOKENS"
retained_base = retention_tests.retained_base


def parse(extra=None):
    vllm, transfer = _make_vllm_config(extra)
    return config.parse_connector_config(vllm, transfer, None)


def test_interval_defaults_off_and_does_not_change_cache_identity(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    disabled = parse()
    enabled = parse({KEY: 16384})
    assert disabled.page_snapshot_interval_tokens == 0
    assert enabled.page_snapshot_interval_tokens == 16384
    assert disabled.build_identity(0, 0).to_wire() == enabled.build_identity(0, 0).to_wire()


def test_interval_extra_config_overrides_environment(monkeypatch):
    monkeypatch.setenv(ENV, "16384")
    assert parse().page_snapshot_interval_tokens == 16384
    assert parse({KEY: 8192}).page_snapshot_interval_tokens == 8192
    assert parse({KEY: 0}).page_snapshot_interval_tokens == 0


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "1.5", "bad", None])
def test_interval_rejects_invalid_settings(value):
    with pytest.raises(RuntimeError, match=KEY):
        parse({KEY: value})


@pytest.mark.parametrize("interval,base,result,full", [
    (0, 14336, 16384, False),
    (16384, 12288, 14336, False),
    (16384, 14336, 16384, True),
    (16384, 16384, 18432, False),
    (16384, 14336, 49152, True),
])
def test_periodic_selection_crosses_buckets_without_history_reads(
    retained_base, monkeypatch, interval, base, result, full
):
    connector, runtime, original, *_ = retained_base
    connector._page_snapshot_interval_tokens = interval
    plan = replace(original, base_span_tokens=base, span_tokens=result)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("capture policy read the filesystem"))
    connector.wait_for_save()
    submitted = runtime.submitted[0][0]
    assert submitted.base_context_digest == ("" if full else plan.base_context_digest)
    assert submitted.base_span_tokens == (0 if full else base)
    assert submitted.token_ids == plan.token_ids
    assert submitted.digest == plan.digest
    assert connector.counters.get("publication_periodic_full_capture_selected", 0) == int(full)
    assert connector.counters.get("publication_base_full_capture_fallback", 0) == 0
    assert bool(connector._publication_base_pins) is (not full)


def test_periodic_capture_submits_complete_groups_and_respects_ring_rejection(retained_base):
    connector, _runtime, plan, *_ = retained_base
    plan = replace(plan, block_ids_by_group=((0, 1), (2, 3, 4)))
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    connector._page_layout = PageLayout((
        PageGroup(256, (PageLayer("attention", "u8", (64,), 64),)),
        PageGroup(1, (PageLayer("recurrent", "u8", (8,), 8),)),
    ))
    connector._page_snapshot_interval_tokens = 512
    connector._select_group_blocks_for_span = lambda groups, *args, **kw: groups
    submitted = []

    class BoundedRing:
        active_ticket_count = 0

        def submit(self, **kwargs):
            submitted.append(kwargs)
            return None

        def shutdown(self):
            return None

    runtime = ManagerPageCaptureRuntime(connector, ring=BoundedRing(),
        progress_thread_initializer=lambda: None)
    connector._async_page_capture_runtime = runtime
    connector.wait_for_save()
    assert submitted[0]["logical_start"] == 0
    assert submitted[0]["physical_pages_by_group"] == plan.group_block_ids
    assert connector._store_inflight == 0
    assert not connector._publication_base_pins
    assert connector.counters["async_page_capture_aborted"] == 1
    assert connector.counters["publication_periodic_full_capture_selected"] == 1
    assert runtime.take_finished({plan.request_id}) == {plan.request_id}


def test_already_complete_snapshot_is_not_counted_as_periodic(retained_base):
    connector, runtime, plan, *_ = retained_base
    connector._page_snapshot_interval_tokens = 512
    plan = replace(plan, base_context_digest="", base_span_tokens=0)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    connector.wait_for_save()
    assert runtime.submitted[0][0] == plan
    assert connector.counters.get("publication_periodic_full_capture_selected", 0) == 0


def test_periodic_full_capture_publishes_independent_existing_format(retained_base, monkeypatch):
    connector, _runtime, plan, identity, layout, *_ = retained_base
    connector._page_snapshot_interval_tokens = 512
    connector._select_group_blocks_for_span = lambda groups, *args, **kw: groups
    ring = FakeSparseRing(b"a" * 128)
    ring.ready.set()
    connector._async_page_capture_runtime = ManagerPageCaptureRuntime(
        connector, ring=ring, progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None)
    monkeypatch.setattr(connector._store, "commit_page_extension",
                        lambda **kw: pytest.fail("periodic capture used a dependent publication"))
    connector.wait_for_save()
    assert connector.wait_for_pending_stores(timeout=5)
    lookup = connector._store.lookup(identity, plan.digest, verify_chunks=False)
    assert lookup.is_hit and lookup.root_kind == "page_snapshot"
    restored = connector._store.restore_page_snapshot(lookup, layout=layout,
        result_block_counts=(2,), result_boundary_tokens=512)
    assert restored == encode_page_snapshot(layout, (2,), {"page": b"a" * 128})
    assert connector.counters["publication_periodic_full_capture_selected"] == 1
