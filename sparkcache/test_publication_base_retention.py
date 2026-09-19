"""Bounded publication-base retention across asynchronous capture and eviction."""

from sparkcache.request_cache_scope import UNSALTED_SCOPE

import os
from dataclasses import replace
from types import SimpleNamespace
from pathlib import Path

import pytest

from sparkcache.persistent_context_cache.cache_manifest import CapacityPolicy
from sparkcache.spark_context_cache_hybrid import PageGroup, PageLayer, PageLayout, encode_page_snapshot
from sparkcache.test_async_page_capture_connector import FakeRuntime
from sparkcache.test_spark_context_cache_connector import _hybrid_kv_cache_config, _make_connector
from sparkcache.spark_context_cache_connector import SparkCacheConnectorMetadata, _ReqPlan


@pytest.fixture
def retained_base(tmp_path, monkeypatch):
    connector = _make_connector(tmp_path, 0, tp=1, dcp=1,
        extra_config={"spark_cache_model_profile": "deepseek-v4-fp8-hma",
                      "spark_cache_publication_schema": "tail-cow-v2"},
        kv_cache_config=_hybrid_kv_cache_config())
    layout = PageLayout((PageGroup(256, (PageLayer("page", "u8", (64,), 64),)),))
    connector._page_layout = layout
    connector._group_block_counts_for_span = lambda span: (span // 256,)
    identity = connector._identity(0)
    tokens = tuple(range(768))
    base = connector._digest(tokens, 256)
    result = connector._digest(tokens, 512)
    connector._store.commit_page_snapshot(identity=identity, context_digest=base, span_tokens=256,
        snapshot=encode_page_snapshot(layout, (1,), {"page": b"a" * 64}))
    connector._held.add(base)
    policy = CapacityPolicy(max_bytes=10**9, low_watermark_bytes=10**9)
    base_bytes = connector._store.maintain(policy).bytes_after
    os.utime(connector._store._manifest_path(identity, base), (1, 1))
    connector._capacity_policy = policy
    connector._capacity_status["capacity_satisfied"] = True
    runtime = FakeRuntime()
    connector._async_page_capture_enabled = True
    connector._async_page_capture_runtime = runtime
    plan = _ReqPlan("extension", result, 512, (0, 1), True,
        block_ids_by_group=((0, 1),), token_ids=tokens[:512],
        base_context_digest=base, base_span_tokens=256, request_scope=UNSALTED_SCOPE)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    monkeypatch.setattr("torch.cuda.current_stream", lambda: SimpleNamespace(cuda_stream=123))
    yield connector, runtime, plan, identity, layout, base_bytes
    connector._finish_store(plan.digest, committed=False)
    connector.shutdown()


def test_queued_extension_base_survives_capacity_pressure(retained_base):
    connector, runtime, plan, identity, layout, base_bytes = retained_base
    connector.wait_for_save()
    assert runtime.submitted[0][0].base_context_digest == plan.base_context_digest
    connector._store.commit_page_snapshot(identity=identity, context_digest="b" * 64, span_tokens=256,
        snapshot=encode_page_snapshot(layout, (1,), {"page": b"b" * 64}))
    connector._capacity_policy = CapacityPolicy(max_bytes=base_bytes + 1, low_watermark_bytes=base_bytes)
    report = connector._maintain_capacity(force=True)
    assert report.capacity_satisfied
    assert connector._store.lookup(identity, plan.base_context_digest).is_hit
    assert not connector._store.lookup(identity, "b" * 64).is_hit


def test_completed_result_is_protected_until_capacity_reconciliation(retained_base):
    connector, _runtime, plan, identity, layout, *_ = retained_base
    connector.wait_for_save()
    connector._store.commit_page_extension(identity=identity,
        base_context_digest=plan.base_context_digest, token_ids=plan.token_ids,
        identity_salt=connector._scope_salt(plan.request_scope), layout=layout,
        base_block_counts=(1,), result_block_counts=(2,),
        base_boundary_tokens=256, result_boundary_tokens=512,
        result_snapshot=encode_page_snapshot(layout, (2,), {"page": b"a" * 128}))
    connector._capacity_policy = CapacityPolicy(max_bytes=1, low_watermark_bytes=1)
    report = connector._maintain_capacity(force=True)
    assert not report.capacity_satisfied
    assert connector._store.lookup(identity, plan.digest).is_hit
    connector._finish_store(plan.digest, committed=False)
    assert connector._maintain_capacity(force=True).capacity_satisfied


def test_missing_base_selects_full_capture_before_any_tail_is_submitted(retained_base):
    connector, runtime, plan, *_ = retained_base
    connector._held.discard(plan.base_context_digest)
    connector.wait_for_save()
    submitted = runtime.submitted[0][0]
    assert submitted.base_context_digest == ""
    assert submitted.base_span_tokens == 0
    assert submitted.digest == plan.digest
    assert submitted.token_ids == plan.token_ids


@pytest.mark.parametrize("committed", [False, True])
def test_terminal_store_releases_retention_and_capacity_can_recover(retained_base, committed):
    connector, _runtime, plan, identity, *_ = retained_base
    connector.wait_for_save()
    connector._capacity_policy = CapacityPolicy(max_bytes=1, low_watermark_bytes=1)
    report = connector._maintain_capacity(force=True)
    assert not report.capacity_satisfied
    assert connector._store.lookup(identity, plan.base_context_digest).is_hit
    # The single inflight admission prevents another protected publication.
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[replace(plan, scope_binding="", digest="c" * 64)]))
    connector.wait_for_save()
    assert connector._store_inflight == 1
    connector._finish_store(plan.digest, committed=committed)
    report = connector._maintain_capacity(force=True)
    assert report.capacity_satisfied
    assert not connector._store.lookup(identity, plan.base_context_digest).is_hit


def test_capture_pin_registration_performs_no_filesystem_read(retained_base, monkeypatch):
    connector, runtime, plan, *_ = retained_base
    monkeypatch.setattr(Path, "read_bytes", lambda *args: pytest.fail("model callback read the filesystem"))
    monkeypatch.setattr(connector._store, "lookup", lambda *args, **kw: pytest.fail("model callback looked up a root"))
    connector.wait_for_save()
    assert runtime.submitted[0][0] == plan
    assert len(connector._publication_base_pins) == 1


def test_busy_capacity_guard_uses_full_capture_without_waiting(retained_base, monkeypatch):
    connector, runtime, plan, *_ = retained_base

    class BusyLock:
        def acquire(self, *, blocking):
            assert blocking is False
            return False

    with monkeypatch.context() as local:
        local.setattr(connector, "_capacity_lock", BusyLock())
        connector.wait_for_save()
    assert runtime.submitted[0][0].base_context_digest == ""
    assert not connector._publication_base_pins
    assert connector.counters["publication_base_full_capture_fallback"] == 1


def test_unsatisfied_capacity_does_not_admit_another_base_pin(retained_base):
    connector, runtime, _plan, *_ = retained_base
    connector._capacity_status["capacity_satisfied"] = False
    connector.wait_for_save()
    assert runtime.submitted[0][0].base_context_digest == ""
    assert not connector._publication_base_pins


def test_capture_submission_failure_releases_base_pin(retained_base):
    connector, runtime, _plan, *_ = retained_base
    runtime.submit_error = ValueError("capture exceeds bounded ring capacity")
    connector.wait_for_save()
    assert not connector._publication_base_pins
    assert connector._store_inflight == 0


def test_capture_preemption_releases_base_pin(retained_base):
    connector, _runtime, plan, *_ = retained_base
    connector.wait_for_save()
    connector._abort_async_page_capture(plan.digest, "request was preempted")
    assert not connector._publication_base_pins
    assert connector._store_inflight == 0


def test_shutdown_drain_releases_base_pin(retained_base):
    connector, runtime, plan, *_ = retained_base
    connector.wait_for_save()
    runtime.quiesce = lambda: connector._abort_async_page_capture(plan.digest, "capture shutdown")
    connector.shutdown()
    assert not connector._publication_base_pins
    assert connector._store_inflight == 0


def test_invalid_protected_metadata_does_not_block_cleanup(retained_base):
    connector, _runtime, plan, identity, *_ = retained_base
    connector.wait_for_save()
    connector._store._manifest_path(identity, plan.base_context_digest).write_bytes(b"corrupt")
    report = connector._maintain_capacity(force=True)
    assert report.manifests_evicted == 1
    assert not connector._store.lookup(identity, plan.base_context_digest).is_hit


def test_publication_releases_base_before_post_commit_maintenance(retained_base, monkeypatch):
    from sparkcache.spark_context_cache_connector import _HybridStoreSnapshot

    connector, _runtime, plan, identity, layout, _base_bytes = retained_base
    connector.wait_for_save()
    snapshot = _HybridStoreSnapshot(plan=plan, rank=0, identity=identity, positions=(),
        encoded_pages=encode_page_snapshot(layout, (2,), {"page": b"a" * 128}), block_counts=(2,))
    checked = []
    original = connector._post_commit_was_evicted_locked

    def post_commit(*args, **kwargs):
        assert not connector._publication_base_pins
        assert connector._store.lookup(identity, plan.digest).is_hit
        checked.append(True)
        return original(*args, **kwargs)

    monkeypatch.setattr(connector, "_post_commit_was_evicted_locked", post_commit)
    connector._store_queue.put(snapshot)
    connector._store_queue.put(None)
    connector._store_worker_main()
    assert checked == [True]
    assert connector._store_inflight == 0
