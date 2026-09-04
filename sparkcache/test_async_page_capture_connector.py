from __future__ import annotations

import threading
import tempfile
import types
from collections import Counter
from pathlib import Path

import torch
import pytest

import sparkcache.test_spark_context_cache_connector  # noqa: F401

from sparkcache.spark_context_cache_connector import (
    SparkCacheConnectorMetadata,
    SparkContextCacheConnector,
    _ReqPlan,
    configure_async_page_capture_runtime,
)
from sparkcache.test_spark_context_cache_connector import (
    _hybrid_kv_cache_config,
    _make_connector,
)
from sparkcache.streaming.manager_page_runtime import ManagerPageCaptureRuntime


class FakeRuntime:
    def __init__(self) -> None:
        self.submitted: list[tuple[object, int]] = []
        self.preempted: list[str] = []
        self.finished: set[str] = set()
        self.finished_manager_pages: dict[str, int] = {}
        self.submit_error: Exception | None = None

    def finish_without_capture(
        self,
        request_id: str,
        *,
        retained_manager_pages: int = 0,
    ) -> None:
        self.finished.add(request_id)
        self.finished_manager_pages[request_id] = retained_manager_pages

    def submit(self, plan: object, *, producer_stream: int) -> bool:
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append((plan, producer_stream))
        return True

    def preempt(self, request_id: str) -> None:
        self.preempted.append(request_id)

    def take_finished(self, finished_req_ids: set[str]) -> set[str]:
        ready = self.finished & finished_req_ids
        self.finished.difference_update(ready)
        return ready

    def status(self) -> dict[str, object]:
        return {
            "pending_requests": 2,
            "completed_notifications": 1,
            "delayed_requests": 3,
            "retained_manager_pages": 14,
            "oldest_delayed_ms": 125.0,
            "ownership_uncertain": False,
        }

    def quiesce(self) -> None:
        return None

    def shutdown(self) -> bool:
        return True


def test_runtime_contract_requires_terminal_skip_completion(monkeypatch) -> None:
    class RuntimeWithoutTerminalSkipCompletion:
        submit = staticmethod(lambda *_args, **_kwargs: True)
        preempt = staticmethod(lambda *_args, **_kwargs: None)
        take_finished = staticmethod(lambda *_args, **_kwargs: set())
        quiesce = staticmethod(lambda *_args, **_kwargs: None)
        shutdown = staticmethod(lambda *_args, **_kwargs: None)

    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setattr(
            "sparkcache.streaming.manager_page_factory."
            "verify_manager_page_lease_contract",
            lambda _settings: (),
        )
        configure_async_page_capture_runtime(
            lambda _connector: RuntimeWithoutTerminalSkipCompletion()
        )
        try:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                    "spark_cache_publication_schema": "tail-cow-v1",
                    "spark_cache_async_page_capture": "1",
                    "spark_cache_async_page_capture_library": "/tmp/libsparkcache-snapshot.so",
                    "spark_cache_async_page_capture_library_sha256": "a" * 64,
                    "spark_cache_async_page_capture_slot_bytes": "1024",
                },
                tp=1,
                dcp=1,
                kv_cache_config=_hybrid_kv_cache_config(),
            )
            pools = {
                "compressed": torch.zeros((10, 2, 8), dtype=torch.uint8),
                "full": torch.zeros((10, 64, 8), dtype=torch.uint8),
                "state": torch.zeros((10, 4, 16), dtype=torch.float32),
            }

            with pytest.raises(RuntimeError, match="finish_without_capture"):
                connector.register_kv_caches(pools)
        finally:
            configure_async_page_capture_runtime(None)


def test_worker_stats_include_async_capture_ownership(monkeypatch) -> None:
    runtime = FakeRuntime()
    connector = None
    with tempfile.TemporaryDirectory() as directory:
        monkeypatch.setattr(
            "sparkcache.streaming.manager_page_factory."
            "verify_manager_page_lease_contract",
            lambda _settings: (),
        )
        configure_async_page_capture_runtime(lambda _connector: runtime)
        try:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                    "spark_cache_publication_schema": "tail-cow-v1",
                    "spark_cache_async_page_capture": "1",
                    "spark_cache_async_page_capture_library": "/tmp/libsparkcache-snapshot.so",
                    "spark_cache_async_page_capture_library_sha256": "a" * 64,
                    "spark_cache_async_page_capture_slot_bytes": "1024",
                },
                tp=1,
                dcp=1,
                kv_cache_config=_hybrid_kv_cache_config(),
            )
            connector.register_kv_caches(
                {
                    "compressed": torch.zeros((10, 2, 8), dtype=torch.uint8),
                    "full": torch.zeros((10, 64, 8), dtype=torch.uint8),
                    "state": torch.zeros((10, 4, 16), dtype=torch.float32),
                }
            )
            connector._store_inflight = 1

            stats = connector.get_kv_connector_stats()
            report = stats.data["reports"][0]

            assert report["async_capture"] == {
                **runtime.status(),
                "store_inflight": True,
            }
        finally:
            configure_async_page_capture_runtime(None)
            if connector is not None:
                connector.shutdown()


class FakeSparseRing:
    def __init__(self, payload: bytes) -> None:
        self.payload = memoryview(payload)
        self.ready = threading.Event()
        self.active: set[int] = set()

    @property
    def active_ticket_count(self) -> int:
        return len(self.active)

    def submit(self, **kwargs):
        sequence = kwargs["context_sequence"]
        self.active.add(sequence)
        return types.SimpleNamespace(context_sequence=sequence)

    def poll(self, ticket):
        return types.SimpleNamespace(payload=self.payload) if self.ready.is_set() else None

    def claim(self, ticket):
        return types.SimpleNamespace(payload=self.payload)

    def release(self, ticket) -> None:
        self.active.discard(ticket.context_sequence)

    def drain_context(self, context_sequence: int) -> None:
        self.active.discard(context_sequence)

    def shutdown(self) -> None:
        return None


def _connector(plan: _ReqPlan, runtime: FakeRuntime) -> SparkContextCacheConnector:
    connector = SparkContextCacheConnector.__new__(SparkContextCacheConnector)
    connector._store_enabled = True
    connector._streaming_snapshots_enabled = False
    connector._streaming_runtime = None
    connector._async_page_capture_enabled = True
    connector._async_page_capture_runtime = runtime
    connector._async_page_capture_eligible = {plan.request_id}
    connector._store_cv = threading.Condition()
    connector._store_accepting = True
    connector._store_inflight = 0
    connector._held = set()
    connector.counters = Counter()
    connector._get_connector_metadata = lambda: SparkCacheConnectorMetadata(
        plans=[plan]
    )
    connector._snapshot_store = lambda _plan: (_ for _ in ()).throw(
        AssertionError("synchronous snapshot path was called")
    )
    connector._load_cv = threading.Condition()
    connector._finished_load_reqs = set()
    connector._page_base_reads = types.SimpleNamespace(cancel=lambda _request: None)
    connector._emit_page_base_flight_summaries = lambda: None
    connector._restore_flight_followers = {}
    connector._restore_flight_leaders = {}
    connector._restore_flights = {}
    connector._need_load = {plan.request_id: (plan.digest, plan.span_tokens)}
    connector._pending_async_loads = {}
    connector._admitted = {}
    connector._store_progress = {}
    connector._store_token_ids = {}
    connector._store_bases = {}
    connector._store_recurrent_boundaries = {}
    return connector


def test_restore_only_worker_ignores_stale_async_store_plan(monkeypatch) -> None:
    plan = _ReqPlan("request", "a" * 64, 512, (2, 5), True)
    runtime = FakeRuntime()
    connector = _connector(plan, runtime)
    connector._store_enabled = False
    monkeypatch.setattr(
        "sparkcache.spark_context_cache_connector.torch.cuda.current_stream",
        lambda: (_ for _ in ()).throw(
            AssertionError("restore-only mode inspected a CUDA producer stream")
        ),
    )

    connector.wait_for_save()

    assert runtime.submitted == []
    assert connector._store_inflight == 0


def test_wait_for_save_submits_native_pages_without_synchronous_snapshot(
    monkeypatch,
) -> None:
    plan = _ReqPlan(
        "request",
        "a" * 64,
        512,
        (2, 5),
        True,
        block_ids_by_group=((2, 5), (7,)),
        recurrent_boundary_blocks=((1, 7),),
    )
    runtime = FakeRuntime()
    connector = _connector(plan, runtime)
    monkeypatch.setattr(
        "sparkcache.spark_context_cache_connector.torch.cuda.current_stream",
        lambda: types.SimpleNamespace(cuda_stream=91),
    )

    connector.wait_for_save()

    assert runtime.submitted == [(plan, 91)]
    assert connector._store_inflight == 1


def test_all_group_lifetime_ends_only_after_worker_completion() -> None:
    plan = _ReqPlan("request", "a" * 64, 512, (2, 5), True)
    runtime = FakeRuntime()
    connector = _connector(plan, runtime)
    request = types.SimpleNamespace(request_id="request")

    assert connector.request_finished_all_groups(request, ([2, 5], [7])) == (
        True,
        None,
    )
    assert "request" not in connector._need_load
    assert connector.get_finished({"request"})[0] is None
    runtime.finished.add("request")
    assert connector.get_finished({"request"})[0] == {"request"}


def test_preemption_uses_the_runtime_drain_edge() -> None:
    plan = _ReqPlan("request", "a" * 64, 512, (2, 5), True)
    runtime = FakeRuntime()
    connector = _connector(plan, runtime)
    metadata = SparkCacheConnectorMetadata(preempted_request_ids=("request",))

    connector.handle_preemptions(metadata)

    assert runtime.preempted == ["request"]


def test_submission_error_releases_delayed_free_ownership(monkeypatch) -> None:
    plan = _ReqPlan("request", "a" * 64, 512, (2, 5), True)
    runtime = FakeRuntime()
    runtime.submit_error = RuntimeError("rejected")
    connector = _connector(plan, runtime)
    monkeypatch.setattr(
        "sparkcache.spark_context_cache_connector.torch.cuda.current_stream",
        lambda: types.SimpleNamespace(cuda_stream=91),
    )

    connector.wait_for_save()

    assert runtime.preempted == ["request"]
    assert connector._store_inflight == 0


def test_async_full_page_capture_publishes_authenticated_page_delta() -> None:
    class FullAttentionSpec:
        block_size = 256
        storage_block_size = 256
        page_size_bytes = 8

    class MambaSpec:
        block_size = 256
        storage_block_size = 256
        page_size_bytes = 8
        mamba_cache_mode = "align"
        tokens_per_state = 256
        num_speculative_blocks = 0
        num_prefill_checkpoint_blocks = 1

    cache_config = types.SimpleNamespace(
        num_blocks=10,
        kv_cache_groups=(
            types.SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(),
                is_eagle_group=False,
                layer_names=("full",),
            ),
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(),
                is_eagle_group=False,
                layer_names=("state",),
            ),
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        connector = _make_connector(
            Path(directory),
            0,
            extra_config={
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_publication_schema": "tail-cow-v1",
            },
            tp=1,
            dcp=1,
            kv_cache_config=cache_config,
        )
        connector.register_kv_caches(
            {
                "full": torch.arange(80, dtype=torch.uint8).reshape(10, 1, 8),
                "state": torch.arange(80, dtype=torch.uint8).reshape(10, 1, 8),
            }
        )
        tokens = tuple(range(512))
        base_digest = connector._digest(list(tokens), 256)
        connector._store_one(
            _ReqPlan(
                "base",
                base_digest,
                256,
                (3,),
                True,
                block_ids_by_group=((3,), (4,)),
                token_ids=tokens[:256],
            )
        )
        extension = _ReqPlan(
            "extension",
            connector._digest(list(tokens), 512),
            512,
            (3, 5),
            True,
            block_ids_by_group=((3, 5), (0, 7)),
            token_ids=tokens,
            base_context_digest=base_digest,
            base_span_tokens=256,
            recurrent_boundary_blocks=((1, 7),),
        )
        captured = connector._snapshot_hybrid_store(extension)

        connector._store_inflight = 1
        connector._complete_async_page_capture(
            extension,
            captured.encoded_pages,
            captured.block_counts,
        )
        assert connector.wait_for_pending_stores(timeout=5)

        lookup = connector._store.lookup(
            connector._identity(0), extension.digest
        )
        assert lookup.is_hit, lookup.reason
        assert lookup.root_kind == "page_delta"
        restored = connector._store.restore_page_snapshot(
            lookup,
            layout=connector._page_layout,
            result_block_counts=captured.block_counts,
            result_boundary_tokens=extension.span_tokens,
        )
        assert restored == captured.encoded_pages
        connector.shutdown()


def test_sparse_async_capture_publishes_restorable_page_delta(monkeypatch) -> None:
    class FullAttentionSpec:
        block_size = 256
        storage_block_size = 256
        page_size_bytes = 8

    class MambaSpec:
        block_size = 256
        storage_block_size = 256
        page_size_bytes = 8
        mamba_cache_mode = "align"
        tokens_per_state = 256
        num_speculative_blocks = 0
        num_prefill_checkpoint_blocks = 1

    cache_config = types.SimpleNamespace(
        num_blocks=10,
        kv_cache_groups=(
            types.SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(),
                is_eagle_group=False,
                layer_names=("full",),
            ),
            types.SimpleNamespace(
                kv_cache_spec=MambaSpec(),
                is_eagle_group=False,
                layer_names=("state",),
            ),
        ),
    )
    with tempfile.TemporaryDirectory() as directory:
        connector = _make_connector(
            Path(directory),
            0,
            extra_config={
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_publication_schema": "tail-cow-v2",
            },
            tp=1,
            dcp=1,
            kv_cache_config=cache_config,
        )
        connector.register_kv_caches(
            {
                "full": torch.arange(80, dtype=torch.uint8).reshape(10, 1, 8),
                "state": torch.arange(80, dtype=torch.uint8).reshape(10, 1, 8),
            }
        )
        tokens = tuple(range(512))
        base_digest = connector._digest(list(tokens), 256)
        connector._store_one(
            _ReqPlan(
                "base",
                base_digest,
                256,
                (3,),
                True,
                block_ids_by_group=((3,), (4,)),
                token_ids=tokens[:256],
            )
        )
        extension = _ReqPlan(
            "sparse-extension",
            connector._digest(list(tokens), 512),
            512,
            (3, 5),
            True,
            block_ids_by_group=((3, 5), (0, 7)),
            token_ids=tokens,
            base_context_digest=base_digest,
            base_span_tokens=256,
            recurrent_boundary_blocks=((1, 7),),
        )
        expected = connector._snapshot_hybrid_store(extension)
        ring = FakeSparseRing(bytes(range(40, 48)) + bytes(range(56, 64)))
        runtime = ManagerPageCaptureRuntime(
            connector,
            ring=ring,
            progress_poll_seconds=0.001,
            progress_thread_initializer=lambda: None,
        )
        connector._store_inflight = 1
        restore_calls = 0
        restore_page_snapshot = connector._store.restore_page_snapshot

        def counted_restore(*args, **kwargs):
            nonlocal restore_calls
            restore_calls += 1
            return restore_page_snapshot(*args, **kwargs)

        connector._store.restore_page_snapshot = counted_restore
        monkeypatch.setattr(
            "sparkcache.spark_context_cache_connector."
            "materialize_page_extension_capture",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("sparse capture was materialized")
            ),
        )

        assert runtime.submit(extension, producer_stream=91)
        ring.ready.set()
        assert runtime.wait_idle(timeout=1)
        assert connector.wait_for_pending_stores(timeout=5)

        lookup = connector._store.lookup(
            connector._identity(0), extension.digest
        )
        assert lookup.is_hit, lookup.reason
        assert lookup.root_kind == "page_delta"
        restored = connector._store.restore_page_snapshot(
            lookup,
            layout=connector._page_layout,
            result_block_counts=expected.block_counts,
            result_boundary_tokens=extension.span_tokens,
        )
        assert restored == expected.encoded_pages
        # One reconstruction read during publication, followed by the test's
        # result-delta and embedded-base restore calls. The commit path must
        # not read the already verified base a second time.
        assert restore_calls == 3
        runtime.shutdown()
        connector.shutdown()
