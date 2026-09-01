from __future__ import annotations

import threading
import types
from collections import Counter

import sparkcache.test_spark_context_cache_connector  # noqa: F401

from sparkcache.spark_context_cache_connector import (
    SparkCacheConnectorMetadata,
    SparkContextCacheConnector,
    _ReqPlan,
)


class FakeRuntime:
    def __init__(self) -> None:
        self.submitted: list[tuple[object, int]] = []
        self.preempted: list[str] = []
        self.finished: set[str] = set()
        self.submit_error: Exception | None = None

    def submit(self, plan: object, *, producer_stream: int) -> bool:
        if self.submit_error is not None:
            raise self.submit_error
        self.submitted.append((plan, producer_stream))
        return True

    def preempt(self, request_id: str) -> None:
        self.preempted.append(request_id)

    def take_finished(self, finished_req_ids: set[str]) -> set[str]:
        return self.finished & finished_req_ids


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
