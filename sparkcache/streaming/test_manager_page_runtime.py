from __future__ import annotations

import threading
import time
import types
from pathlib import Path

import pytest

from sparkcache.streaming import manager_page_runtime
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
)
from sparkcache.streaming.manager_page_runtime import ManagerPageCaptureRuntime
from sparkcache.streaming.manager_page_factory import ManagerPageCaptureSettings


class FakeView:
    def __init__(self, payload: bytes) -> None:
        self.payload = memoryview(payload)


class FakeRing:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload
        self.ready = threading.Event()
        self.submissions: list[tuple] = []
        self.drained: list[int] = []
        self.released: list[object] = []
        self.closed = False
        self.poll_error: Exception | None = None
        self.active: set[int] = set()

    @property
    def active_ticket_count(self) -> int:
        return len(self.active)

    def submit(self, **kwargs):
        self.submissions.append(tuple(sorted(kwargs.items())))
        self.active.add(kwargs["context_sequence"])
        return types.SimpleNamespace(context_sequence=kwargs["context_sequence"])

    def poll(self, ticket):
        if self.poll_error is not None:
            raise self.poll_error
        return FakeView(self.payload) if self.ready.is_set() else None

    def claim(self, ticket):
        return FakeView(self.payload)

    def release(self, ticket) -> None:
        self.released.append(ticket)
        self.active.discard(ticket.context_sequence)

    def drain_context(self, context_sequence: int) -> None:
        self.drained.append(context_sequence)
        self.active.discard(context_sequence)

    def shutdown(self) -> None:
        self.closed = True


class FakeConnector:
    def __init__(self) -> None:
        self._page_layout = PageLayout(
            (
                PageGroup(
                    256,
                    (PageLayer("attention", "u8", (2,), 2),),
                ),
                PageGroup(
                    2304,
                    (PageLayer("recurrent", "u8", (3,), 3),),
                    reuse_policy="recurrent_align",
                ),
            )
        )
        self._group_topology = (
            {
                "logical_tokens_per_block": 256,
                "reuse_policy": "full",
            },
            {
                "logical_tokens_per_block": 2304,
                "reuse_policy": "recurrent_align",
            },
        )
        self.completed: list[tuple[object, bytes, tuple[int, ...]]] = []
        self.completed_extensions: list[
            tuple[object, bytes, tuple[int, ...], tuple[int, ...]]
        ] = []
        self.aborted: list[tuple[str, str]] = []
        self.completed_event = threading.Event()

    @staticmethod
    def _worker_rank() -> int:
        return 3

    @staticmethod
    def _select_group_blocks_for_span(
        groups,
        span_tokens,
        *,
        recurrent_boundary_blocks,
    ):
        del span_tokens, recurrent_boundary_blocks
        return groups

    @staticmethod
    def _group_block_counts_for_span(span_tokens: int) -> tuple[int, int]:
        return ((span_tokens + 255) // 256, 1)

    def _complete_async_page_capture(
        self,
        plan: object,
        encoded_pages: bytes,
        block_counts: tuple[int, ...],
    ) -> None:
        reused = getattr(encoded_pages, "reused_pages_by_group", None)
        if reused is not None:
            materialized = encoded_pages.read_range(
                0, encoded_pages.total_bytes
            )
            encoded_pages.release()
            self.completed_extensions.append(
                (plan, materialized, block_counts, tuple(reused))
            )
            self.completed_event.set()
            return
        if hasattr(encoded_pages, "read_range"):
            materialized = encoded_pages.read_range(
                0, encoded_pages.total_bytes
            )
            encoded_pages.release()
        else:
            materialized = encoded_pages
        self.completed.append((plan, materialized, block_counts))
        self.completed_event.set()

    def _abort_async_page_capture(self, digest: str, reason: str) -> None:
        self.aborted.append((digest, reason))


def _plan(request_id: str = "request"):
    return types.SimpleNamespace(
        request_id=request_id,
        digest="a" * 64,
        span_tokens=512,
        group_block_ids=((2, 5), (7,)),
        recurrent_boundary_blocks=((1, 7),),
        base_context_digest="",
        base_span_tokens=0,
    )


def test_runtime_completes_native_capture_then_hands_off_encoded_snapshot(
    caplog: pytest.LogCaptureFixture,
) -> None:
    connector = FakeConnector()
    raw_body = b"a2a5" + b"r07"
    ring = FakeRing(raw_body)
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    plan = _plan()
    with caplog.at_level("INFO", logger="vllm.spark_context_cache"):
        assert runtime.submit(plan, producer_stream=91)
        assert connector.completed == []
        ring.ready.set()
        assert connector.completed_event.wait(timeout=1)

    assert len(connector.completed) == 1
    completed_plan, encoded, block_counts = connector.completed[0]
    assert completed_plan is plan
    assert block_counts == (2, 1)
    assert decode_page_snapshot(
        connector._page_layout, encoded, block_counts
    ) == {"attention": b"a2a5", "recurrent": b"r07"}
    assert runtime.take_finished(set()) == set()
    assert runtime.take_finished({"request"}) == {"request"}
    capture_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("sparkcache: capture ")
    ]
    assert len(capture_lines) == 1
    assert "rank=3" in capture_lines[0]
    assert "digest=aaaaaaaaaaaa" in capture_lines[0]
    assert "tokens=512" in capture_lines[0]
    assert "observed=" in capture_lines[0]
    assert "rate=" in capture_lines[0]
    assert " tok/s" in capture_lines[0]
    assert "bytes=0.0MiB" in capture_lines[0]
    runtime.shutdown()
    assert ring.closed


def test_runtime_submits_only_unreused_pages_for_an_extension() -> None:
    connector = FakeConnector()
    ring = FakeRing(b"")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    plan = _plan("extension")
    plan.base_context_digest = "b" * 64
    plan.base_span_tokens = 256

    assert runtime.submit(plan, producer_stream=91)

    submission = dict(ring.submissions[0])
    assert submission["logical_start"] == 256
    assert submission["physical_pages_by_group"] == ((5,), (7,))
    runtime.preempt("extension")
    runtime.shutdown()


def test_runtime_hands_off_sparse_extension_geometry() -> None:
    connector = FakeConnector()
    ring = FakeRing(b"a5r07")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    plan = _plan("extension-complete")
    plan.base_context_digest = "b" * 64
    plan.base_span_tokens = 256

    assert runtime.submit(plan, producer_stream=91)
    ring.ready.set()
    assert connector.completed_event.wait(timeout=1)

    assert connector.completed == []
    assert connector.completed_extensions == [
        (plan, b"a5r07", (2, 1), (1, 0))
    ]
    assert runtime.take_finished({"extension-complete"}) == {
        "extension-complete"
    }
    runtime.shutdown()


def test_capture_telemetry_failure_does_not_abort_publication(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = FakeConnector()
    ring = FakeRing(b"a2a5r07")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )

    def fail_log(*args, **kwargs) -> None:
        del args, kwargs
        raise RuntimeError("broken diagnostic handler")

    monkeypatch.setattr(manager_page_runtime.logger, "info", fail_log)
    assert runtime.submit(_plan("diagnostic-failure"), producer_stream=91)
    ring.ready.set()
    assert connector.completed_event.wait(timeout=1)

    assert len(connector.completed) == 1
    assert connector.aborted == []
    assert runtime.take_finished({"diagnostic-failure"}) == {
        "diagnostic-failure"
    }
    runtime.shutdown()


def test_runtime_drains_preempted_capture_before_returning() -> None:
    connector = FakeConnector()
    ring = FakeRing(b"a2a5r07")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    assert runtime.submit(_plan("preempted"), producer_stream=91)
    runtime.preempt("preempted")
    assert ring.drained == [1]
    assert connector.completed == []
    assert runtime.take_finished({"preempted"}) == {"preempted"}
    runtime.shutdown()


def test_capture_settings_require_an_attested_bounded_artifact() -> None:
    settings = ManagerPageCaptureSettings(
        library_path=Path("/opt/sparkcache/libspark_cache_snapshot.so"),
        library_sha256="a" * 64,
        slot_bytes=8 * 1024**3,
        slot_count=2,
    )
    assert settings.slot_bytes == 8 * 1024**3

    with pytest.raises(RuntimeError, match="path must be absolute"):
        ManagerPageCaptureSettings(Path("relative.so"), "a" * 64, 1024)
    with pytest.raises(RuntimeError, match="lowercase SHA-256"):
        ManagerPageCaptureSettings(Path("/absolute.so"), "bad", 1024)
    with pytest.raises(RuntimeError, match="two or three"):
        ManagerPageCaptureSettings(Path("/absolute.so"), "a" * 64, 1024, 4)


def test_unknown_background_ownership_never_reports_finished() -> None:
    connector = FakeConnector()
    ring = FakeRing(b"a2a5r07")
    ring.poll_error = RuntimeError("CUDA event state unknown")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    assert runtime.submit(_plan("uncertain"), producer_stream=91)
    assert not runtime.wait_idle(timeout=0.05)
    with pytest.raises(RuntimeError, match="ownership is uncertain"):
        runtime.take_finished({"uncertain"})
    runtime.shutdown()


def test_unknown_preemption_never_reports_an_unowned_request() -> None:
    connector = FakeConnector()
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=FakeRing(b"a2a5r07"),
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )

    runtime.preempt("never-submitted")

    assert runtime.take_finished({"never-submitted"}) == set()
    runtime.shutdown()


def test_finished_reporting_does_not_wait_for_writer_handoff() -> None:
    class BlockingConnector(FakeConnector):
        def __init__(self) -> None:
            super().__init__()
            self.handoff_started = threading.Event()
            self.allow_handoff = threading.Event()

        def _complete_async_page_capture(
            self,
            plan: object,
            encoded_pages: object,
            block_counts: tuple[int, ...],
        ) -> None:
            self.handoff_started.set()
            self.allow_handoff.wait(timeout=1)
            super()._complete_async_page_capture(
                plan, encoded_pages, block_counts
            )

    connector = BlockingConnector()
    ring = FakeRing(b"a2a5r07")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    assert runtime.submit(_plan("blocked-writer"), producer_stream=91)
    ring.ready.set()
    assert connector.handoff_started.wait(timeout=1)

    started = time.perf_counter()
    assert runtime.take_finished({"blocked-writer"}) == {"blocked-writer"}
    assert time.perf_counter() - started < 0.05

    connector.allow_handoff.set()
    assert connector.completed_event.wait(timeout=1)
    runtime.shutdown()


def test_shutdown_retains_ring_while_durable_writer_owns_scatter() -> None:
    class HoldingConnector(FakeConnector):
        def __init__(self) -> None:
            super().__init__()
            self.scatter = None

        def _complete_async_page_capture(
            self,
            plan: object,
            encoded_pages: object,
            block_counts: tuple[int, ...],
        ) -> None:
            del plan, block_counts
            self.scatter = encoded_pages
            self.completed_event.set()

    connector = HoldingConnector()
    ring = FakeRing(b"a2a5r07")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
    )
    assert runtime.submit(_plan("slow-store"), producer_stream=91)
    ring.ready.set()
    assert connector.completed_event.wait(timeout=1)

    assert runtime.shutdown() is False
    assert not ring.closed
    assert connector.scatter is not None

    connector.scatter.release()
    assert runtime.shutdown() is True
    assert ring.closed
