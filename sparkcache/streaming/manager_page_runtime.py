"""Background progress for one bounded CUDA manager-page capture ring."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable

from sparkcache.spark_context_cache_hybrid import (
    PageLayout,
    encode_page_snapshot_header,
)
from sparkcache.streaming.manager_page_capture import (
    select_manager_page_extension_pages,
)

logger = logging.getLogger("vllm.spark_context_cache")


def _format_token_rate(token_rate: float) -> str:
    if token_rate >= 1_000_000:
        return f"{token_rate / 1_000_000:.2f}M"
    if token_rate >= 1_000:
        return f"{token_rate / 1_000:.0f}K"
    return f"{token_rate:.0f}"


def _log_capture_observed(
    *,
    rank: int,
    digest: str,
    span_tokens: int,
    elapsed_ns: int,
    payload_bytes: int,
) -> None:
    """Report optional capture telemetry without affecting publication."""

    token_rate = (
        span_tokens * 1_000_000_000 / elapsed_ns if elapsed_ns else 0.0
    )
    try:
        logger.info(
            "sparkcache: capture rank=%d digest=%s tokens=%d observed=%.1fms"
            " rate=%s tok/s bytes=%.1fMiB",
            rank,
            digest[:12],
            span_tokens,
            elapsed_ns / 1_000_000,
            _format_token_rate(token_rate),
            payload_bytes / 1024**2,
        )
    except Exception:  # noqa: BLE001 - diagnostics never alter cache behavior
        return


@dataclass(slots=True)
class _PendingCapture:
    request_id: str
    context_sequence: int
    plan: Any
    ticket: Any
    capture_block_counts: tuple[int, ...]
    result_block_counts: tuple[int, ...]
    reused_pages_by_group: tuple[int, ...]
    submitted_ns: int


class PageSnapshotScatter:
    """Small header plus a claimed ring view with bounded read access."""

    def __init__(
        self,
        header: bytes,
        body: memoryview,
        release: Callable[[], None],
    ) -> None:
        self.header_bytes = bytes(header)
        self._body = body.toreadonly()
        self._release_callback = release
        self._released = False
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return len(self.header_bytes) + self._body.nbytes

    def read_range(self, start: int, end: int) -> bytes:
        if self._released:
            raise RuntimeError("page snapshot scatter view was released")
        if not 0 <= start <= end <= self.total_bytes:
            raise ValueError("page snapshot scatter range is invalid")
        header_length = len(self.header_bytes)
        parts = []
        if start < header_length:
            parts.append(self.header_bytes[start : min(end, header_length)])
        if end > header_length:
            body_start = max(0, start - header_length)
            body_end = end - header_length
            parts.append(self._body[body_start:body_end].tobytes())
        return b"".join(parts)

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._release_callback()
            self._body.release()
            self._released = True


class PageExtensionScatter:
    """Claimed changed-page bytes plus their authenticated base geometry."""

    def __init__(
        self,
        body: memoryview,
        *,
        reused_pages_by_group: tuple[int, ...],
        result_block_counts: tuple[int, ...],
        release: Callable[[], None],
    ) -> None:
        self._body = body.toreadonly()
        self.reused_pages_by_group = reused_pages_by_group
        self.result_block_counts = result_block_counts
        self._release_callback = release
        self._released = False
        self._lock = threading.Lock()

    @property
    def total_bytes(self) -> int:
        return self._body.nbytes

    def read_range(self, start: int, end: int) -> bytes:
        if self._released:
            raise RuntimeError("page extension scatter view was released")
        if not 0 <= start <= end <= self.total_bytes:
            raise ValueError("page extension scatter range is invalid")
        return self._body[start:end].tobytes()

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._release_callback()
            self._body.release()
            self._released = True


class ManagerPageCaptureRuntime:
    """Poll native capture off the inference thread and hand off CPU bytes."""

    def __init__(
        self,
        connector: Any,
        *,
        ring: Any,
        progress_poll_seconds: float = 0.005,
        progress_thread_initializer: Callable[[], None],
    ) -> None:
        if progress_poll_seconds <= 0:
            raise ValueError("progress_poll_seconds must be positive")
        layout = getattr(connector, "_page_layout", None)
        if not isinstance(layout, PageLayout):
            raise RuntimeError(
                "asynchronous manager-page capture requires a registered page layout"
            )
        self._connector = connector
        self._rank = int(connector._worker_rank())
        self._layout = layout
        self._ring = ring
        self._poll_seconds = float(progress_poll_seconds)
        self._thread_initializer = progress_thread_initializer
        self._cv = threading.Condition(threading.RLock())
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_sequence = 1
        self._pending: dict[str, _PendingCapture] = {}
        self._completed: set[str] = set()
        self._closed = False
        self._fatal: BaseException | None = None

    def submit(self, plan: Any, *, producer_stream: int) -> bool:
        request_id = str(plan.request_id)
        groups = self._connector._select_group_blocks_for_span(
            plan.group_block_ids,
            plan.span_tokens,
            recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
        )
        capture_groups = groups
        result_block_counts = tuple(len(group) for group in groups)
        reused_pages_by_group = tuple(0 for _ in groups)
        logical_start = 0
        if plan.base_context_digest:
            capture_groups, _base_counts, result_block_counts, reused_pages_by_group = (
                select_manager_page_extension_pages(
                    groups,
                    base_page_counts=self._connector._group_block_counts_for_span(
                        plan.base_span_tokens
                    ),
                    logical_tokens_per_page=tuple(
                        int(group["logical_tokens_per_block"])
                        for group in self._connector._group_topology
                    ),
                    reuse_policies=tuple(
                        str(group["reuse_policy"])
                        for group in self._connector._group_topology
                    ),
                    base_boundary_tokens=plan.base_span_tokens,
                )
            )
            logical_start = plan.base_span_tokens
        with self._cv:
            if self._closed:
                raise RuntimeError("manager-page capture runtime is closed")
            if self._fatal is not None:
                raise RuntimeError(
                    "manager-page capture ownership is uncertain"
                ) from self._fatal
            if request_id in self._pending:
                raise RuntimeError("request already has a manager-page capture")
            self._completed.discard(request_id)
            context_sequence = self._next_sequence
            self._next_sequence += 1
            submitted_ns = time.perf_counter_ns()
            try:
                ticket = self._ring.submit(
                    context_sequence=context_sequence,
                    logical_start=logical_start,
                    physical_pages_by_group=capture_groups,
                    producer_stream=producer_stream,
                )
            except Exception:
                self._completed.add(request_id)
                self._cv.notify_all()
                raise
            if ticket is None:
                self._completed.add(request_id)
                self._connector._abort_async_page_capture(
                    plan.digest, "capture ring is busy or the payload was rejected"
                )
                self._cv.notify_all()
                return False
            self._pending[request_id] = _PendingCapture(
                request_id=request_id,
                context_sequence=context_sequence,
                plan=plan,
                ticket=ticket,
                capture_block_counts=tuple(len(group) for group in capture_groups),
                result_block_counts=result_block_counts,
                reused_pages_by_group=reused_pages_by_group,
                submitted_ns=submitted_ns,
            )
            self._ensure_thread_locked()
            self._wake.set()
            return True

    def preempt(self, request_id: str) -> None:
        with self._cv:
            pending = self._pending.pop(request_id, None)
            if pending is None:
                return
            # vLLM may reuse every group page as soon as this callback returns.
            # The native drain synchronizes only this context's capture event.
            self._ring.drain_context(pending.context_sequence)
            self._connector._abort_async_page_capture(
                pending.plan.digest, "request was preempted"
            )
            self._completed.add(request_id)
            self._cv.notify_all()

    def finish_without_capture(self, request_id: str) -> None:
        """Report a terminal store outcome when no CUDA work was submitted.

        Scheduler-side asynchronous publication eligibility delays KV-block
        reclamation before workers decide whether they can accept the store.
        A worker that skips before submission therefore owes the same terminal
        completion as a drained capture. An in-flight request cannot use this
        path because its pages remain owned by the native completion fence.
        """

        with self._cv:
            if self._closed:
                raise RuntimeError("manager-page capture runtime is closed")
            if request_id in self._pending:
                raise RuntimeError(
                    "cannot finish a manager-page capture before its fence drains"
                )
            self._completed.add(request_id)
            self._cv.notify_all()

    def take_finished(self, finished_request_ids: set[str]) -> set[str]:
        with self._cv:
            if self._fatal is not None:
                raise RuntimeError(
                    "manager-page capture ownership is uncertain"
                ) from self._fatal
            ready = self._completed & set(finished_request_ids)
            self._completed.difference_update(ready)
            return ready

    def wait_idle(self, timeout: float | None = None) -> bool:
        with self._cv:
            return self._cv.wait_for(lambda: not self._pending, timeout)

    def shutdown(self) -> bool:
        self.quiesce()
        if int(getattr(self._ring, "active_ticket_count", 0)):
            return False
        self._ring.shutdown()
        return True

    def quiesce(self) -> None:
        with self._cv:
            if self._closed:
                return
            self._closed = True
            pending = tuple(self._pending.values())
            self._pending.clear()
            self._stop.set()
            self._wake.set()
            thread = self._thread
        for capture in pending:
            self._ring.drain_context(capture.context_sequence)
            self._connector._abort_async_page_capture(
                capture.plan.digest, "capture runtime shut down"
            )
        if thread is not None and thread is not threading.current_thread():
            thread.join()

    def _ensure_thread_locked(self) -> None:
        if self._thread is not None:
            return
        thread = threading.Thread(
            target=self._progress_main,
            name="sparkcache-manager-page-capture",
            daemon=True,
        )
        self._thread = thread
        thread.start()

    def _progress_main(self) -> None:
        try:
            self._thread_initializer()
            while not self._stop.is_set():
                self._wake.wait()
                if self._stop.is_set():
                    return
                self._progress_once()
                if self._stop.wait(self._poll_seconds):
                    return
        except Exception as error:  # noqa: BLE001 - background failure is isolated
            with self._cv:
                self._fatal = error
                for capture in self._pending.values():
                    self._connector._abort_async_page_capture(
                        capture.plan.digest,
                        f"background capture failed: {error}",
                    )
                # Do not report finished_sending. vLLM must retain every page
                # until a later drain or worker termination proves ownership.
                self._cv.notify_all()

    def _progress_once(self) -> None:
        ready: list[tuple[_PendingCapture, Any, int]] = []
        with self._cv:
            for capture in tuple(self._pending.values()):
                view = self._ring.poll(capture.ticket)
                if view is None:
                    continue
                claimed = self._ring.claim(capture.ticket)
                if self._pending.pop(capture.request_id, None) is not None:
                    self._completed.add(capture.request_id)
                    ready.append((capture, claimed, time.perf_counter_ns()))
                if not self._pending:
                    self._wake.clear()
                self._cv.notify_all()
        for capture, claimed, completed_ns in ready:
            scatter = None
            try:
                elapsed_ns = max(0, completed_ns - capture.submitted_ns)
                _log_capture_observed(
                    rank=self._rank,
                    digest=capture.plan.digest,
                    span_tokens=capture.plan.span_tokens,
                    elapsed_ns=elapsed_ns,
                    payload_bytes=claimed.payload.nbytes,
                )
                def release(ticket=capture.ticket) -> None:
                    self._ring.release(ticket)

                if capture.plan.base_context_digest:
                    scatter = PageExtensionScatter(
                        claimed.payload,
                        reused_pages_by_group=capture.reused_pages_by_group,
                        result_block_counts=capture.result_block_counts,
                        release=release,
                    )
                else:
                    header = encode_page_snapshot_header(
                        self._layout, capture.capture_block_counts
                    )
                    scatter = PageSnapshotScatter(
                        header,
                        claimed.payload,
                        release,
                    )
                self._connector._complete_async_page_capture(
                    capture.plan,
                    scatter,
                    capture.result_block_counts,
                )
            except Exception as error:  # noqa: BLE001 - serving continues
                if scatter is not None:
                    scatter.release()
                else:
                    self._ring.drain_context(capture.context_sequence)
                self._connector._abort_async_page_capture(
                    capture.plan.digest,
                    f"capture completion failed: {error}",
                )


__all__ = [
    "ManagerPageCaptureRuntime",
    "PageExtensionScatter",
    "PageSnapshotScatter",
]
