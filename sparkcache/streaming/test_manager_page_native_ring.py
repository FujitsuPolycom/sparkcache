from __future__ import annotations

import ctypes
from collections import deque

from sparkcache.streaming.manager_page_capture import ManagerPageSource
from sparkcache.streaming.manager_page_native_ring import NativeManagerPageRing
from sparkcache.streaming.manager_page_native_ring_ctypes import (
    PageCaptureAbiInfo,
    PageCaptureGroup,
    PageCapturePlan,
    PageCaptureSource,
    PageCaptureSpan,
    PageCaptureSubmission,
)
from sparkcache.streaming.native_ring import (
    NativeRingConfig,
    NativeStatus,
    RawReadyView,
    RawTicket,
)


class FakePageBackend:
    def __init__(self) -> None:
        self.config: NativeRingConfig | None = None
        self.sources: tuple[ManagerPageSource, ...] = ()
        self.group_count = 0
        self.entries: dict[tuple[int, int], dict] = {}
        self.submit_statuses: deque[int] = deque()
        self.calls: list[tuple] = []
        self.generation = 1

    def create(self, config: NativeRingConfig) -> int:
        self.config = config
        self.calls.append(("create", config))
        return NativeStatus.OK

    def configure_page_sources(
        self,
        sources: tuple[ManagerPageSource, ...],
        group_count: int,
    ) -> int:
        self.sources = sources
        self.group_count = group_count
        self.calls.append(("configure", sources, group_count))
        return NativeStatus.OK

    def submit_pages(
        self,
        *,
        context_sequence: int,
        logical_start: int,
        groups: tuple[tuple[int, ...], ...],
        producer_stream: int,
        used_bytes: int,
    ) -> tuple[int, RawTicket | None]:
        self.calls.append(
            (
                "submit",
                context_sequence,
                logical_start,
                groups,
                producer_stream,
                used_bytes,
            )
        )
        status = self.submit_statuses.popleft() if self.submit_statuses else 0
        if status != NativeStatus.OK:
            return status, None
        ticket = RawTicket(self.generation, 0)
        self.generation += 1
        self.entries[(ticket.slot_index, ticket.generation)] = {
            "ticket": ticket,
            "context_sequence": context_sequence,
            "logical_start": logical_start,
            "groups": groups,
            "used_bytes": used_bytes,
            "ready": False,
            "buffer": bytearray(range(used_bytes)),
        }
        return NativeStatus.OK, ticket

    def poll(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        entry = self.entries[(ticket.slot_index, ticket.generation)]
        if not entry["ready"]:
            return NativeStatus.NOT_READY, None
        return NativeStatus.OK, self._view(entry, state=2)

    def claim(self, ticket: RawTicket) -> tuple[int, RawReadyView | None]:
        entry = self.entries[(ticket.slot_index, ticket.generation)]
        if not entry["ready"]:
            return NativeStatus.NOT_READY, None
        return NativeStatus.OK, self._view(entry, state=3)

    def release(self, ticket: RawTicket) -> int:
        self.entries.pop((ticket.slot_index, ticket.generation))
        return NativeStatus.OK

    def drain_context(self, context_sequence: int) -> int:
        self.calls.append(("drain", context_sequence))
        for key, entry in tuple(self.entries.items()):
            if entry["context_sequence"] == context_sequence:
                self.entries.pop(key)
        return NativeStatus.OK

    def abandon_context(self, context_sequence: int) -> int:
        return self.drain_context(context_sequence)

    def shutdown(self) -> int:
        self.entries.clear()
        return NativeStatus.OK

    def destroy(self) -> None:
        self.calls.append(("destroy",))

    def status_text(self, status: int) -> str:
        return NativeStatus(status).name.lower()

    def ready(self, ticket) -> None:
        self.entries[(ticket.slot_index, ticket.generation)]["ready"] = True

    def _view(self, entry: dict, *, state: int) -> RawReadyView:
        assert self.config is not None
        ticket = entry["ticket"]
        return RawReadyView(
            payload=memoryview(entry["buffer"]),
            capacity_bytes=self.config.slot_bytes,
            used_bytes=entry["used_bytes"],
            context_sequence=entry["context_sequence"],
            logical_start=entry["logical_start"],
            generation=ticket.generation,
            row_count=sum(len(group) for group in entry["groups"]),
            slot_index=ticket.slot_index,
            record_mask=0,
            state=state,
            record_offsets=(0, 0, 0, 0),
            record_lengths=(0, 0, 0, 0),
        )


def _config() -> NativeRingConfig:
    return NativeRingConfig(1, 64, 2, 3, 8, 0)


def _sources() -> tuple[ManagerPageSource, ...]:
    return (
        ManagerPageSource(0x1000, 8, 4, 4, 0, 0),
        ManagerPageSource(0x2000, 8, 6, 6, 0, 1),
        ManagerPageSource(0x3000, 8, 3, 3, 1, 0),
    )


def test_page_ring_submits_grouped_pages_and_exposes_completed_bytes() -> None:
    backend = FakePageBackend()
    ring = NativeManagerPageRing(_config(), backend=backend)
    ring.configure_sources(_sources(), group_count=2)
    ticket = ring.submit(
        context_sequence=7,
        logical_start=0,
        physical_pages_by_group=((2, 5), (7,)),
        producer_stream=91,
    )
    assert ticket is not None
    assert ticket.plan.used_bytes == 23
    assert ring.poll(ticket) is None

    backend.ready(ticket)
    assert ring.poll(ticket) is not None
    view = ring.claim(ticket)
    assert bytes(view.payload) == bytes(range(23))
    ring.release(ticket)
    ring.shutdown()


def test_page_ring_returns_immediately_on_saturation_and_drains_preemption() -> None:
    backend = FakePageBackend()
    ring = NativeManagerPageRing(_config(), backend=backend)
    ring.configure_sources(_sources(), group_count=2)
    backend.submit_statuses.append(NativeStatus.WOULD_BLOCK)
    assert (
        ring.submit(
            context_sequence=8,
            logical_start=0,
            physical_pages_by_group=((2, 5), (7,)),
            producer_stream=91,
        )
        is None
    )
    ticket = ring.submit(
        context_sequence=9,
        logical_start=0,
        physical_pages_by_group=((2, 5), (7,)),
        producer_stream=91,
    )
    assert ticket is not None
    ring.drain_context(9)
    assert ("drain", 9) in backend.calls
    ring.shutdown()


def test_manager_page_ctypes_sizes_match_the_fixed_contract() -> None:
    assert ctypes.sizeof(PageCaptureSource) == 40
    assert ctypes.sizeof(PageCaptureGroup) == 16
    assert ctypes.sizeof(PageCaptureSpan) == 32
    assert ctypes.sizeof(PageCapturePlan) == 24
    assert ctypes.sizeof(PageCaptureSubmission) == 32
    assert ctypes.sizeof(PageCaptureAbiInfo) == 48


def test_drain_ready_unclaimed_capture_returns_the_slot_to_service() -> None:
    backend = FakePageBackend()
    ring = NativeManagerPageRing(_config(), backend=backend)
    ring.configure_sources(_sources(), group_count=2)
    ticket = ring.submit(
        context_sequence=10,
        logical_start=0,
        physical_pages_by_group=((2, 5), (7,)),
        producer_stream=91,
    )
    assert ticket is not None
    backend.ready(ticket)
    assert ring.poll(ticket) is not None

    ring.drain_context(10)

    replacement = ring.submit(
        context_sequence=11,
        logical_start=0,
        physical_pages_by_group=((2, 5), (7,)),
        producer_stream=91,
    )
    assert replacement is not None
    assert replacement.generation != ticket.generation
    ring.drain_context(11)
    ring.shutdown()
