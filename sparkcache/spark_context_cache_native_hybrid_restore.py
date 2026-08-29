"""Native mapped-host placement for an authenticated hybrid page snapshot."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Sequence

from sparkcache import spark_cache_native as native
from sparkcache.spark_context_cache_hybrid import PageSnapshotPlan
from sparkcache.spark_context_cache_native_placement import (
    NativePlacementContractError,
    RestoreState,
)


class NativeHybridRestoreError(RuntimeError):
    """Hybrid native placement cannot safely complete."""


@dataclass(frozen=True)
class NativeHybridRestoreResult:
    placement_stats: Any
    source_bytes: int
    copy_and_submit_ms: float
    finish_ms: float


@dataclass(frozen=True)
class NativePageSlab:
    payload_start: int
    payload_end: int
    spans: tuple[native.PageCopySpan, ...]


def build_page_copy_spans(plan: PageSnapshotPlan) -> tuple[native.PageCopySpan, ...]:
    """Map authenticated snapshot payload ranges onto page destinations."""

    return tuple(
        native.PageCopySpan(
            span.source_start - plan.header_bytes,
            span.source_start - plan.header_bytes,
            0,
            span.source_end - span.source_start,
            destination_index,
            0,
        )
        for destination_index, span in enumerate(plan.spans)
    )


def plan_page_slabs(
    plan: PageSnapshotPlan,
    *,
    arena_bytes: int,
) -> tuple[NativePageSlab, ...]:
    if arena_bytes <= 0:
        raise NativeHybridRestoreError("native page arena bytes must be positive")
    payload_bytes = plan.total_bytes - plan.header_bytes
    slabs = []
    for slab_start in range(0, payload_bytes, arena_bytes):
        slab_end = min(payload_bytes, slab_start + arena_bytes)
        spans = []
        for destination_index, layer in enumerate(plan.spans):
            layer_start = layer.source_start - plan.header_bytes
            layer_end = layer.source_end - plan.header_bytes
            start = max(slab_start, layer_start)
            end = min(slab_end, layer_end)
            if start >= end:
                continue
            spans.append(
                native.PageCopySpan(
                    start - slab_start,
                    start,
                    start - layer_start,
                    end - start,
                    destination_index,
                    0,
                )
            )
        if not spans:
            raise NativeHybridRestoreError("native page slab has no copy spans")
        slabs.append(NativePageSlab(slab_start, slab_end, tuple(spans)))
    return tuple(slabs)


def execute_native_hybrid_placement(
    *,
    adapter: Any,
    request_id: str,
    encoded_pages: bytes,
    plan: PageSnapshotPlan,
    group_slots: Sequence[Sequence[int]],
) -> NativeHybridRestoreResult:
    """Copy one verified snapshot into a mapped arena and scatter it once."""

    if len(encoded_pages) != plan.total_bytes:
        raise NativeHybridRestoreError("hybrid snapshot length differs from page plan")
    payload_bytes = plan.total_bytes - plan.header_bytes
    try:
        transaction = adapter.begin_parked_page_restore(
            request_id,
            group_slots,
            snapshot_bytes=payload_bytes,
        )
        with transaction:
            started = time.perf_counter()
            first_arena = transaction.acquire_arena(0)
            if first_arena.arena_mode != native.ARENA_MAPPED_HOST:
                raise NativeHybridRestoreError(
                    "hybrid native restore requires a mapped-host arena"
                )
            slabs = plan_page_slabs(plan, arena_bytes=first_arena.capacity_bytes)
            for slab_index, slab in enumerate(slabs):
                arena_index = slab_index % native.ARENA_COUNT
                arena = (
                    first_arena
                    if slab_index == 0
                    else transaction.acquire_arena(arena_index)
                )
                used = slab.payload_end - slab.payload_start
                arena_buffer = native.arena_memoryview(arena, length=used)
                source_start = plan.header_bytes + slab.payload_start
                source_end = plan.header_bytes + slab.payload_end
                arena_buffer[:] = encoded_pages[source_start:source_end]
                arena_buffer.release()
                transaction.submit_page_slab(
                    arena_index=arena_index,
                    arena_used_bytes=used,
                    spans=slab.spans,
                )
            copy_and_submit_ms = 1e3 * (time.perf_counter() - started)
            started = time.perf_counter()
            stats = transaction.finish()
            finish_ms = 1e3 * (time.perf_counter() - started)
            if transaction.state is not RestoreState.FINISHED or not transaction.can_resume:
                raise NativeHybridRestoreError(
                    "native page finish did not release the parked request"
                )
    except NativeHybridRestoreError:
        raise
    except (NativePlacementContractError, RuntimeError, TypeError, ValueError) as error:
        raise NativeHybridRestoreError(
            f"native hybrid placement failed: {error}"
        ) from error
    if (
        int(stats.slot_uploads) != 1
        or int(stats.destination_table_uploads) != 1
        or int(stats.slabs_submitted) != len(slabs)
        or int(stats.scatter_kernel_launches) != len(slabs)
        or int(stats.device_error) != 0
        or int(stats.staged_h2d_bytes) != 0
    ):
        raise NativeHybridRestoreError(
            "native hybrid placement statistics violate the mapped transaction"
        )
    return NativeHybridRestoreResult(
        placement_stats=stats,
        source_bytes=len(encoded_pages),
        copy_and_submit_ms=copy_and_submit_ms,
        finish_ms=finish_ms,
    )


__all__ = [
    "NativeHybridRestoreError",
    "NativeHybridRestoreResult",
    "NativePageSlab",
    "build_page_copy_spans",
    "plan_page_slabs",
    "execute_native_hybrid_placement",
]
