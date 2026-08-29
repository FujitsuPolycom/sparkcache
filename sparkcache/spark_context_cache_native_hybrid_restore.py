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


def build_page_copy_spans(plan: PageSnapshotPlan) -> tuple[native.PageCopySpan, ...]:
    """Map authenticated snapshot payload ranges onto page destinations."""

    return tuple(
        native.PageCopySpan(
            span.source_start,
            span.source_start - plan.header_bytes,
            0,
            span.source_end - span.source_start,
            destination_index,
            0,
        )
        for destination_index, span in enumerate(plan.spans)
    )


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
    spans = build_page_copy_spans(plan)
    try:
        transaction = adapter.begin_parked_page_restore(
            request_id,
            group_slots,
            snapshot_bytes=payload_bytes,
        )
        with transaction:
            arena = transaction.acquire_arena(0)
            if arena.arena_mode != native.ARENA_MAPPED_HOST:
                raise NativeHybridRestoreError(
                    "hybrid native restore requires a mapped-host arena"
                )
            if len(encoded_pages) > arena.capacity_bytes:
                raise NativeHybridRestoreError(
                    "hybrid snapshot exceeds one native arena"
                )
            started = time.perf_counter()
            arena_buffer = native.arena_memoryview(arena, length=len(encoded_pages))
            arena_buffer[:] = encoded_pages
            arena_buffer.release()
            transaction.submit_page_slab(
                arena_index=0,
                arena_used_bytes=len(encoded_pages),
                spans=spans,
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
        or int(stats.slabs_submitted) != 1
        or int(stats.scatter_kernel_launches) != 1
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
    "build_page_copy_spans",
    "execute_native_hybrid_placement",
]
