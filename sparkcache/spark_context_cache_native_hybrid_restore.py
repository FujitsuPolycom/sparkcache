"""Native mapped-host placement for an authenticated hybrid page snapshot."""

from __future__ import annotations

import concurrent.futures
import json
import os
import struct
import time
from dataclasses import dataclass
from typing import Any, Sequence

from sparkcache import spark_cache_native as native
from sparkcache.spark_context_cache_hybrid import PageSnapshotPlan
from sparkcache.spark_context_cache_hybrid import PageLayout, plan_page_snapshot
from sparkcache.spark_context_cache_native_placement import (
    NativePlacementContractError,
    RestoreState,
)
from sparkcache.spark_context_cache_native_restore import (
    _read_and_authenticate,
    plan_native_restore,
)

_CHUNK_PREFIX = struct.Struct("<8sII")
_CHUNK_MAGIC = b"SPCKV001"
_TARGET_KIND = native.RECORD_TARGET_CKV


class NativeHybridRestoreError(RuntimeError):
    """Hybrid native placement cannot safely complete."""


@dataclass(frozen=True)
class NativeHybridRestoreResult:
    placement_stats: Any
    source_bytes: int
    copy_and_submit_ms: float
    finish_ms: float
    slabs: int = 1
    read_and_hash_ms: float = 0.0


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
        slabs=len(slabs),
        read_and_hash_ms=0.0,
    )


def _target_record_plan(path: Any, encoded_bytes: int) -> tuple[int, int, int]:
    fd = os.open(path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        prefix = os.pread(fd, _CHUNK_PREFIX.size, 0)
        if len(prefix) != _CHUNK_PREFIX.size:
            raise NativeHybridRestoreError("hybrid chunk prefix is truncated")
        magic, abi, header_bytes = _CHUNK_PREFIX.unpack(prefix)
        if magic != _CHUNK_MAGIC or abi != 1 or header_bytes > 1 << 20:
            raise NativeHybridRestoreError("hybrid chunk prefix is unsupported")
        raw = os.pread(fd, header_bytes, _CHUNK_PREFIX.size)
        if len(raw) != header_bytes:
            raise NativeHybridRestoreError("hybrid chunk header is truncated")
        header = json.loads(raw)
        payload_offset = _CHUNK_PREFIX.size + header_bytes
        for record in header.get("records", ()):
            if record.get("kind") == "target_ckv":
                offset = int(record["offset"])
                length = int(record["length"])
                if offset < 0 or length <= 0 or payload_offset + offset + length > encoded_bytes:
                    break
                return payload_offset, offset, length
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise NativeHybridRestoreError(f"cannot plan hybrid chunk: {error}") from error
    finally:
        os.close(fd)
    raise NativeHybridRestoreError("hybrid chunk has no valid target_ckv record")


def execute_native_hybrid_restore(
    *,
    adapter: Any,
    request_id: str,
    lookup: Any,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    io_workers: int = 8,
) -> NativeHybridRestoreResult:
    """Pipeline authenticated .spcc reads directly into mapped page scatter."""

    slabs = plan_native_restore(
        lookup,
        cache_root=cache_root,
        expected_span_tokens=expected_span_tokens,
        dcp_degree=1,
        arena_bytes=arena_bytes,
    )
    target_plans = {}
    snapshot_bytes = 0
    for slab in slabs:
        for chunk in slab.chunks:
            target = _target_record_plan(chunk.path, chunk.encoded_bytes)
            target_plans[chunk.path] = target
            snapshot_bytes += target[2]
    first = slabs[0].chunks[0]
    payload_offset, target_offset, target_length = target_plans[first.path]
    fd = os.open(first.path, os.O_RDONLY | getattr(os, "O_CLOEXEC", 0))
    try:
        header_prefix = os.pread(
            fd,
            min(target_length, 1 << 20),
            payload_offset + target_offset,
        )
    finally:
        os.close(fd)
    page_plan = plan_page_snapshot(
        layout,
        header_prefix,
        tuple(len(group) for group in group_slots),
        total_bytes=snapshot_bytes,
    )
    transaction = adapter.begin_parked_page_restore(
        request_id,
        group_slots,
        snapshot_bytes=snapshot_bytes - page_plan.header_bytes,
    )
    read_ms = submit_ms = 0.0
    stream_offset = 0
    with transaction:
        for slab_index, slab in enumerate(slabs):
            arena_index = slab_index % native.ARENA_COUNT
            arena = transaction.acquire_arena(arena_index)
            buffer = native.arena_memoryview(arena, length=slab.arena_used_bytes)
            views = tuple(
                buffer[c.arena_offset_bytes : c.arena_offset_bytes + c.encoded_bytes]
                for c in slab.chunks
            )
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(io_workers, len(slab.chunks))
            ) as pool:
                tuple(pool.map(lambda item: _read_and_authenticate(*item), zip(slab.chunks, views, strict=True)))
            read_ms += 1e3 * (time.perf_counter() - started)
            del views
            del buffer
            started = time.perf_counter()
            spans = []
            for chunk in slab.chunks:
                parsed = transaction.parse_verified_chunk(
                    arena=arena,
                    arena_used_bytes=slab.arena_used_bytes,
                    arena_offset_bytes=chunk.arena_offset_bytes,
                    encoded_bytes=chunk.encoded_bytes,
                    expected_logical_start=chunk.logical_start,
                    dcp_degree=1,
                    dcp_rank=0,
                    first_slot_index=chunk.first_slot_index,
                    required_data_record_mask=1 << _TARGET_KIND,
                )
                payload, target_offset, target_length = target_plans[chunk.path]
                if (
                    int(parsed.payload_offset_bytes) != payload
                    or int(parsed.record_offset_bytes[_TARGET_KIND]) != target_offset
                    or int(parsed.record_length_bytes[_TARGET_KIND]) != target_length
                ):
                    raise NativeHybridRestoreError("authenticated target extent changed")
                extent_start = stream_offset
                extent_end = stream_offset + target_length
                for destination_index, layer in enumerate(page_plan.spans):
                    start = max(extent_start, layer.source_start)
                    end = min(extent_end, layer.source_end)
                    if start >= end:
                        continue
                    spans.append(
                        native.PageCopySpan(
                            chunk.arena_offset_bytes + payload + target_offset + start - extent_start,
                            start - page_plan.header_bytes,
                            start - layer.source_start,
                            end - start,
                            destination_index,
                            0,
                        )
                    )
                stream_offset = extent_end
            transaction.submit_page_slab(
                arena_index=arena_index,
                arena_used_bytes=slab.arena_used_bytes,
                spans=spans,
            )
            submit_ms += 1e3 * (time.perf_counter() - started)
        started = time.perf_counter()
        stats = transaction.finish()
        finish_ms = 1e3 * (time.perf_counter() - started)
    return NativeHybridRestoreResult(
        placement_stats=stats,
        source_bytes=snapshot_bytes,
        copy_and_submit_ms=submit_ms,
        finish_ms=finish_ms,
        slabs=len(slabs),
        read_and_hash_ms=read_ms,
    )


__all__ = [
    "NativeHybridRestoreError",
    "NativeHybridRestoreResult",
    "NativePageSlab",
    "build_page_copy_spans",
    "plan_page_slabs",
    "execute_native_hybrid_placement",
    "execute_native_hybrid_restore",
]
