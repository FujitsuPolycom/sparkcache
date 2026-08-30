"""SparkCache CUDA restore and placement for authenticated opaque page snapshots."""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import re
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from sparkcache import spark_cache_cuda as cuda
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    StateRecord,
    _canonical_json,
)
from sparkcache.spark_context_cache_hybrid import PageSnapshotPlan
from sparkcache.spark_context_cache_hybrid import PageLayout, plan_page_snapshot
from sparkcache.spark_context_cache_cuda_placement import (
    CudaPlacementContractError,
    RestoreState,
)
from sparkcache.spark_context_cache_cuda_restore import plan_cuda_restore
from sparkcache.spark_context_cache_native_restore import (
    _pread_exact_into,
    _read_and_authenticate,
)

_CHUNK_PREFIX = struct.Struct("<8sII")
_CHUNK_MAGIC = b"SPCKV001"
_TARGET_KIND = cuda.RECORD_TARGET_CKV
_PAGE_SNAPSHOT_MANIFEST_SCHEMA = "sparkcache-page-snapshot-manifest/v2"
_MAX_PAGE_OBJECT_READ_WORKERS = 2
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")


class CudaHybridRestoreError(RuntimeError):
    """SparkCache CUDA page placement was rejected before safe completion."""


@dataclass(frozen=True)
class CudaHybridRestoreResult:
    placement_stats: Any
    source_bytes: int
    copy_and_submit_ms: float
    finish_ms: float
    slabs: int = 1
    read_and_hash_ms: float = 0.0


@dataclass(frozen=True)
class CudaPageSlab:
    payload_start: int
    payload_end: int
    spans: tuple[cuda.PageCopySpan, ...]


@dataclass(frozen=True)
class CudaPageObject:
    path: Path
    sha256: str
    encoded_bytes: int
    encoded_start: int
    encoded_end: int


def build_page_copy_spans(plan: PageSnapshotPlan) -> tuple[cuda.PageCopySpan, ...]:
    """Map authenticated snapshot payload ranges onto page destinations."""

    return tuple(
        cuda.PageCopySpan(
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
) -> tuple[CudaPageSlab, ...]:
    if arena_bytes <= 0:
        raise CudaHybridRestoreError(
            "SparkCache CUDA placement arena bytes must be positive"
        )
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
                cuda.PageCopySpan(
                    start - slab_start,
                    start,
                    start - layer_start,
                    end - start,
                    destination_index,
                    0,
                )
            )
        if not spans:
            raise CudaHybridRestoreError("SparkCache CUDA page slab has no copy spans")
        slabs.append(CudaPageSlab(slab_start, slab_end, tuple(spans)))
    return tuple(slabs)


def build_page_object_spans(
    plan: PageSnapshotPlan,
    *,
    encoded_start: int,
    encoded_end: int,
) -> tuple[cuda.PageCopySpan, ...]:
    """Map one authenticated flat-object range onto page destinations."""

    spans = []
    for destination_index, layer in enumerate(plan.spans):
        start = max(encoded_start, layer.source_start)
        end = min(encoded_end, layer.source_end)
        if start >= end:
            continue
        spans.append(
            cuda.PageCopySpan(
                start - encoded_start,
                start - plan.header_bytes,
                start - layer.source_start,
                end - start,
                destination_index,
                0,
            )
        )
    return tuple(spans)


def execute_cuda_hybrid_placement(
    *,
    adapter: Any,
    request_id: str,
    encoded_pages: bytes,
    plan: PageSnapshotPlan,
    group_slots: Sequence[Sequence[int]],
) -> CudaHybridRestoreResult:
    """Copy one verified snapshot into a mapped arena and scatter it once."""

    if len(encoded_pages) != plan.total_bytes:
        raise CudaHybridRestoreError("hybrid snapshot length differs from page plan")
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
            if first_arena.arena_mode != cuda.ARENA_MAPPED_HOST:
                raise CudaHybridRestoreError(
                    "SparkCache CUDA restore requires a mapped-host arena"
                )
            slabs = plan_page_slabs(plan, arena_bytes=first_arena.capacity_bytes)
            for slab_index, slab in enumerate(slabs):
                arena_index = slab_index % cuda.ARENA_COUNT
                arena = (
                    first_arena
                    if slab_index == 0
                    else transaction.acquire_arena(arena_index)
                )
                used = slab.payload_end - slab.payload_start
                arena_buffer = cuda.arena_memoryview(arena, length=used)
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
            if (
                transaction.state is not RestoreState.FINISHED
                or not transaction.can_resume
            ):
                raise CudaHybridRestoreError(
                    "SparkCache CUDA placement did not release the parked request"
                )
    except CudaHybridRestoreError:
        raise
    except (CudaPlacementContractError, RuntimeError, TypeError, ValueError) as error:
        raise CudaHybridRestoreError(
            f"SparkCache CUDA page placement was rejected: {error}"
        ) from error
    if (
        int(stats.slot_uploads) != 1
        or int(stats.destination_table_uploads) != 1
        or int(stats.slabs_submitted) != len(slabs)
        or int(stats.scatter_kernel_launches) != len(slabs)
        or int(stats.device_error) != 0
        or int(stats.staged_h2d_bytes) != 0
    ):
        raise CudaHybridRestoreError(
            "SparkCache CUDA placement statistics violate the mapped transaction"
        )
    return CudaHybridRestoreResult(
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
            raise CudaHybridRestoreError("hybrid chunk prefix is truncated")
        magic, abi, header_bytes = _CHUNK_PREFIX.unpack(prefix)
        if magic != _CHUNK_MAGIC or abi != 1 or header_bytes > 1 << 20:
            raise CudaHybridRestoreError("hybrid chunk prefix is unsupported")
        raw = os.pread(fd, header_bytes, _CHUNK_PREFIX.size)
        if len(raw) != header_bytes:
            raise CudaHybridRestoreError("hybrid chunk header is truncated")
        header = json.loads(raw)
        payload_offset = _CHUNK_PREFIX.size + header_bytes
        for record in header.get("records", ()):
            if record.get("kind") == "target_ckv":
                offset = int(record["offset"])
                length = int(record["length"])
                if (
                    offset < 0
                    or length <= 0
                    or payload_offset + offset + length > encoded_bytes
                ):
                    break
                return payload_offset, offset, length
    except (OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CudaHybridRestoreError(
            f"hybrid chunk planning did not complete: {error}"
        ) from error
    finally:
        os.close(fd)
    raise CudaHybridRestoreError("hybrid chunk has no valid target_ckv record")


def _plan_page_objects(
    lookup: Any,
    *,
    cache_root: Any,
    expected_span_tokens: int,
    arena_bytes: int,
) -> tuple[tuple[CudaPageObject, ...], int]:
    manifest = getattr(lookup, "_manifest", None)
    if (
        not getattr(lookup, "is_hit", False)
        or getattr(lookup, "root_kind", None) != "page_snapshot"
        or not isinstance(manifest, dict)
        or manifest.get("schema") != _PAGE_SNAPSHOT_MANIFEST_SCHEMA
        or manifest.get("committed_tokens") != expected_span_tokens
    ):
        raise CudaHybridRestoreError(
            "flat page macro restore requires a compatible cache hit"
        )
    try:
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = CacheIdentity(**identity_wire)
        context_digest = manifest["context_digest"]
        metadata_digest = manifest["metadata_sha256"]
        if (
            not isinstance(context_digest, str)
            or _DIGEST.fullmatch(context_digest) is None
            or not isinstance(metadata_digest, str)
            or _DIGEST.fullmatch(metadata_digest) is None
        ):
            raise ValueError("manifest digest fields are invalid")
        authenticated = dict(manifest)
        authenticated.pop("metadata_sha256")
        manifest_path = (
            Path(cache_root)
            / "manifests"
            / identity.storage_key
            / f"{context_digest}.json"
        )
        encoded_manifest = manifest_path.read_bytes()
        persisted = json.loads(encoded_manifest)
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise CudaHybridRestoreError(
            f"flat page root identity was rejected: {error}"
        ) from error
    if (
        identity.publication_schema not in ("", "page-tail-cow-v1")
        or identity.required_records
        != frozenset((StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS))
        or persisted != manifest
        or hashlib.sha256(encoded_manifest).hexdigest()
        != getattr(lookup, "manifest_digest", None)
        or hashlib.sha256(_canonical_json(authenticated)).hexdigest() != metadata_digest
    ):
        raise CudaHybridRestoreError("flat page root identity is not authenticated")
    total = manifest.get("snapshot_encoded_bytes")
    raw_objects = manifest.get("snapshot_objects")
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
        or not isinstance(raw_objects, list)
        or not raw_objects
    ):
        raise CudaHybridRestoreError("flat page macro geometry is invalid")
    root = (Path(cache_root) / "chunks").resolve()
    expected_start = 0
    objects = []
    expected_keys = {"sha256", "bytes", "encoded_start", "encoded_end"}
    for index, raw in enumerate(raw_objects):
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise CudaHybridRestoreError(
                f"flat page object {index} has an invalid descriptor"
            )
        digest = raw["sha256"]
        size = raw["bytes"]
        start = raw["encoded_start"]
        end = raw["encoded_end"]
        if (
            not isinstance(digest, str)
            or _DIGEST.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size <= 0
            or size > arena_bytes
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end != start + size
        ):
            raise CudaHybridRestoreError(
                f"flat page object {index} geometry is invalid"
            )
        objects.append(
            CudaPageObject(
                path=root / f"{digest}.spcc",
                sha256=digest,
                encoded_bytes=size,
                encoded_start=start,
                encoded_end=end,
            )
        )
        expected_start = end
    if expected_start != total:
        raise CudaHybridRestoreError("flat page objects do not cover the snapshot")
    return tuple(objects), total


def _execute_page_object_restore(
    *,
    adapter: Any,
    request_id: str,
    lookup: Any,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    io_workers: int,
) -> CudaHybridRestoreResult:
    """Authenticate each flat extent before submitting its page-copy spans."""

    objects, snapshot_bytes = _plan_page_objects(
        lookup,
        cache_root=cache_root,
        expected_span_tokens=expected_span_tokens,
        arena_bytes=arena_bytes,
    )
    first = objects[0]
    first_payload = bytearray(first.encoded_bytes)
    started = time.perf_counter()
    _pread_exact_into(first.path, first.encoded_bytes, memoryview(first_payload))
    if hashlib.sha256(first_payload).hexdigest() != first.sha256:
        raise CudaHybridRestoreError(f"page object SHA-256 mismatch for {first.path}")
    snapshot_digest = hashlib.sha256()
    snapshot_digest.update(first_payload)
    read_ms = 1e3 * (time.perf_counter() - started)
    page_plan = plan_page_snapshot(
        layout,
        first_payload,
        tuple(len(group) for group in group_slots),
        total_bytes=snapshot_bytes,
    )
    transaction = adapter.begin_parked_page_restore(
        request_id,
        group_slots,
        snapshot_bytes=snapshot_bytes - page_plan.header_bytes,
    )
    submit_ms = 0.0
    with transaction:
        first_arena = transaction.acquire_arena(0)
        first_buffer = cuda.arena_memoryview(
            first_arena,
            length=first.encoded_bytes,
        )
        try:
            first_buffer[:] = first_payload
            first_payload.clear()
            spans = build_page_object_spans(
                page_plan,
                encoded_start=first.encoded_start,
                encoded_end=first.encoded_end,
            )
            if not spans:
                raise CudaHybridRestoreError(
                    "flat page object contains no restorable payload"
                )
            started = time.perf_counter()
            transaction.submit_page_slab(
                arena_index=0,
                arena_used_bytes=first.encoded_bytes,
                spans=spans,
            )
            submit_ms += 1e3 * (time.perf_counter() - started)
        finally:
            first_buffer.release()

        read_workers = min(
            io_workers,
            _MAX_PAGE_OBJECT_READ_WORKERS,
            cuda.ARENA_COUNT,
        )

        def read_and_authenticate(item: tuple[CudaPageObject, memoryview]) -> None:
            page_object, buffer = item
            _pread_exact_into(
                page_object.path,
                page_object.encoded_bytes,
                buffer,
            )
            if hashlib.sha256(buffer).hexdigest() != page_object.sha256:
                raise CudaHybridRestoreError(
                    f"page object SHA-256 mismatch for {page_object.path}"
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=read_workers
        ) as read_pool:
            for batch_start in range(1, len(objects), read_workers):
                batch = objects[batch_start : batch_start + read_workers]
                staged: list[tuple[int, CudaPageObject, memoryview]] = []
                try:
                    for offset, page_object in enumerate(batch):
                        object_index = batch_start + offset
                        arena_index = object_index % cuda.ARENA_COUNT
                        arena = transaction.acquire_arena(arena_index)
                        buffer = cuda.arena_memoryview(
                            arena,
                            length=page_object.encoded_bytes,
                        )
                        staged.append((arena_index, page_object, buffer))
                    started = time.perf_counter()
                    tuple(
                        read_pool.map(
                            read_and_authenticate,
                            (
                                (page_object, buffer)
                                for _arena_index, page_object, buffer in staged
                            ),
                        )
                    )
                    read_ms += 1e3 * (time.perf_counter() - started)
                    for arena_index, page_object, buffer in staged:
                        snapshot_digest.update(buffer)
                        spans = build_page_object_spans(
                            page_plan,
                            encoded_start=page_object.encoded_start,
                            encoded_end=page_object.encoded_end,
                        )
                        if not spans:
                            raise CudaHybridRestoreError(
                                "flat page object contains no restorable payload"
                            )
                        started = time.perf_counter()
                        transaction.submit_page_slab(
                            arena_index=arena_index,
                            arena_used_bytes=page_object.encoded_bytes,
                            spans=spans,
                        )
                        submit_ms += 1e3 * (time.perf_counter() - started)
                finally:
                    for _arena_index, _page_object, buffer in staged:
                        buffer.release()
        manifest = lookup._manifest
        if snapshot_digest.hexdigest() != manifest.get("snapshot_sha256"):
            raise CudaHybridRestoreError("flat page snapshot checksum mismatch")
        started = time.perf_counter()
        stats = transaction.finish()
        finish_ms = 1e3 * (time.perf_counter() - started)
        if transaction.state is not RestoreState.FINISHED or not transaction.can_resume:
            raise CudaHybridRestoreError(
                "flat page placement did not release the parked request"
            )
    expected_stats = {
        # The C++ ABI counts every authenticated arena byte submitted through
        # spark_cache_placement_submit_page_slab. Flat macro objects include
        # the encoded snapshot header in the first arena even though page-copy
        # spans cover only payload bytes after that header.
        "source_bytes": snapshot_bytes,
        "slabs_submitted": len(objects),
        "scatter_kernel_launches": len(objects),
        "slot_uploads": 1,
        "destination_table_uploads": 1,
        "device_error": 0,
        "staged_h2d_bytes": 0,
    }
    mismatches = {
        name: {"expected": expected, "actual": int(getattr(stats, name, -1))}
        for name, expected in expected_stats.items()
        if int(getattr(stats, name, -1)) != expected
    }
    if mismatches:
        raise CudaHybridRestoreError(
            f"flat page placement statistics violate the restore contract: {mismatches}"
        )
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=snapshot_bytes,
        copy_and_submit_ms=submit_ms,
        finish_ms=finish_ms,
        slabs=len(objects),
        read_and_hash_ms=read_ms,
    )


def execute_cuda_hybrid_restore(
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
) -> CudaHybridRestoreResult:
    """Pipeline authenticated .spcc reads directly into mapped page scatter."""

    manifest = getattr(lookup, "_manifest", None)
    if isinstance(manifest, dict) and (
        manifest.get("schema") == _PAGE_SNAPSHOT_MANIFEST_SCHEMA
    ):
        return _execute_page_object_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=cache_root,
            layout=layout,
            group_slots=group_slots,
            expected_span_tokens=expected_span_tokens,
            arena_bytes=arena_bytes,
            io_workers=io_workers,
        )

    slabs = plan_cuda_restore(
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
            arena_index = slab_index % cuda.ARENA_COUNT
            arena = transaction.acquire_arena(arena_index)
            buffer = cuda.arena_memoryview(arena, length=slab.arena_used_bytes)
            views = tuple(
                buffer[c.arena_offset_bytes : c.arena_offset_bytes + c.encoded_bytes]
                for c in slab.chunks
            )
            started = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(io_workers, len(slab.chunks))
            ) as pool:
                tuple(
                    pool.map(
                        lambda item: _read_and_authenticate(*item),
                        zip(slab.chunks, views, strict=True),
                    )
                )
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
                    raise CudaHybridRestoreError("authenticated target extent changed")
                extent_start = stream_offset
                extent_end = stream_offset + target_length
                for destination_index, layer in enumerate(page_plan.spans):
                    start = max(extent_start, layer.source_start)
                    end = min(extent_end, layer.source_end)
                    if start >= end:
                        continue
                    spans.append(
                        cuda.PageCopySpan(
                            chunk.arena_offset_bytes
                            + payload
                            + target_offset
                            + start
                            - extent_start,
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
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=snapshot_bytes,
        copy_and_submit_ms=submit_ms,
        finish_ms=finish_ms,
        slabs=len(slabs),
        read_and_hash_ms=read_ms,
    )


CudaPageRestoreError = CudaHybridRestoreError
CudaPageRestoreResult = CudaHybridRestoreResult
execute_cuda_direct_restore = execute_cuda_hybrid_restore
execute_cuda_page_placement = execute_cuda_hybrid_placement


__all__ = [
    "CudaHybridRestoreError",
    "CudaHybridRestoreResult",
    "CudaPageObject",
    "CudaPageRestoreError",
    "CudaPageRestoreResult",
    "CudaPageSlab",
    "build_page_copy_spans",
    "build_page_object_spans",
    "execute_cuda_direct_restore",
    "execute_cuda_hybrid_placement",
    "execute_cuda_hybrid_restore",
    "execute_cuda_page_placement",
    "plan_page_slabs",
]
