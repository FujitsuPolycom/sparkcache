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
from typing import Any, Callable, Sequence

from sparkcache import spark_cache_cuda as cuda
from sparkcache.page_base_read_flights import (
    PageBaseReadEvidence,
    PageBaseReadError,
    PageBaseReadResult,
)
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheFormatError,
    CacheIdentity,
    StateRecord,
    _is_page_delta_root,
    _is_page_snapshot_root,
    _validate_page_delta_root,
    _validate_page_snapshot_root,
)
from sparkcache.spark_context_cache_hybrid import PageDeltaPlan, PageSnapshotPlan
from sparkcache.spark_context_cache_hybrid import (
    HybridCodecError,
    PageLayout,
    encode_page_snapshot_header,
    page_snapshot_encoded_size,
    plan_page_delta,
    plan_page_snapshot,
)
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
_PAGE_DELTA_MANIFEST_SCHEMA = "sparkcache-page-delta-manifest/v2"
_PAGE_DELTA_MANIFEST_SCHEMA_V3 = "sparkcache-page-delta-manifest/v3"
_MAX_PAGE_OBJECT_READ_WORKERS = 4
_MAX_PAGE_OBJECT_PREFETCH_BYTES = 256 * 1024 * 1024
_MAX_PAGE_DELTA_DEPTH = 2
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
    arena_wait_ms: float = 0.0
    host_copy_ms: float = 0.0
    submit_call_ms: float = 0.0
    read_source_bytes: int = 0
    skipped_base_object_bytes: int = 0


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


@dataclass(frozen=True)
class CudaAuthenticatedPageObject:
    source: CudaPageObject
    payload: bytes


@dataclass(frozen=True)
class CudaPageSourceSpan:
    """One final logical payload range backed by an authenticated object."""

    source: CudaPageObject
    source_offset_bytes: int
    snapshot_offset_bytes: int
    destination_byte_offset: int
    byte_count: int
    destination_index: int


@dataclass(frozen=True)
class CudaPageDeltaRestorePlan:
    page_plan: PageSnapshotPlan
    result_snapshot_sha256: str
    source_spans: tuple[CudaPageSourceSpan, ...]
    prefetched_objects: tuple[CudaAuthenticatedPageObject, ...]
    referenced_object_bytes: int
    skipped_base_object_bytes: int


@dataclass(frozen=True)
class _CudaPageDeltaStage:
    plan: PageDeltaPlan
    objects: tuple[CudaPageObject, ...]


@dataclass(frozen=True)
class _LogicalSourceRange:
    destination_start: int
    destination_end: int
    source_start: int
    objects: tuple[CudaPageObject, ...]


def _page_objects(
    descriptors: Sequence[Any],
    *,
    cache_root: Any,
    total_bytes: int,
    arena_bytes: int,
    label: str,
) -> tuple[CudaPageObject, ...]:
    if (
        isinstance(total_bytes, bool)
        or not isinstance(total_bytes, int)
        or total_bytes <= 0
        or arena_bytes <= 0
    ):
        raise CudaHybridRestoreError(f"{label} object geometry is invalid")
    root = (Path(cache_root) / "chunks").resolve()
    expected_start = 0
    objects = []
    expected_keys = {"sha256", "bytes", "encoded_start", "encoded_end"}
    for index, raw in enumerate(descriptors):
        if not isinstance(raw, dict) or set(raw) != expected_keys:
            raise CudaHybridRestoreError(
                f"{label} object {index} has an invalid descriptor"
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
            or size > _MAX_PAGE_OBJECT_PREFETCH_BYTES
            or isinstance(start, bool)
            or not isinstance(start, int)
            or isinstance(end, bool)
            or not isinstance(end, int)
            or start != expected_start
            or end != start + size
        ):
            raise CudaHybridRestoreError(
                f"{label} object {index} geometry is invalid"
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
    if not objects or expected_start != total_bytes:
        raise CudaHybridRestoreError(f"{label} objects do not cover the payload")
    return tuple(objects)


def _read_authenticated_page_object(
    source: CudaPageObject,
) -> CudaAuthenticatedPageObject:
    payload = bytearray(source.encoded_bytes)
    view = memoryview(payload)
    try:
        _pread_exact_into(source.path, source.encoded_bytes, view)
    finally:
        view.release()
    if hashlib.sha256(payload).hexdigest() != source.sha256:
        raise CudaHybridRestoreError(
            f"page object SHA-256 mismatch for {source.path}"
        )
    return CudaAuthenticatedPageObject(source, bytes(payload))


def _read_authenticated_page_base(
    objects: Sequence[CudaPageObject],
) -> PageBaseReadResult:
    """Read every flat-base object and publish only authenticated bytes."""

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_MAX_PAGE_OBJECT_READ_WORKERS, len(objects))
    ) as read_pool:
        authenticated = tuple(
            read_pool.map(_read_authenticated_page_object, objects)
        )
    return PageBaseReadResult(
        value=authenticated,
        encoded_bytes=sum(item.source.encoded_bytes for item in authenticated),
    )


def _validated_shared_page_base(
    result: PageBaseReadResult | bytes,
    objects: Sequence[CudaPageObject],
) -> tuple[CudaAuthenticatedPageObject, ...]:
    # The only producer is _read_authenticated_page_base, which hashes every
    # immutable object before publication. Re-hashing here would turn every
    # follower into another full-base authentication pass and defeat the
    # SparkCache CUDA flight; descriptor and length checks reject cross-base reuse.
    if not isinstance(result, PageBaseReadResult) or not isinstance(
        result.value, tuple
    ):
        raise CudaHybridRestoreError(
            "SparkCache CUDA page-base reader returned an incompatible representation"
        )
    authenticated = result.value
    if (
        len(authenticated) != len(objects)
        or any(
            not isinstance(item, CudaAuthenticatedPageObject)
            or item.source != source
            or not isinstance(item.payload, bytes)
            or len(item.payload) != source.encoded_bytes
            for item, source in zip(authenticated, objects, strict=True)
        )
        or result.encoded_bytes != sum(item.encoded_bytes for item in objects)
    ):
        raise CudaHybridRestoreError(
            "SparkCache CUDA page-base reader returned incompatible authenticated objects"
        )
    return authenticated


def _split_logical_source_range(
    source_range: _LogicalSourceRange,
    *,
    snapshot_base: int,
    destination_index: int,
) -> tuple[CudaPageSourceSpan, ...]:
    source_start = source_range.source_start
    source_end = source_start + (
        source_range.destination_end - source_range.destination_start
    )
    cursor = source_start
    result = []
    for source in source_range.objects:
        start = max(cursor, source.encoded_start)
        end = min(source_end, source.encoded_end)
        if start >= end:
            continue
        logical_offset = start - source_start
        result.append(
            CudaPageSourceSpan(
                source=source,
                source_offset_bytes=start - source.encoded_start,
                snapshot_offset_bytes=(
                    snapshot_base + source_range.destination_start + logical_offset
                ),
                destination_byte_offset=(
                    source_range.destination_start + logical_offset
                ),
                byte_count=end - start,
                destination_index=destination_index,
            )
        )
        cursor = end
        if cursor == source_end:
            break
    if cursor != source_end:
        raise CudaHybridRestoreError("page source objects do not cover a logical span")
    return tuple(result)


def plan_cuda_page_delta_restore(
    lookup: Any,
    *,
    cache_root: Any,
    layout: PageLayout,
    group_slots: Sequence[Sequence[int]],
    expected_span_tokens: int,
    arena_bytes: int,
    base_reader: Callable[
        [PageBaseReadEvidence, Callable[[], PageBaseReadResult]],
        PageBaseReadResult | bytes,
    ]
    | None = None,
) -> CudaPageDeltaRestorePlan:
    """Authenticate a page-delta graph and plan final-order source fragments."""

    manifest = getattr(lookup, "_manifest", None)
    if (
        not getattr(lookup, "is_hit", False)
        or getattr(lookup, "root_kind", None) != "page_delta"
        or not isinstance(manifest, dict)
        or manifest.get("schema")
        not in (_PAGE_DELTA_MANIFEST_SCHEMA, _PAGE_DELTA_MANIFEST_SCHEMA_V3)
        or manifest.get("committed_tokens") != expected_span_tokens
    ):
        raise CudaHybridRestoreError(
            "page-delta direct restore requires a compatible cache hit"
        )
    try:
        identity_wire = dict(manifest["identity"])
        if "record_schema" in identity_wire:
            identity_wire["record_schema"] = tuple(identity_wire["record_schema"])
        identity = CacheIdentity(**identity_wire)
        context_digest = manifest["context_digest"]
        if (
            not isinstance(context_digest, str)
            or _DIGEST.fullmatch(context_digest) is None
        ):
            raise ValueError("manifest digest fields are invalid")
        manifest_path = (
            Path(cache_root)
            / "manifests"
            / identity.storage_key
            / f"{context_digest}.json"
        )
        encoded_manifest = manifest_path.read_bytes()
        persisted = json.loads(encoded_manifest)
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise CudaHybridRestoreError(
            f"page-delta root identity was rejected: {error}"
        ) from error
    if (
        identity.publication_schema
        not in ("page-tail-cow-v1", "page-tail-cow-v2")
        or identity.required_records
        != frozenset((StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS))
        or persisted != manifest
        or hashlib.sha256(encoded_manifest).hexdigest()
        != getattr(lookup, "manifest_digest", None)
    ):
        raise CudaHybridRestoreError("page-delta root identity is not authenticated")

    stages_newest_first: list[_CudaPageDeltaStage] = []
    prefetched: dict[Path, CudaAuthenticatedPageObject] = {}
    root = manifest
    root_digest = context_digest
    try:
        if root.get("schema") == _PAGE_DELTA_MANIFEST_SCHEMA_V3:
            base_root, _descriptors = _validate_page_delta_root(
                root,
                identity=identity,
                context_digest=root_digest,
            )
            flat_stages: list[
                tuple[dict[str, Any], tuple[CudaPageObject, ...]]
            ] = []
            for stage in reversed(root["delta_stages"]):
                objects = _page_objects(
                    stage["delta_objects"],
                    cache_root=cache_root,
                    total_bytes=stage["delta_encoded_bytes"],
                    arena_bytes=arena_bytes,
                    label="flat page delta",
                )
                flat_stages.append((stage, objects))
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=min(
                    _MAX_PAGE_OBJECT_READ_WORKERS,
                    len(flat_stages),
                )
            ) as executor:
                first_objects = tuple(
                    executor.map(
                        _read_authenticated_page_object,
                        (objects[0] for _stage, objects in flat_stages),
                    )
                )
            for (stage, objects), authenticated in zip(
                flat_stages,
                first_objects,
                strict=True,
            ):
                prefetched[objects[0].path] = authenticated
                delta_plan = plan_page_delta(
                    layout,
                    authenticated.payload,
                    base_block_counts=stage["base_block_counts"],
                    result_block_counts=stage["result_block_counts"],
                    base_boundary_tokens=stage["base_committed_tokens"],
                    result_boundary_tokens=stage["committed_tokens"],
                    total_bytes=stage["delta_encoded_bytes"],
                )
                stages_newest_first.append(
                    _CudaPageDeltaStage(delta_plan, objects)
                )
            root_digest = root["base_context_digest"]
            root = base_root
        else:
            for depth in range(_MAX_PAGE_DELTA_DEPTH):
                if not _is_page_delta_root(root):
                    break
                if root.get("schema") != _PAGE_DELTA_MANIFEST_SCHEMA:
                    raise CudaHybridRestoreError(
                        "page-delta direct restore requires v2 macro objects"
                    )
                if root.get("layout_sha256") != layout.digest:
                    raise CudaHybridRestoreError("page-delta root layout differs")
                base_root, descriptors = _validate_page_delta_root(
                    root,
                    identity=identity,
                    context_digest=root_digest,
                )
                objects = _page_objects(
                    descriptors,
                    cache_root=cache_root,
                    total_bytes=root["delta_encoded_bytes"],
                    arena_bytes=arena_bytes,
                    label="page delta",
                )
                authenticated = _read_authenticated_page_object(objects[0])
                prefetched[objects[0].path] = authenticated
                delta_plan = plan_page_delta(
                    layout,
                    authenticated.payload,
                    base_block_counts=root["base_block_counts"],
                    result_block_counts=root["result_block_counts"],
                    base_boundary_tokens=root["base_committed_tokens"],
                    result_boundary_tokens=root["committed_tokens"],
                    total_bytes=root["delta_encoded_bytes"],
                )
                stages_newest_first.append(
                    _CudaPageDeltaStage(delta_plan, objects)
                )
                if base_root.get("committed_tokens") != root["base_committed_tokens"]:
                    raise CudaHybridRestoreError(
                        "page-delta base boundary differs"
                    )
                root_digest = root["base_context_digest"]
                root = base_root
        if _is_page_delta_root(root):
            raise CudaHybridRestoreError("page-delta graph exceeds the depth limit")
        if not _is_page_snapshot_root(root):
            raise CudaHybridRestoreError(
                "page-delta direct restore requires a v2 page-snapshot base"
            )
        base_descriptors = _validate_page_snapshot_root(
            root,
            identity=identity,
            context_digest=root_digest,
        )
        base_objects = _page_objects(
            base_descriptors,
            cache_root=cache_root,
            total_bytes=root["snapshot_encoded_bytes"],
            arena_bytes=arena_bytes,
            label="page snapshot",
        )
        stages = tuple(reversed(stages_newest_first))
        if not stages:
            raise CudaHybridRestoreError("page-delta graph contains no delta")
        shared_base_used = base_reader is not None and (
            len(stages) == 1
            or manifest.get("schema") == _PAGE_DELTA_MANIFEST_SCHEMA_V3
        )
        if shared_base_used:
            evidence = PageBaseReadEvidence(
                identity_storage_key=identity.storage_key,
                base_context_digest=manifest["base_context_digest"],
                base_root_sha256=manifest["base_root_sha256"],
                base_root_kind="page_snapshot",
                layout_sha256=manifest["layout_sha256"],
                base_block_counts=tuple(manifest["base_block_counts"]),
                base_boundary_tokens=manifest["base_committed_tokens"],
                base_encoded_bytes=root["snapshot_encoded_bytes"],
            )
            try:
                shared_base = base_reader(
                    evidence,
                    lambda: _read_authenticated_page_base(base_objects),
                )
            except PageBaseReadError as error:
                raise CudaHybridRestoreError(
                    f"SparkCache CUDA shared page-base read was rejected: {error}"
                ) from error
            authenticated_base_objects = _validated_shared_page_base(
                shared_base,
                base_objects,
            )
            prefetched.update(
                (item.source.path, item)
                for item in authenticated_base_objects
            )
            authenticated_base = authenticated_base_objects[0]
        else:
            authenticated_base = _read_authenticated_page_object(base_objects[0])
            prefetched[base_objects[0].path] = authenticated_base
        base_counts = stages[0].plan.base_block_counts
        base_page_plan = plan_page_snapshot(
            layout,
            authenticated_base.payload,
            base_counts,
            total_bytes=root["snapshot_encoded_bytes"],
        )
        if (
            root["committed_tokens"] != stages[0].plan.base_boundary_tokens
            or root["snapshot_encoded_bytes"]
            != page_snapshot_encoded_size(layout, base_counts)
            or root["snapshot_sha256"] != stages[0].plan.base_snapshot_sha256
        ):
            raise CudaHybridRestoreError("page-delta base snapshot differs")
    except (CacheFormatError, HybridCodecError, KeyError, TypeError, ValueError) as error:
        raise CudaHybridRestoreError(
            f"page-delta descriptor planning was rejected: {error}"
        ) from error

    ranges: list[list[_LogicalSourceRange]] = []
    for span in base_page_plan.spans:
        ranges.append(
            [
                _LogicalSourceRange(
                    destination_start=0,
                    destination_end=span.source_end - span.source_start,
                    source_start=span.source_start,
                    objects=base_objects,
                )
            ]
        )
    current_counts = base_counts
    current_sha256 = root["snapshot_sha256"]
    for stage in stages:
        if (
            stage.plan.base_block_counts != current_counts
            or stage.plan.base_snapshot_sha256 != current_sha256
        ):
            raise CudaHybridRestoreError("page-delta stage base identity differs")
        for destination_index, tail in enumerate(stage.plan.tails):
            existing = ranges[destination_index]
            cutoff = tail.destination_byte_offset
            kept = []
            for item in existing:
                if item.destination_start >= cutoff:
                    break
                end = min(item.destination_end, cutoff)
                kept.append(
                    _LogicalSourceRange(
                        destination_start=item.destination_start,
                        destination_end=end,
                        source_start=item.source_start,
                        objects=item.objects,
                    )
                )
                if end == cutoff:
                    break
            expected_prefix = (
                kept[-1].destination_end if kept else 0
            )
            if expected_prefix != cutoff:
                raise CudaHybridRestoreError("page-delta prefix coverage differs")
            kept.append(
                _LogicalSourceRange(
                    destination_start=cutoff,
                    destination_end=cutoff + tail.source_end - tail.source_start,
                    source_start=tail.source_start,
                    objects=stage.objects,
                )
            )
            ranges[destination_index] = kept
        current_counts = stage.plan.result_block_counts
        current_sha256 = stage.plan.result_snapshot_sha256

    expected_counts = tuple(len(group) for group in group_slots)
    if current_counts != expected_counts:
        raise CudaHybridRestoreError("page-delta result block counts differ")
    result_header = encode_page_snapshot_header(layout, current_counts)
    result_total = page_snapshot_encoded_size(layout, current_counts)
    page_plan = plan_page_snapshot(
        layout,
        result_header,
        current_counts,
        total_bytes=result_total,
    )
    source_spans = []
    snapshot_base = 0
    for destination_index, (layer_plan, layer_ranges) in enumerate(
        zip(page_plan.spans, ranges, strict=True)
    ):
        expected_destination = 0
        for item in layer_ranges:
            if item.destination_start != expected_destination:
                raise CudaHybridRestoreError("page-delta result coverage has a gap")
            source_spans.extend(
                _split_logical_source_range(
                    item,
                    snapshot_base=snapshot_base,
                    destination_index=destination_index,
                )
            )
            expected_destination = item.destination_end
        expected_bytes = layer_plan.source_end - layer_plan.source_start
        if expected_destination != expected_bytes:
            raise CudaHybridRestoreError("page-delta result coverage differs")
        snapshot_base += expected_bytes
    if snapshot_base != result_total - page_plan.header_bytes:
        raise CudaHybridRestoreError("page-delta result byte count differs")
    referenced = {span.source.path: span.source for span in source_spans}
    base_paths = {source.path: source for source in base_objects}
    return CudaPageDeltaRestorePlan(
        page_plan=page_plan,
        result_snapshot_sha256=current_sha256,
        source_spans=tuple(source_spans),
        prefetched_objects=tuple(prefetched.values()),
        referenced_object_bytes=sum(
            source.encoded_bytes for source in referenced.values()
        ),
        skipped_base_object_bytes=sum(
            source.encoded_bytes
            for path, source in base_paths.items()
            if (
                not shared_base_used
                and path not in referenced
                and path != base_objects[0].path
            )
        ),
    )


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
        if (
            not isinstance(context_digest, str)
            or _DIGEST.fullmatch(context_digest) is None
        ):
            raise ValueError("manifest digest fields are invalid")
        validated_objects = _validate_page_snapshot_root(
            manifest,
            identity=identity,
            context_digest=context_digest,
        )
        manifest_path = (
            Path(cache_root)
            / "manifests"
            / identity.storage_key
            / f"{context_digest}.json"
        )
        encoded_manifest = manifest_path.read_bytes()
        persisted = json.loads(encoded_manifest)
    except (
        CacheFormatError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        json.JSONDecodeError,
    ) as error:
        raise CudaHybridRestoreError(
            f"flat page root identity was rejected: {error}"
        ) from error
    if (
        identity.publication_schema
        not in ("", "page-tail-cow-v1", "page-tail-cow-v2")
        or identity.required_records
        != frozenset((StateRecord.TARGET_CKV, StateRecord.LOGICAL_POSITIONS))
        or persisted != manifest
        or hashlib.sha256(encoded_manifest).hexdigest()
        != getattr(lookup, "manifest_digest", None)
    ):
        raise CudaHybridRestoreError("flat page root identity is not authenticated")
    total = manifest.get("snapshot_encoded_bytes")
    raw_objects = validated_objects
    if (
        isinstance(total, bool)
        or not isinstance(total, int)
        or total <= 0
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
    arena_wait_ms = 0.0
    host_copy_ms = 0.0
    submit_call_ms = 0.0

    def submit_authenticated_object(
        object_index: int,
        page_object: CudaPageObject,
        payload: bytearray,
    ) -> None:
        nonlocal arena_wait_ms, host_copy_ms, submit_call_ms
        arena_index = object_index % cuda.ARENA_COUNT
        started = time.perf_counter()
        arena = transaction.acquire_arena(arena_index)
        arena_wait_ms += 1e3 * (time.perf_counter() - started)
        buffer = cuda.arena_memoryview(
            arena,
            length=page_object.encoded_bytes,
        )
        try:
            started = time.perf_counter()
            buffer[:] = payload
            host_copy_ms += 1e3 * (time.perf_counter() - started)
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
            submit_call_ms += 1e3 * (time.perf_counter() - started)
        finally:
            buffer.release()
            payload.clear()

    with transaction:
        submit_authenticated_object(0, first, first_payload)

        read_workers = min(
            io_workers,
            _MAX_PAGE_OBJECT_READ_WORKERS,
        )

        def read_and_authenticate(
            item: tuple[CudaPageObject, bytearray],
        ) -> None:
            page_object, buffer = item
            target = memoryview(buffer)
            try:
                _pread_exact_into(
                    page_object.path,
                    page_object.encoded_bytes,
                    target,
                )
            finally:
                target.release()
            if hashlib.sha256(buffer).hexdigest() != page_object.sha256:
                raise CudaHybridRestoreError(
                    f"page object SHA-256 mismatch for {page_object.path}"
                )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=read_workers
        ) as read_pool:
            object_index = 1
            while object_index < len(objects):
                batch: list[tuple[int, CudaPageObject]] = []
                batch_bytes = 0
                while (
                    object_index < len(objects)
                    and len(batch) < read_workers
                ):
                    page_object = objects[object_index]
                    if (
                        batch
                        and batch_bytes + page_object.encoded_bytes
                        > _MAX_PAGE_OBJECT_PREFETCH_BYTES
                    ):
                        break
                    if page_object.encoded_bytes > _MAX_PAGE_OBJECT_PREFETCH_BYTES:
                        raise CudaHybridRestoreError(
                            "flat page object exceeds the host prefetch bound"
                        )
                    batch.append((object_index, page_object))
                    batch_bytes += page_object.encoded_bytes
                    object_index += 1
                staged = [
                    (index, page_object, bytearray(page_object.encoded_bytes))
                    for index, page_object in batch
                ]
                started = time.perf_counter()
                tuple(
                    read_pool.map(
                        read_and_authenticate,
                        (
                            (page_object, buffer)
                            for _index, page_object, buffer in staged
                        ),
                    )
                )
                read_ms += 1e3 * (time.perf_counter() - started)
                # Every object in the bounded batch is authenticated before
                # any of its bytes reach a mapped CUDA arena. Reading into
                # request-private host buffers lets storage work overlap the
                # preceding arena's in-flight placement without exposing an
                # unauthenticated partial batch to the GPU.
                for index, page_object, buffer in staged:
                    submit_authenticated_object(index, page_object, buffer)
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
        copy_and_submit_ms=arena_wait_ms + host_copy_ms + submit_call_ms,
        finish_ms=finish_ms,
        slabs=len(objects),
        read_and_hash_ms=read_ms,
        arena_wait_ms=arena_wait_ms,
        host_copy_ms=host_copy_ms,
        submit_call_ms=submit_call_ms,
    )


def _delta_submission_batches(
    spans: Sequence[CudaPageSourceSpan],
    *,
    arena_bytes: int,
) -> tuple[tuple[CudaPageSourceSpan, ...], ...]:
    """Pack final-order fragments within arena, prefetch, and ABI bounds."""

    batches: list[tuple[CudaPageSourceSpan, ...]] = []
    batch: list[CudaPageSourceSpan] = []
    batch_bytes = 0
    batch_objects: dict[Path, CudaPageObject] = {}
    batch_object_bytes = 0
    expected_snapshot = 0
    destination_covered: dict[int, int] = {}
    for span in spans:
        if (
            span.byte_count <= 0
            or span.byte_count > arena_bytes
            or span.snapshot_offset_bytes != expected_snapshot
            or span.destination_byte_offset
            != destination_covered.get(span.destination_index, 0)
        ):
            raise CudaHybridRestoreError(
                "page-delta source spans violate final coverage"
            )
        new_object_bytes = (
            0
            if span.source.path in batch_objects
            else span.source.encoded_bytes
        )
        if batch and (
            batch_bytes + span.byte_count > arena_bytes
            or batch_object_bytes + new_object_bytes
            > _MAX_PAGE_OBJECT_PREFETCH_BYTES
            or len(batch) >= 4096
        ):
            batches.append(tuple(batch))
            batch = []
            batch_bytes = 0
            batch_objects = {}
            batch_object_bytes = 0
            new_object_bytes = span.source.encoded_bytes
        if new_object_bytes > _MAX_PAGE_OBJECT_PREFETCH_BYTES:
            raise CudaHybridRestoreError(
                "page-delta source object exceeds the host prefetch bound"
            )
        batch.append(span)
        batch_bytes += span.byte_count
        if span.source.path not in batch_objects:
            batch_objects[span.source.path] = span.source
            batch_object_bytes += span.source.encoded_bytes
        expected_snapshot += span.byte_count
        destination_covered[span.destination_index] = (
            span.destination_byte_offset + span.byte_count
        )
    if batch:
        batches.append(tuple(batch))
    if not batches:
        raise CudaHybridRestoreError("page-delta restore has no source spans")
    return tuple(batches)


def _execute_page_delta_restore(
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
    base_reader: Callable[
        [PageBaseReadEvidence, Callable[[], PageBaseReadResult]],
        PageBaseReadResult | bytes,
    ]
    | None,
) -> CudaHybridRestoreResult:
    """Place a verified base+delta graph without assembling its final bytes."""

    planning_started = time.perf_counter()
    plan = plan_cuda_page_delta_restore(
        lookup,
        cache_root=cache_root,
        layout=layout,
        group_slots=group_slots,
        expected_span_tokens=expected_span_tokens,
        arena_bytes=arena_bytes,
        base_reader=base_reader,
    )
    read_ms = 1e3 * (time.perf_counter() - planning_started)
    batches = _delta_submission_batches(plan.source_spans, arena_bytes=arena_bytes)
    payload_bytes = plan.page_plan.total_bytes - plan.page_plan.header_bytes
    prefetched = {
        item.source.path: item for item in plan.prefetched_objects
    }
    referenced_paths = {span.source.path for span in plan.source_spans}
    for path in tuple(prefetched):
        if path not in referenced_paths:
            prefetched.pop(path)
    read_source_bytes = sum(
        item.source.encoded_bytes for item in plan.prefetched_objects
    )
    transaction = adapter.begin_parked_page_restore(
        request_id,
        group_slots,
        snapshot_bytes=payload_bytes,
    )
    arena_wait_ms = 0.0
    host_copy_ms = 0.0
    submit_call_ms = 0.0
    final_sha256 = hashlib.sha256()
    final_sha256.update(
        encode_page_snapshot_header(layout, plan.page_plan.block_counts)
    )

    # Each submission launches CUDA work asynchronously. Reading and
    # authenticating the following batch at the top of the next iteration
    # therefore overlaps the preceding placement. Keep one executor for the
    # transaction so the bounded read lane does not repeatedly create threads.
    read_workers = min(max(1, io_workers), _MAX_PAGE_OBJECT_READ_WORKERS)
    with transaction:
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=read_workers
        ) as read_pool:
            for batch_index, batch in enumerate(batches):
                sources: dict[Path, CudaPageObject] = {}
                for span in batch:
                    sources.setdefault(span.source.path, span.source)
                loaded: dict[Path, CudaAuthenticatedPageObject] = {
                    path: prefetched[path]
                    for path in sources
                    if path in prefetched
                }
                pending = tuple(
                    source
                    for path, source in sources.items()
                    if path not in loaded
                )
                if pending:
                    started = time.perf_counter()
                    authenticated = tuple(
                        read_pool.map(_read_authenticated_page_object, pending)
                    )
                    read_ms += 1e3 * (time.perf_counter() - started)
                    for item in authenticated:
                        loaded[item.source.path] = item
                        read_source_bytes += item.source.encoded_bytes

                arena_index = batch_index % cuda.ARENA_COUNT
                started = time.perf_counter()
                arena = transaction.acquire_arena(arena_index)
                arena_wait_ms += 1e3 * (time.perf_counter() - started)
                used_bytes = sum(span.byte_count for span in batch)
                arena_buffer = cuda.arena_memoryview(arena, length=used_bytes)
                native_spans = []
                arena_offset = 0
                try:
                    started = time.perf_counter()
                    for span in batch:
                        payload = loaded[span.source.path].payload
                        payload_view = memoryview(payload)
                        try:
                            source_end = span.source_offset_bytes + span.byte_count
                            arena_buffer[
                                arena_offset : arena_offset + span.byte_count
                            ] = payload_view[span.source_offset_bytes:source_end]
                        finally:
                            payload_view.release()
                        native_spans.append(
                            cuda.PageCopySpan(
                                arena_offset,
                                span.snapshot_offset_bytes,
                                span.destination_byte_offset,
                                span.byte_count,
                                span.destination_index,
                                0,
                            )
                        )
                        arena_offset += span.byte_count
                    host_copy_ms += 1e3 * (time.perf_counter() - started)
                    final_sha256.update(arena_buffer)
                finally:
                    arena_buffer.release()
                started = time.perf_counter()
                transaction.submit_page_slab(
                    arena_index=arena_index,
                    arena_used_bytes=used_bytes,
                    spans=native_spans,
                )
                submit_call_ms += 1e3 * (time.perf_counter() - started)
        if final_sha256.hexdigest() != plan.result_snapshot_sha256:
            raise CudaHybridRestoreError(
                "page-delta reconstructed snapshot checksum mismatch"
            )
        started = time.perf_counter()
        stats = transaction.finish()
        finish_ms = 1e3 * (time.perf_counter() - started)
        if transaction.state is not RestoreState.FINISHED or not transaction.can_resume:
            raise CudaHybridRestoreError(
                "page-delta placement did not release the parked request"
            )
    expected_stats = {
        "source_bytes": payload_bytes,
        "slabs_submitted": len(batches),
        "scatter_kernel_launches": len(batches),
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
            "page-delta placement statistics violate the restore contract:"
            f" {mismatches}"
        )
    return CudaHybridRestoreResult(
        placement_stats=stats,
        source_bytes=plan.page_plan.total_bytes,
        copy_and_submit_ms=arena_wait_ms + host_copy_ms + submit_call_ms,
        finish_ms=finish_ms,
        slabs=len(batches),
        read_and_hash_ms=read_ms,
        arena_wait_ms=arena_wait_ms,
        host_copy_ms=host_copy_ms,
        submit_call_ms=submit_call_ms,
        read_source_bytes=read_source_bytes,
        skipped_base_object_bytes=plan.skipped_base_object_bytes,
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
    dcp_degree: int = 1,
    dcp_rank: int = 0,
    base_reader: Callable[
        [PageBaseReadEvidence, Callable[[], PageBaseReadResult]],
        PageBaseReadResult | bytes,
    ]
    | None = None,
) -> CudaHybridRestoreResult:
    """Pipeline authenticated .spcc reads directly into mapped page scatter."""

    manifest = getattr(lookup, "_manifest", None)
    if isinstance(manifest, dict) and (
        manifest.get("schema")
        in (_PAGE_DELTA_MANIFEST_SCHEMA, _PAGE_DELTA_MANIFEST_SCHEMA_V3)
    ):
        return _execute_page_delta_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=cache_root,
            layout=layout,
            group_slots=group_slots,
            expected_span_tokens=expected_span_tokens,
            arena_bytes=arena_bytes,
            io_workers=io_workers,
            base_reader=base_reader,
        )
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

    if dcp_degree != 1 or dcp_rank != 0:
        raise CudaHybridRestoreError(
            "legacy chunk-based page restore supports only DCP1;"
            " publish a v2 page snapshot before using DCP"
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
    "CudaAuthenticatedPageObject",
    "CudaHybridRestoreError",
    "CudaHybridRestoreResult",
    "CudaPageDeltaRestorePlan",
    "CudaPageObject",
    "CudaPageRestoreError",
    "CudaPageRestoreResult",
    "CudaPageSlab",
    "CudaPageSourceSpan",
    "build_page_copy_spans",
    "build_page_object_spans",
    "execute_cuda_direct_restore",
    "execute_cuda_hybrid_placement",
    "execute_cuda_hybrid_restore",
    "execute_cuda_page_placement",
    "plan_cuda_page_delta_restore",
    "plan_page_slabs",
]
