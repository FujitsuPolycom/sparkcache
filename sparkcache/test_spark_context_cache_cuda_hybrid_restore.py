from __future__ import annotations

import hashlib
import threading
import time
from types import SimpleNamespace

import pytest

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
import sparkcache.spark_context_cache_cuda_hybrid_restore as cuda_hybrid
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ManifestStore,
)
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
    plan_page_snapshot,
)
from sparkcache.spark_context_cache_cuda_hybrid_restore import (
    build_page_copy_spans,
    build_page_object_spans,
    execute_cuda_hybrid_restore,
    plan_page_slabs,
)
from sparkcache.spark_context_cache_cuda_placement import RestoreState


def test_page_copy_spans_cover_payload_once_in_layout_order() -> None:
    layout = PageLayout(
        (
            PageGroup(
                256,
                (
                    PageLayer("a", "torch.uint8", (4,), 4),
                    PageLayer("b", "torch.uint8", (2,), 2),
                ),
            ),
            PageGroup(1, (PageLayer("state", "torch.uint8", (3,), 3),)),
        )
    )
    encoded = encode_page_snapshot(
        layout,
        (2, 1),
        {"a": bytes(8), "b": bytes(4), "state": bytes(3)},
    )
    plan = plan_page_snapshot(layout, encoded, (2, 1))

    spans = build_page_copy_spans(plan)

    assert [span.destination_index for span in spans] == [0, 1, 2]
    assert [span.snapshot_offset_bytes for span in spans] == [0, 8, 12]
    assert [span.destination_byte_offset for span in spans] == [0, 0, 0]
    assert sum(span.byte_count for span in spans) == 15
    assert spans[0].arena_offset_bytes == 0

    slabs = plan_page_slabs(plan, arena_bytes=7)
    assert [(slab.payload_start, slab.payload_end) for slab in slabs] == [
        (0, 7),
        (7, 14),
        (14, 15),
    ]
    flattened = [span for slab in slabs for span in slab.spans]
    assert [span.snapshot_offset_bytes for span in flattened] == [0, 7, 8, 12, 14]
    assert sum(span.byte_count for span in flattened) == 15


def test_flat_macro_objects_authenticate_before_direct_page_submission(
    tmp_path, monkeypatch
) -> None:
    layout = PageLayout((PageGroup(2, (PageLayer("page", "torch.uint8", (4,), 4),)),))
    encoded = encode_page_snapshot(layout, (4,), {"page": bytes(range(16))})
    plan = plan_page_snapshot(layout, encoded, (4,))
    cut = plan.header_bytes + 5
    identity = CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-block-pages-v1",
        rope_layout="test-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
    )
    digest = hashlib.sha256(b"native-flat-macro").hexdigest()
    store = ManifestStore(tmp_path)
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", cut)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=digest,
        span_tokens=256,
        snapshot=encoded,
    )
    lookup = store.lookup(identity, digest, verify_chunks=False)
    assert lookup.is_hit
    manifest = lookup._manifest
    assert manifest is not None
    descriptors = manifest["snapshot_objects"]
    chunk_root = tmp_path / "chunks"

    class Arena:
        def __init__(self, size: int) -> None:
            self.payload = bytearray(size)

    class Transaction:
        def __init__(self) -> None:
            self.arenas = (Arena(cut), Arena(cut))
            self.submissions = []
            self.state = RestoreState.PARKED
            self.can_resume = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, *_args):
            if exc_type is not None:
                self.state = RestoreState.ABORTED
                self.can_resume = False
            return None

        def acquire_arena(self, index):
            return self.arenas[index]

        def submit_page_slab(self, *, arena_index, arena_used_bytes, spans):
            source = self.arenas[arena_index].payload
            self.submissions.extend(
                (
                    span.snapshot_offset_bytes,
                    bytes(
                        source[
                            span.arena_offset_bytes : span.arena_offset_bytes
                            + span.byte_count
                        ]
                    ),
                )
                for span in spans
            )

        def finish(self):
            self.state = RestoreState.FINISHED
            self.can_resume = True
            return SimpleNamespace(
                # The CUDA ABI counts the complete authenticated arena bytes,
                # including the snapshot header carried by the first object.
                source_bytes=len(encoded),
                slabs_submitted=2,
                scatter_kernel_launches=2,
                slot_uploads=1,
                destination_table_uploads=1,
                device_error=0,
                staged_h2d_bytes=0,
            )

    class Adapter:
        def __init__(self) -> None:
            self.transactions = []

        def begin_parked_page_restore(self, *_args, **_kwargs):
            transaction = Transaction()
            self.transactions.append(transaction)
            return transaction

    adapter = Adapter()
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )
    original_sha256 = cuda_hybrid.hashlib.sha256
    missing_payload = object()

    def object_sha256(payload=missing_payload):
        if payload is missing_payload:
            raise AssertionError(
                "authenticated page objects must not be hashed a second time"
            )
        return original_sha256(payload)

    monkeypatch.setattr(cuda_hybrid.hashlib, "sha256", object_sha256)

    result = execute_cuda_hybrid_restore(
        adapter=adapter,
        request_id="flat-macro-cuda",
        lookup=lookup,
        cache_root=tmp_path,
        layout=layout,
        group_slots=((3, 4, 5, 6),),
        expected_span_tokens=256,
        arena_bytes=cut,
    )

    transaction = adapter.transactions[-1]
    restored_payload = b"".join(
        payload for _offset, payload in sorted(transaction.submissions)
    )
    assert restored_payload == encoded[plan.header_bytes :]
    assert sum(len(payload) for _offset, payload in transaction.submissions) == (
        len(encoded) - plan.header_bytes
    )
    assert result.placement_stats.source_bytes == len(encoded)
    assert result.source_bytes == len(encoded)
    assert result.slabs == 2

    forged_lookup = type(lookup)(
        True,
        "hit",
        manifest_digest="0" * 64,
        _manifest=manifest,
        root_kind="page_snapshot",
    )
    transactions_before_forgery = len(adapter.transactions)
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="identity is not authenticated",
    ):
        execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-forged-lookup",
            lookup=forged_lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    assert len(adapter.transactions) == transactions_before_forgery

    damaged_path = chunk_root / f"{descriptors[1]['sha256']}.spcc"
    healthy = damaged_path.read_bytes()
    damaged = bytearray(healthy)
    damaged[-1] ^= 0xFF
    damaged_path.write_bytes(damaged)
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="SHA-256 mismatch",
    ):
        execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-corrupt",
            lookup=lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    damaged_transaction = adapter.transactions[-1]
    assert len(damaged_transaction.submissions) == 1
    assert damaged_transaction.state is RestoreState.ABORTED
    assert not damaged_transaction.can_resume

    damaged_path.write_bytes(healthy)
    manifest_path = tmp_path / "manifests" / identity.storage_key / f"{digest}.json"
    wrong_root = dict(manifest)
    wrong_root["snapshot_sha256"] = "0" * 64
    wrong_encoded = cache_manifest._canonical_json(wrong_root)
    manifest_path.write_bytes(wrong_encoded)
    wrong_lookup = type(lookup)(
        True,
        "hit",
        manifest_digest=hashlib.sha256(wrong_encoded).hexdigest(),
        _manifest=wrong_root,
        root_kind="page_snapshot",
    )
    transactions_before_root_damage = len(adapter.transactions)
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="identity is not authenticated",
    ):
        execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-wrong-root-digest",
            lookup=wrong_lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    assert len(adapter.transactions) == transactions_before_root_damage


@pytest.mark.parametrize("damage", ("reorder", "range"))
def test_flat_macro_descriptor_damage_is_rejected_before_transaction(
    tmp_path,
    monkeypatch,
    damage,
) -> None:
    layout = PageLayout(
        (PageGroup(2, (PageLayer("page", "torch.uint8", (128,), 128),)),)
    )
    encoded = encode_page_snapshot(
        layout,
        (8,),
        {"page": bytes(index % 251 for index in range(1024))},
    )
    plan = plan_page_snapshot(layout, encoded, (8,))
    object_bytes = plan.header_bytes + 32
    identity = CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-block-pages-v1",
        rope_layout="test-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
    )
    digest = hashlib.sha256(f"flat-macro-{damage}".encode()).hexdigest()
    store = ManifestStore(tmp_path)
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", object_bytes)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=digest,
        span_tokens=256,
        snapshot=encoded,
    )
    lookup = store.lookup(identity, digest, verify_chunks=False)
    assert lookup.is_hit and lookup._manifest is not None
    damaged_root = dict(lookup._manifest)
    descriptors = [dict(item) for item in damaged_root["snapshot_objects"]]
    damaged_root["snapshot_objects"] = descriptors
    if damage == "reorder":
        descriptors[0], descriptors[1] = descriptors[1], descriptors[0]
    else:
        descriptors[1]["encoded_start"] += 1
    damaged_root.pop("metadata_sha256")
    damaged_root["metadata_sha256"] = hashlib.sha256(
        cache_manifest._canonical_json(damaged_root)
    ).hexdigest()
    encoded_root = cache_manifest._canonical_json(damaged_root)
    manifest_path = (
        tmp_path / "manifests" / identity.storage_key / f"{digest}.json"
    )
    manifest_path.write_bytes(encoded_root)
    damaged_lookup = type(lookup)(
        True,
        "hit",
        manifest_digest=hashlib.sha256(encoded_root).hexdigest(),
        _manifest=damaged_root,
        root_kind="page_snapshot",
    )

    class RejectTransaction:
        def begin_parked_page_restore(self, *_args, **_kwargs):
            raise AssertionError("descriptor damage must fail before a transaction")

    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="geometry is invalid",
    ):
        execute_cuda_hybrid_restore(
            adapter=RejectTransaction(),
            request_id=f"flat-macro-{damage}",
            lookup=damaged_lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=(tuple(range(8)),),
            expected_span_tokens=256,
            arena_bytes=object_bytes,
        )


def test_flat_macro_prefetches_four_objects_before_arena_wait(
    tmp_path,
    monkeypatch,
) -> None:
    layout = PageLayout(
        (PageGroup(2, (PageLayer("page", "torch.uint8", (128,), 128),)),)
    )
    encoded = encode_page_snapshot(
        layout,
        (8,),
        {"page": bytes(index % 251 for index in range(1024))},
    )
    plan = plan_page_snapshot(layout, encoded, (8,))
    object_bytes = plan.header_bytes + 32
    identity = CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-block-pages-v1",
        rope_layout="test-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
    )
    digest = hashlib.sha256(b"parallel-flat-macro").hexdigest()
    store = ManifestStore(tmp_path)
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", object_bytes)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=digest,
        span_tokens=256,
        snapshot=encoded,
    )
    lookup = store.lookup(identity, digest, verify_chunks=False)
    manifest = lookup._manifest
    assert manifest is not None
    assert len(manifest["snapshot_objects"]) >= 3

    class Arena:
        arena_mode = cuda_hybrid.cuda.ARENA_MAPPED_HOST

        def __init__(self) -> None:
            self.payload = bytearray(object_bytes)
            self.capacity_bytes = object_bytes

    class Transaction:
        def __init__(self) -> None:
            self.arenas = (Arena(), Arena())
            self.submissions: list[tuple[int, tuple[object, ...]]] = []
            self.state = RestoreState.PARKED
            self.can_resume = False
            self.arena_acquisitions = 0

        def __enter__(self):
            return self

        def __exit__(self, exc_type, *_args):
            if exc_type is not None:
                self.state = RestoreState.ABORTED
            return None

        def acquire_arena(self, index):
            self.arena_acquisitions += 1
            if self.arena_acquisitions > 1:
                assert four_readers.wait(timeout=0.5), (
                    "the next authenticated object batch must be read before "
                    "waiting for a mapped CUDA arena"
                )
                time.sleep(0.001)
            return self.arenas[index]

        def submit_page_slab(self, *, arena_index, arena_used_bytes, spans):
            self.submissions.append((arena_index, tuple(spans)))

        def finish(self):
            self.state = RestoreState.FINISHED
            self.can_resume = True
            count = len(manifest["snapshot_objects"])
            return SimpleNamespace(
                source_bytes=len(encoded),
                slabs_submitted=count,
                scatter_kernel_launches=count,
                slot_uploads=1,
                destination_table_uploads=1,
                device_error=0,
                staged_h2d_bytes=0,
            )

    class Adapter:
        def __init__(self) -> None:
            self.transaction = Transaction()

        def begin_parked_page_restore(self, *_args, **_kwargs):
            return self.transaction

    original_read = cuda_hybrid._pread_exact_into
    lock = threading.Lock()
    four_readers = threading.Event()
    call_count = 0
    active_reads = 0
    maximum_active_reads = 0
    active_prefetch_bytes = 0
    maximum_prefetch_bytes = 0
    require_overlap = True

    def read_with_overlap(path, encoded_bytes, target):
        nonlocal call_count, active_reads, maximum_active_reads
        nonlocal active_prefetch_bytes, maximum_prefetch_bytes, require_overlap
        with lock:
            call_index = call_count
            call_count += 1
            if call_index > 0:
                assert all(
                    target.obj is not arena.payload
                    for arena in adapter.transaction.arenas
                )
                active_reads += 1
                maximum_active_reads = max(maximum_active_reads, active_reads)
                active_prefetch_bytes += encoded_bytes
                maximum_prefetch_bytes = max(
                    maximum_prefetch_bytes,
                    active_prefetch_bytes,
                )
                if active_reads == 4:
                    four_readers.set()
        if call_index > 0 and require_overlap:
            assert four_readers.wait(timeout=0.5)
        try:
            return original_read(path, encoded_bytes, target)
        finally:
            if call_index > 0:
                with lock:
                    active_reads -= 1
                    active_prefetch_bytes -= encoded_bytes

    adapter = Adapter()
    monkeypatch.setattr(cuda_hybrid, "_pread_exact_into", read_with_overlap)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    result = execute_cuda_hybrid_restore(
        adapter=adapter,
        request_id="parallel-flat-macro",
        lookup=lookup,
        cache_root=tmp_path,
        layout=layout,
        group_slots=(tuple(range(8)),),
        expected_span_tokens=256,
        arena_bytes=object_bytes,
        io_workers=8,
    )

    assert maximum_active_reads == 4
    assert maximum_prefetch_bytes <= 256 * 1024 * 1024
    assert result.arena_wait_ms > 0
    assert result.host_copy_ms > 0
    assert result.submit_call_ms >= 0
    assert result.copy_and_submit_ms == pytest.approx(
        result.arena_wait_ms + result.host_copy_ms + result.submit_call_ms
    )
    assert result.slabs == len(manifest["snapshot_objects"])
    submitted_offsets = [
        span.snapshot_offset_bytes
        for _arena_index, spans in adapter.transaction.submissions
        for span in spans
    ]
    assert submitted_offsets == sorted(submitted_offsets)

    damaged_descriptor = manifest["snapshot_objects"][2]
    damaged_path = tmp_path / "chunks" / f"{damaged_descriptor['sha256']}.spcc"
    healthy = damaged_path.read_bytes()
    damaged = bytearray(healthy)
    damaged[-1] ^= 1
    damaged_path.write_bytes(damaged)
    rejected = Adapter()
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="page object SHA-256 mismatch",
    ):
        execute_cuda_hybrid_restore(
            adapter=rejected,
            request_id="parallel-flat-macro-corrupt",
            lookup=lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=(tuple(range(8)),),
            expected_span_tokens=256,
            arena_bytes=object_bytes,
            io_workers=8,
        )
    assert rejected.transaction.state is RestoreState.ABORTED
    assert not rejected.transaction.can_resume
    assert len(rejected.transaction.submissions) == 1

    damaged_path.write_bytes(healthy)
    call_count = 0
    active_reads = 0
    maximum_active_reads = 0
    require_overlap = False
    fallback = Adapter()
    fallback_result = execute_cuda_hybrid_restore(
        adapter=fallback,
        request_id="sequential-flat-macro",
        lookup=lookup,
        cache_root=tmp_path,
        layout=layout,
        group_slots=(tuple(range(8)),),
        expected_span_tokens=256,
        arena_bytes=object_bytes,
        io_workers=1,
    )
    assert maximum_active_reads == 1
    assert fallback_result.slabs == len(manifest["snapshot_objects"])
    assert len(fallback.transaction.submissions) == len(
        manifest["snapshot_objects"]
    )


def test_page_object_spans_exclude_header_and_cover_payload_contiguously() -> None:
    layout = PageLayout((PageGroup(2, (PageLayer("page", "torch.uint8", (4,), 4),)),))
    encoded = encode_page_snapshot(layout, (4,), {"page": bytes(range(16))})
    plan = plan_page_snapshot(layout, encoded, (4,))
    cut = plan.header_bytes + 5

    first = build_page_object_spans(
        plan,
        encoded_start=0,
        encoded_end=cut,
    )
    second = build_page_object_spans(
        plan,
        encoded_start=cut,
        encoded_end=len(encoded),
    )

    assert [span.snapshot_offset_bytes for span in (*first, *second)] == [0, 5]
    assert sum(span.byte_count for span in (*first, *second)) == 16
    assert first[0].arena_offset_bytes == plan.header_bytes
    assert second[0].arena_offset_bytes == 0
