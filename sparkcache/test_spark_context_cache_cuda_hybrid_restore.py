from __future__ import annotations

import concurrent.futures
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
import sparkcache.spark_context_cache_cuda_hybrid_restore as cuda_hybrid
from sparkcache.page_base_read_flights import (
    PageBaseReadFlightKey,
    PageBaseReadFlights,
)
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ManifestStore,
)
from sparkcache.spark_context_cache_codec import context_prefix_digest
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    decode_page_snapshot,
    encode_page_delta,
    encode_page_snapshot,
    plan_page_delta,
    plan_page_snapshot,
)
from sparkcache.spark_context_cache_cuda_hybrid_restore import (
    build_page_copy_spans,
    build_page_object_spans,
    execute_cuda_hybrid_restore,
    plan_cuda_page_delta_restore,
    plan_page_slabs,
)
from sparkcache.spark_context_cache_cuda_placement import RestoreState


def _page_delta_fixture(
    tmp_path,
    monkeypatch,
    *,
    publication_schema: str = "page-tail-cow-v1",
    minimum_object_bytes: int = 0,
):
    layout = PageLayout(
        (PageGroup(256, (PageLayer("page", "torch.uint8", (128,), 128),)),)
    )
    base_blocks = 8
    result_blocks = 16
    base_payload = b"".join(bytes((index,)) * 128 for index in range(base_blocks))
    result_payload = base_payload[:128] + b"".join(
        bytes((64 + index,)) * 128 for index in range(1, result_blocks)
    )
    base = encode_page_snapshot(layout, (base_blocks,), {"page": base_payload})
    result = encode_page_snapshot(layout, (result_blocks,), {"page": result_payload})
    delta = encode_page_delta(
        layout,
        base,
        result,
        base_block_counts=(base_blocks,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=base_blocks * 256,
        result_boundary_tokens=result_blocks * 256,
    )
    base_plan = plan_page_snapshot(layout, base, (base_blocks,))
    delta_plan = plan_page_delta(
        layout,
        delta,
        base_block_counts=(base_blocks,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=base_blocks * 256,
        result_boundary_tokens=result_blocks * 256,
    )
    object_bytes = max(
        minimum_object_bytes,
        max(base_plan.header_bytes, delta_plan.header_bytes) + 1,
    )
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", object_bytes)
    monkeypatch.setattr(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", object_bytes)
    identity = CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-block-pages-v1",
        rope_layout="test-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
        publication_schema=publication_schema,
    )
    tokens = tuple(range(result_blocks * 256))
    salt = "native-page-delta"
    base_digest = context_prefix_digest(
        tokens, salt, token_count=base_blocks * 256
    )
    result_digest = context_prefix_digest(
        tokens, salt, token_count=result_blocks * 256
    )
    store = ManifestStore(tmp_path)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=base_digest,
        span_tokens=base_blocks * 256,
        snapshot=base,
    )
    store.commit_page_extension(
        identity=identity,
        base_context_digest=base_digest,
        token_ids=tokens,
        identity_salt=salt,
        layout=layout,
        base_block_counts=(base_blocks,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=base_blocks * 256,
        result_boundary_tokens=result_blocks * 256,
        result_snapshot=result,
    )
    lookup = store.lookup(identity, result_digest, verify_chunks=False)
    assert lookup.is_hit and lookup.root_kind == "page_delta"
    return SimpleNamespace(
        store=store,
        identity=identity,
        salt=salt,
        result_digest=result_digest,
        layout=layout,
        base=base,
        result=result,
        lookup=lookup,
        object_bytes=object_bytes,
        result_blocks=result_blocks,
        result_tokens=result_blocks * 256,
    )


def _shared_base_delta_fixture(tmp_path, monkeypatch):
    layout = PageLayout(
        (PageGroup(256, (PageLayer("page", "torch.uint8", (128,), 128),)),)
    )
    base_blocks = 8
    result_blocks = 16
    base_payload = b"".join(
        bytes((index,)) * 128 for index in range(base_blocks)
    )
    red_payload = base_payload + b"".join(
        bytes((64 + index,)) * 128
        for index in range(base_blocks, result_blocks)
    )
    blue_payload = base_payload + b"".join(
        bytes((128 + index,)) * 128
        for index in range(base_blocks, result_blocks)
    )
    base = encode_page_snapshot(layout, (base_blocks,), {"page": base_payload})
    red = encode_page_snapshot(layout, (result_blocks,), {"page": red_payload})
    blue = encode_page_snapshot(layout, (result_blocks,), {"page": blue_payload})
    base_plan = plan_page_snapshot(layout, base, (base_blocks,))
    red_delta = encode_page_delta(
        layout,
        base,
        red,
        base_block_counts=(base_blocks,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=base_blocks * 256,
        result_boundary_tokens=result_blocks * 256,
    )
    red_delta_plan = plan_page_delta(
        layout,
        red_delta,
        base_block_counts=(base_blocks,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=base_blocks * 256,
        result_boundary_tokens=result_blocks * 256,
    )
    object_bytes = max(base_plan.header_bytes, red_delta_plan.header_bytes) + 1
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", object_bytes)
    monkeypatch.setattr(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", object_bytes)
    identity = CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-block-pages-v1",
        rope_layout="test-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
        publication_schema="page-tail-cow-v1",
    )
    base_tokens = tuple(range(base_blocks * 256))
    red_tokens = (*base_tokens, *range(base_blocks * 256, result_blocks * 256))
    blue_tokens = (
        *base_tokens,
        *range(100_000, 100_000 + (result_blocks - base_blocks) * 256),
    )
    salt = "native-shared-base-page-delta"
    base_digest = context_prefix_digest(
        base_tokens, salt, token_count=len(base_tokens)
    )
    red_digest = context_prefix_digest(
        red_tokens, salt, token_count=len(red_tokens)
    )
    blue_digest = context_prefix_digest(
        blue_tokens, salt, token_count=len(blue_tokens)
    )
    store = ManifestStore(tmp_path)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=base_digest,
        span_tokens=base_blocks * 256,
        snapshot=base,
    )
    for token_ids, result in ((red_tokens, red), (blue_tokens, blue)):
        store.commit_page_extension(
            identity=identity,
            base_context_digest=base_digest,
            token_ids=token_ids,
            identity_salt=salt,
            layout=layout,
            base_block_counts=(base_blocks,),
            result_block_counts=(result_blocks,),
            base_boundary_tokens=base_blocks * 256,
            result_boundary_tokens=result_blocks * 256,
            result_snapshot=result,
        )
    red_lookup = store.lookup(identity, red_digest, verify_chunks=False)
    blue_lookup = store.lookup(identity, blue_digest, verify_chunks=False)
    assert red_lookup.is_hit and blue_lookup.is_hit
    return SimpleNamespace(
        store=store,
        identity=identity,
        layout=layout,
        base=base,
        red=red,
        blue=blue,
        red_lookup=red_lookup,
        blue_lookup=blue_lookup,
        object_bytes=object_bytes,
        base_blocks=base_blocks,
        result_blocks=result_blocks,
        result_tokens=result_blocks * 256,
    )


class _PageCaptureArena:
    arena_mode = cuda_hybrid.cuda.ARENA_MAPPED_HOST

    def __init__(self, arena_bytes: int) -> None:
        self.payload = bytearray(arena_bytes)
        self.capacity_bytes = arena_bytes


class _PageCaptureTransaction:
    def __init__(self, arena_bytes: int) -> None:
        self.arenas = (
            _PageCaptureArena(arena_bytes),
            _PageCaptureArena(arena_bytes),
        )
        self.submissions = []
        self.submit_count = 0
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
        self.submit_count += 1
        source = self.arenas[arena_index].payload
        self.submissions.extend(
            (
                span.snapshot_offset_bytes,
                span.destination_index,
                span.destination_byte_offset,
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
            source_bytes=sum(len(item[3]) for item in self.submissions),
            slabs_submitted=self.submit_count,
            scatter_kernel_launches=self.submit_count,
            slot_uploads=1,
            destination_table_uploads=1,
            device_error=0,
            staged_h2d_bytes=0,
        )


class _PageCaptureAdapter:
    def __init__(self, arena_bytes: int) -> None:
        self.transaction = _PageCaptureTransaction(arena_bytes)

    def begin_parked_page_restore(self, *_args, **_kwargs):
        return self.transaction


def test_concurrent_distinct_deltas_share_each_authenticated_base_object_once(
    tmp_path, monkeypatch
) -> None:
    fixture = _shared_base_delta_fixture(tmp_path, monkeypatch)
    evidence = fixture.store.page_delta_base_read_evidence(
        fixture.red_lookup,
        layout=fixture.layout,
        result_block_counts=(fixture.result_blocks,),
        result_boundary_tokens=fixture.result_tokens,
    )
    assert evidence == fixture.store.page_delta_base_read_evidence(
        fixture.blue_lookup,
        layout=fixture.layout,
        result_block_counts=(fixture.result_blocks,),
        result_boundary_tokens=fixture.result_tokens,
    )
    key = PageBaseReadFlightKey("worker-0", "block_pages_v1", evidence)
    flights = PageBaseReadFlights()
    assert flights.register_cohort(key, ("red", "blue")).member_ids == (
        "red",
        "blue",
    )
    base_paths = {
        (tmp_path / "chunks" / f"{item['sha256']}.spcc").resolve()
        for item in fixture.red_lookup._manifest["base_root"]["snapshot_objects"]
    }
    physical_reads = []
    authentications = []
    original_read = cuda_hybrid._pread_exact_into
    original_authenticate = cuda_hybrid._read_authenticated_page_object

    def counted_read(path, encoded_bytes, target):
        if path.resolve() in base_paths:
            physical_reads.append(path.resolve())
        return original_read(path, encoded_bytes, target)

    def counted_authenticate(source):
        if source.path.resolve() in base_paths:
            authentications.append(source.path.resolve())
        return original_authenticate(source)

    monkeypatch.setattr(cuda_hybrid, "_pread_exact_into", counted_read)
    monkeypatch.setattr(
        cuda_hybrid,
        "_read_authenticated_page_object",
        counted_authenticate,
    )
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    def restore(request_id, lookup):
        adapter = _PageCaptureAdapter(fixture.object_bytes)
        result = execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
            io_workers=4,
            base_reader=lambda actual_evidence, reader: flights.resolve(
                request_id,
                PageBaseReadFlightKey(
                    "worker-0", "block_pages_v1", actual_evidence
                ),
                reader,
            ),
        )
        return result, b"".join(
            item[3] for item in sorted(adapter.transaction.submissions)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        red_future = executor.submit(restore, "red", fixture.red_lookup)
        blue_future = executor.submit(restore, "blue", fixture.blue_lookup)
        red_result, restored_red = red_future.result(timeout=5)
        blue_result, restored_blue = blue_future.result(timeout=5)

    red_plan = plan_page_snapshot(
        fixture.layout, fixture.red, (fixture.result_blocks,)
    )
    blue_plan = plan_page_snapshot(
        fixture.layout, fixture.blue, (fixture.result_blocks,)
    )
    assert restored_red == fixture.red[red_plan.header_bytes :]
    assert restored_blue == fixture.blue[blue_plan.header_bytes :]
    assert set(physical_reads) == base_paths
    assert len(physical_reads) == len(base_paths)
    assert set(authentications) == base_paths
    assert len(authentications) == len(base_paths)
    assert red_result.source_bytes == len(fixture.red)
    assert blue_result.source_bytes == len(fixture.blue)
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["physical_base_reads"] == 1
    assert summary[0]["avoided_base_reads"] == 1


def test_native_shared_base_follower_cancellation_does_not_abort_leader(
    tmp_path, monkeypatch
) -> None:
    fixture = _shared_base_delta_fixture(tmp_path, monkeypatch)
    evidence = fixture.store.page_delta_base_read_evidence(
        fixture.red_lookup,
        layout=fixture.layout,
        result_block_counts=(fixture.result_blocks,),
        result_boundary_tokens=fixture.result_tokens,
    )
    key = PageBaseReadFlightKey("worker-0", "block_pages_v1", evidence)
    flights = PageBaseReadFlights()
    flights.register_cohort(key, ("red", "blue"))
    base_paths = {
        (tmp_path / "chunks" / f"{item['sha256']}.spcc").resolve()
        for item in fixture.red_lookup._manifest["base_root"]["snapshot_objects"]
    }
    base_started = threading.Event()
    release_base = threading.Event()
    original_read = cuda_hybrid._pread_exact_into

    def blocked_base_read(path, encoded_bytes, target):
        if path.resolve() in base_paths:
            base_started.set()
            assert release_base.wait(timeout=5)
        return original_read(path, encoded_bytes, target)

    monkeypatch.setattr(cuda_hybrid, "_pread_exact_into", blocked_base_read)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    def restore(request_id, lookup):
        adapter = _PageCaptureAdapter(fixture.object_bytes)
        result = execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
            base_reader=lambda actual_evidence, reader: flights.resolve(
                request_id,
                PageBaseReadFlightKey(
                    "worker-0", "block_pages_v1", actual_evidence
                ),
                reader,
            ),
        )
        return result, b"".join(
            item[3] for item in sorted(adapter.transaction.submissions)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        red_future = executor.submit(restore, "red", fixture.red_lookup)
        assert base_started.wait(timeout=5)
        blue_future = executor.submit(restore, "blue", fixture.blue_lookup)
        assert flights.cancel("blue")
        with pytest.raises(
            cuda_hybrid.CudaHybridRestoreError,
            match="left the registered cohort",
        ):
            blue_future.result(timeout=5)
        release_base.set()
        _result, restored_red = red_future.result(timeout=5)

    red_plan = plan_page_snapshot(
        fixture.layout, fixture.red, (fixture.result_blocks,)
    )
    assert restored_red == fixture.red[red_plan.header_bytes :]
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["outcome"] == "verified"
    assert summary[0]["cancelled_members"] == 1
    assert summary[0]["physical_base_reads"] == 1


def test_corrupt_native_shared_base_rejects_only_its_matching_cohort(
    tmp_path, monkeypatch
) -> None:
    fixture = _shared_base_delta_fixture(tmp_path, monkeypatch)
    evidence = fixture.store.page_delta_base_read_evidence(
        fixture.red_lookup,
        layout=fixture.layout,
        result_block_counts=(fixture.result_blocks,),
        result_boundary_tokens=fixture.result_tokens,
    )
    key = PageBaseReadFlightKey("worker-0", "block_pages_v1", evidence)
    flights = PageBaseReadFlights()
    flights.register_cohort(key, ("red", "blue"))
    descriptor = fixture.red_lookup._manifest["base_root"]["snapshot_objects"][0]
    corrupt_path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    damaged = bytearray(corrupt_path.read_bytes())
    damaged[-1] ^= 1
    corrupt_path.write_bytes(damaged)
    physical_reads = []
    original_read = cuda_hybrid._pread_exact_into

    def counted_read(path, encoded_bytes, target):
        physical_reads.append(path.resolve())
        return original_read(path, encoded_bytes, target)

    monkeypatch.setattr(cuda_hybrid, "_pread_exact_into", counted_read)

    def restore(request_id, lookup):
        return execute_cuda_hybrid_restore(
            adapter=_PageCaptureAdapter(fixture.object_bytes),
            request_id=request_id,
            lookup=lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
            base_reader=lambda actual_evidence, reader: flights.resolve(
                request_id,
                PageBaseReadFlightKey(
                    "worker-0", "block_pages_v1", actual_evidence
                ),
                reader,
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(restore, "red", fixture.red_lookup),
            executor.submit(restore, "blue", fixture.blue_lookup),
        )
        for future in futures:
            with pytest.raises(
                cuda_hybrid.CudaHybridRestoreError,
                match="page object SHA-256 mismatch",
            ):
                future.result(timeout=5)

    assert physical_reads.count(corrupt_path.resolve()) == 1
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["outcome"] == "recompute"
    assert summary[0]["physical_base_reads"] == 1


def test_corrupt_private_delta_does_not_poison_shared_base_or_peer(
    tmp_path, monkeypatch
) -> None:
    fixture = _shared_base_delta_fixture(tmp_path, monkeypatch)
    evidence = fixture.store.page_delta_base_read_evidence(
        fixture.red_lookup,
        layout=fixture.layout,
        result_block_counts=(fixture.result_blocks,),
        result_boundary_tokens=fixture.result_tokens,
    )
    key = PageBaseReadFlightKey("worker-0", "block_pages_v1", evidence)
    flights = PageBaseReadFlights()
    flights.register_cohort(key, ("red", "blue"))
    red_descriptors = fixture.red_lookup._manifest["delta_objects"]
    blue_digests = {
        item["sha256"] for item in fixture.blue_lookup._manifest["delta_objects"]
    }
    corrupt_descriptor = next(
        item
        for item in reversed(red_descriptors[1:])
        if item["sha256"] not in blue_digests
    )
    corrupt_path = (
        tmp_path / "chunks" / f"{corrupt_descriptor['sha256']}.spcc"
    )
    damaged = bytearray(corrupt_path.read_bytes())
    damaged[-1] ^= 1
    corrupt_path.write_bytes(damaged)
    base_paths = {
        (tmp_path / "chunks" / f"{item['sha256']}.spcc").resolve()
        for item in fixture.red_lookup._manifest["base_root"]["snapshot_objects"]
    }
    base_reads = []
    original_read = cuda_hybrid._pread_exact_into

    def counted_read(path, encoded_bytes, target):
        if path.resolve() in base_paths:
            base_reads.append(path.resolve())
        return original_read(path, encoded_bytes, target)

    monkeypatch.setattr(cuda_hybrid, "_pread_exact_into", counted_read)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    def restore(request_id, lookup):
        adapter = _PageCaptureAdapter(fixture.object_bytes)
        result = execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id=request_id,
            lookup=lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
            base_reader=lambda actual_evidence, reader: flights.resolve(
                request_id,
                PageBaseReadFlightKey(
                    "worker-0", "block_pages_v1", actual_evidence
                ),
                reader,
            ),
        )
        return result, b"".join(
            item[3] for item in sorted(adapter.transaction.submissions)
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        red_future = executor.submit(restore, "red", fixture.red_lookup)
        blue_future = executor.submit(restore, "blue", fixture.blue_lookup)
        with pytest.raises(
            cuda_hybrid.CudaHybridRestoreError,
            match="page object SHA-256 mismatch",
        ):
            red_future.result(timeout=5)
        _result, restored_blue = blue_future.result(timeout=5)

    blue_plan = plan_page_snapshot(
        fixture.layout, fixture.blue, (fixture.result_blocks,)
    )
    assert restored_blue == fixture.blue[blue_plan.header_bytes :]
    assert set(base_reads) == base_paths
    assert len(base_reads) == len(base_paths)
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["outcome"] == "verified"
    assert summary[0]["physical_base_reads"] == 1
    assert summary[0]["avoided_base_reads"] == 1


def test_page_delta_planner_authenticates_graph_and_applies_delta_precedence(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)

    plan = plan_cuda_page_delta_restore(
        fixture.lookup,
        cache_root=tmp_path,
        layout=fixture.layout,
        group_slots=(tuple(range(fixture.result_blocks)),),
        expected_span_tokens=fixture.result_tokens,
        arena_bytes=fixture.object_bytes,
    )

    assert plan.result_snapshot_sha256 == hashlib.sha256(fixture.result).hexdigest()
    assert sum(span.byte_count for span in plan.source_spans) == (
        len(fixture.result) - plan.page_plan.header_bytes
    )
    assert [span.snapshot_offset_bytes for span in plan.source_spans] == sorted(
        span.snapshot_offset_bytes for span in plan.source_spans
    )
    manifest = fixture.lookup._manifest
    assert manifest is not None
    base_objects = manifest["base_root"]["snapshot_objects"]
    used_paths = {span.source.path for span in plan.source_spans}
    assert any(
        tmp_path / "chunks" / f"{descriptor['sha256']}.spcc" in used_paths
        for descriptor in base_objects
    )
    assert all(
        tmp_path / "chunks" / f"{descriptor['sha256']}.spcc" not in used_paths
        for descriptor in base_objects[2:]
    )


def test_flat_page_delta_planner_reads_descriptor_stages(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(
        tmp_path,
        monkeypatch,
        publication_schema="page-tail-cow-v2",
        minimum_object_bytes=4096,
    )

    plan = plan_cuda_page_delta_restore(
        fixture.lookup,
        cache_root=tmp_path,
        layout=fixture.layout,
        group_slots=(tuple(range(fixture.result_blocks)),),
        expected_span_tokens=fixture.result_tokens,
        arena_bytes=fixture.object_bytes,
    )

    assert plan.result_snapshot_sha256 == hashlib.sha256(fixture.result).hexdigest()
    assert sum(span.byte_count for span in plan.source_spans) == (
        len(fixture.result) - plan.page_plan.header_bytes
    )

    final_blocks = 24
    middle_plan = plan_page_snapshot(
        fixture.layout,
        fixture.result,
        (fixture.result_blocks,),
    )
    final_payload = fixture.result[middle_plan.header_bytes :] + b"".join(
        bytes((128 + index,)) * 128
        for index in range(fixture.result_blocks, final_blocks)
    )
    final_snapshot = encode_page_snapshot(
        fixture.layout,
        (final_blocks,),
        {"page": final_payload},
    )
    tokens = tuple(range(final_blocks * 256))
    final_digest = context_prefix_digest(
        tokens,
        fixture.salt,
        token_count=final_blocks * 256,
    )
    fixture.store.commit_page_extension(
        identity=fixture.identity,
        base_context_digest=fixture.result_digest,
        token_ids=tokens,
        identity_salt=fixture.salt,
        layout=fixture.layout,
        base_block_counts=(fixture.result_blocks,),
        result_block_counts=(final_blocks,),
        base_boundary_tokens=fixture.result_tokens,
        result_boundary_tokens=final_blocks * 256,
        result_snapshot=final_snapshot,
    )
    final_lookup = fixture.store.lookup(
        fixture.identity,
        final_digest,
        verify_chunks=False,
    )
    final_plan = plan_cuda_page_delta_restore(
        final_lookup,
        cache_root=tmp_path,
        layout=fixture.layout,
        group_slots=(tuple(range(final_blocks)),),
        expected_span_tokens=final_blocks * 256,
        arena_bytes=fixture.object_bytes,
    )
    reconstructed = b"".join(
        span.source.path.read_bytes()[
            span.source_offset_bytes : span.source_offset_bytes + span.byte_count
        ]
        for span in final_plan.source_spans
    )
    snapshot_plan = plan_page_snapshot(
        fixture.layout,
        final_snapshot,
        (final_blocks,),
    )
    assert reconstructed == final_snapshot[snapshot_plan.header_bytes :]


def test_nested_page_deltas_resolve_newest_over_middle_over_flat_base(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)
    middle_plan = plan_page_snapshot(
        fixture.layout,
        fixture.result,
        (fixture.result_blocks,),
    )
    middle_payload = fixture.result[middle_plan.header_bytes :]
    final_blocks = 24
    final_payload = middle_payload[: 3 * 128] + b"".join(
        bytes((128 + index,)) * 128 for index in range(3, final_blocks)
    )
    final_snapshot = encode_page_snapshot(
        fixture.layout,
        (final_blocks,),
        {"page": final_payload},
    )
    second_delta = encode_page_delta(
        fixture.layout,
        fixture.result,
        final_snapshot,
        base_block_counts=(fixture.result_blocks,),
        result_block_counts=(final_blocks,),
        base_boundary_tokens=fixture.result_tokens,
        result_boundary_tokens=final_blocks * 256,
    )
    second_plan = plan_page_delta(
        fixture.layout,
        second_delta,
        base_block_counts=(fixture.result_blocks,),
        result_block_counts=(final_blocks,),
        base_boundary_tokens=fixture.result_tokens,
        result_boundary_tokens=final_blocks * 256,
    )
    arena_bytes = max(fixture.object_bytes, second_plan.header_bytes + 1)
    monkeypatch.setattr(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", arena_bytes)
    tokens = tuple(range(final_blocks * 256))
    final_digest = context_prefix_digest(
        tokens,
        fixture.salt,
        token_count=final_blocks * 256,
    )
    fixture.store.commit_page_extension(
        identity=fixture.identity,
        base_context_digest=fixture.result_digest,
        token_ids=tokens,
        identity_salt=fixture.salt,
        layout=fixture.layout,
        base_block_counts=(fixture.result_blocks,),
        result_block_counts=(final_blocks,),
        base_boundary_tokens=fixture.result_tokens,
        result_boundary_tokens=final_blocks * 256,
        result_snapshot=final_snapshot,
    )
    lookup = fixture.store.lookup(
        fixture.identity,
        final_digest,
        verify_chunks=False,
    )
    assert lookup.is_hit and lookup._manifest is not None

    plan = plan_cuda_page_delta_restore(
        lookup,
        cache_root=tmp_path,
        layout=fixture.layout,
        group_slots=(tuple(range(final_blocks)),),
        expected_span_tokens=final_blocks * 256,
        arena_bytes=arena_bytes,
    )

    reconstructed_payload = b"".join(
        span.source.path.read_bytes()[
            span.source_offset_bytes : span.source_offset_bytes + span.byte_count
        ]
        for span in plan.source_spans
    )
    final_plan = plan_page_snapshot(
        fixture.layout,
        final_snapshot,
        (final_blocks,),
    )
    assert reconstructed_payload == final_snapshot[final_plan.header_bytes :]
    used_paths = {span.source.path for span in plan.source_spans}
    outer = lookup._manifest
    middle = outer["base_root"]
    flat = middle["base_root"]
    roots = [
        {tmp_path / "chunks" / f"{item['sha256']}.spcc" for item in objects}
        for objects in (
            flat["snapshot_objects"],
            middle["delta_objects"],
            outer["delta_objects"],
        )
    ]
    assert all(used_paths.intersection(paths) for paths in roots)
    assert plan.result_snapshot_sha256 == hashlib.sha256(final_snapshot).hexdigest()


def test_page_delta_direct_restore_skips_overridden_base_objects_and_full_buffer(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)
    manifest = fixture.lookup._manifest
    assert manifest is not None
    base_objects = manifest["base_root"]["snapshot_objects"]
    skipped_base_paths = {
        (tmp_path / "chunks" / f"{descriptor['sha256']}.spcc").resolve()
        for descriptor in base_objects[2:]
    }

    adapter = _PageCaptureAdapter(fixture.object_bytes)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )
    original_read = cuda_hybrid._pread_exact_into
    read_paths = []
    maximum_read = 0

    def bounded_read(path, encoded_bytes, target):
        nonlocal maximum_read
        read_paths.append(path.resolve())
        maximum_read = max(maximum_read, encoded_bytes)
        return original_read(path, encoded_bytes, target)

    monkeypatch.setattr(cuda_hybrid, "_pread_exact_into", bounded_read)

    result = execute_cuda_hybrid_restore(
        adapter=adapter,
        request_id="page-delta-direct",
        lookup=fixture.lookup,
        cache_root=tmp_path,
        layout=fixture.layout,
        group_slots=(tuple(range(fixture.result_blocks)),),
        expected_span_tokens=fixture.result_tokens,
        arena_bytes=fixture.object_bytes,
        io_workers=4,
    )

    restored_payload = b"".join(
        item[3] for item in sorted(adapter.transaction.submissions)
    )
    final_plan = plan_page_snapshot(
        fixture.layout,
        fixture.result,
        (fixture.result_blocks,),
    )
    assert restored_payload == fixture.result[final_plan.header_bytes :]
    assert maximum_read <= fixture.object_bytes < len(fixture.result)
    assert skipped_base_paths.isdisjoint(read_paths)
    assert result.source_bytes == len(fixture.result)
    assert result.skipped_base_object_bytes > 0
    assert adapter.transaction.state is RestoreState.FINISHED
    assert adapter.transaction.can_resume


def test_page_delta_restore_reuses_one_bounded_read_pool(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)
    adapter = _PageCaptureAdapter(fixture.object_bytes)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )
    original_pool = concurrent.futures.ThreadPoolExecutor
    original_read = cuda_hybrid._read_authenticated_page_object
    pool_sizes = []
    reads_after_submit = []

    def counted_pool(*args, **kwargs):
        workers = kwargs.get("max_workers", args[0] if args else None)
        pool_sizes.append(workers)
        return original_pool(*args, **kwargs)

    monkeypatch.setattr(
        cuda_hybrid.concurrent.futures,
        "ThreadPoolExecutor",
        counted_pool,
    )

    def observed_read(source):
        if adapter.transaction.submit_count:
            reads_after_submit.append(source.path)
            assert adapter.transaction.state is RestoreState.PARKED
        return original_read(source)

    monkeypatch.setattr(
        cuda_hybrid,
        "_read_authenticated_page_object",
        observed_read,
    )

    execute_cuda_hybrid_restore(
        adapter=adapter,
        request_id="page-delta-bounded-read-pool",
        lookup=fixture.lookup,
        cache_root=tmp_path,
        layout=fixture.layout,
        group_slots=(tuple(range(fixture.result_blocks)),),
        expected_span_tokens=fixture.result_tokens,
        arena_bytes=fixture.object_bytes,
        io_workers=8,
    )

    assert pool_sizes == [4]
    assert reads_after_submit


def test_page_delta_direct_restore_maps_multiple_groups_and_layers(
    tmp_path, monkeypatch
) -> None:
    layout = PageLayout(
        (
            PageGroup(
                2,
                (
                    PageLayer("a", "u8", (8,), 8),
                    PageLayer("b", "u8", (4,), 4),
                ),
            ),
            PageGroup(1, (PageLayer("state", "u8", (6,), 6),)),
        )
    )
    base_counts = (2, 2)
    result_counts = (4, 3)
    base_payloads = {
        "a": b"A" * 8 + b"B" * 8,
        "b": b"C" * 4 + b"D" * 4,
        "state": b"E" * 6 + b"F" * 6,
    }
    result_payloads = {
        "a": b"A" * 8 + b"X" * 8 + b"Y" * 8 + b"Z" * 8,
        "b": b"C" * 4 + b"L" * 4 + b"M" * 4 + b"N" * 4,
        "state": base_payloads["state"] + b"G" * 6,
    }
    base = encode_page_snapshot(layout, base_counts, base_payloads)
    result = encode_page_snapshot(layout, result_counts, result_payloads)
    delta = encode_page_delta(
        layout,
        base,
        result,
        base_block_counts=base_counts,
        result_block_counts=result_counts,
        base_boundary_tokens=256,
        result_boundary_tokens=512,
    )
    object_bytes = max(
        plan_page_snapshot(layout, base, base_counts).header_bytes,
        plan_page_delta(
            layout,
            delta,
            base_block_counts=base_counts,
            result_block_counts=result_counts,
            base_boundary_tokens=256,
            result_boundary_tokens=512,
        ).header_bytes,
    ) + 1
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", object_bytes)
    monkeypatch.setattr(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", object_bytes)
    identity = CacheIdentity(
        target_checkpoint="3" * 64,
        draft_checkpoint="4" * 64,
        quantization_layout="multi-group-pages-v1",
        rope_layout="multi-group-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
        publication_schema="page-tail-cow-v1",
    )
    tokens = tuple(range(512))
    salt = "native-multi-group-page-delta"
    base_digest = context_prefix_digest(tokens, salt, token_count=256)
    result_digest = context_prefix_digest(tokens, salt, token_count=512)
    store = ManifestStore(tmp_path)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=base_digest,
        span_tokens=256,
        snapshot=base,
    )
    store.commit_page_extension(
        identity=identity,
        base_context_digest=base_digest,
        token_ids=tokens,
        identity_salt=salt,
        layout=layout,
        base_block_counts=base_counts,
        result_block_counts=result_counts,
        base_boundary_tokens=256,
        result_boundary_tokens=512,
        result_snapshot=result,
    )
    lookup = store.lookup(identity, result_digest, verify_chunks=False)
    assert lookup.is_hit
    adapter = _PageCaptureAdapter(object_bytes)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    execute_cuda_hybrid_restore(
        adapter=adapter,
        request_id="page-delta-multi-group",
        lookup=lookup,
        cache_root=tmp_path,
        layout=layout,
        group_slots=((1, 2, 3, 4), (5, 6, 7)),
        expected_span_tokens=512,
        arena_bytes=object_bytes,
        io_workers=4,
    )

    expected = decode_page_snapshot(layout, result, result_counts)
    layer_names = [layer.name for group in layout.groups for layer in group.layers]
    for destination_index, layer_name in enumerate(layer_names):
        placed = b"".join(
            item[3]
            for item in sorted(
                adapter.transaction.submissions,
                key=lambda item: (item[1], item[2]),
            )
            if item[1] == destination_index
        )
        assert placed == expected[layer_name]


def test_page_delta_planner_rejects_identity_layout_and_object_hash_mismatch(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)
    lookup_type = type(fixture.lookup)
    forged_lookup = lookup_type(
        True,
        "hit",
        manifest_digest="0" * 64,
        _manifest=fixture.lookup._manifest,
        root_kind="page_delta",
    )
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="identity is not authenticated",
    ):
        plan_cuda_page_delta_restore(
            forged_lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
        )

    wrong_layout = PageLayout(
        (PageGroup(256, (PageLayer("page", "torch.uint8", (64,), 64),)),)
    )
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="layout differs",
    ):
        plan_cuda_page_delta_restore(
            fixture.lookup,
            cache_root=tmp_path,
            layout=wrong_layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
        )

    manifest = fixture.lookup._manifest
    assert manifest is not None
    descriptor = manifest["delta_objects"][0]
    object_path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    healthy = object_path.read_bytes()
    damaged = bytearray(healthy)
    damaged[-1] ^= 1
    object_path.write_bytes(damaged)
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="SHA-256 mismatch",
    ):
        plan_cuda_page_delta_restore(
            fixture.lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
        )


def test_page_delta_late_hash_failure_aborts_parked_native_restore(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)
    manifest = fixture.lookup._manifest
    assert manifest is not None
    descriptor = manifest["delta_objects"][-1]
    object_path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    damaged = bytearray(object_path.read_bytes())
    damaged[-1] ^= 1
    object_path.write_bytes(damaged)
    adapter = _PageCaptureAdapter(fixture.object_bytes)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="SHA-256 mismatch",
    ):
        execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="page-delta-late-corruption",
            lookup=fixture.lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
            io_workers=4,
        )

    assert adapter.transaction.submit_count > 0
    assert adapter.transaction.state is RestoreState.ABORTED
    assert not adapter.transaction.can_resume


def test_page_delta_final_snapshot_hash_rejects_self_consistent_bad_tail(
    tmp_path, monkeypatch
) -> None:
    fixture = _page_delta_fixture(tmp_path, monkeypatch)
    manifest = json.loads(json.dumps(fixture.lookup._manifest))
    descriptor = manifest["delta_objects"][-1]
    old_path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    damaged = bytearray(old_path.read_bytes())
    damaged[-1] ^= 1
    new_digest = hashlib.sha256(damaged).hexdigest()
    (tmp_path / "chunks" / f"{new_digest}.spcc").write_bytes(damaged)
    descriptor["sha256"] = new_digest
    aggregate = b"".join(
        (
            tmp_path / "chunks" / f"{item['sha256']}.spcc"
        ).read_bytes()
        for item in manifest["delta_objects"]
    )
    manifest["delta_sha256"] = hashlib.sha256(aggregate).hexdigest()
    manifest.pop("metadata_sha256")
    manifest["metadata_sha256"] = hashlib.sha256(
        cache_manifest._canonical_json(manifest)
    ).hexdigest()
    encoded_manifest = cache_manifest._canonical_json(manifest)
    manifest_path = (
        tmp_path
        / "manifests"
        / fixture.identity.storage_key
        / f"{fixture.result_digest}.json"
    )
    manifest_path.write_bytes(encoded_manifest)
    forged_lookup = type(fixture.lookup)(
        True,
        "hit",
        manifest_digest=hashlib.sha256(encoded_manifest).hexdigest(),
        _manifest=manifest,
        root_kind="page_delta",
    )
    adapter = _PageCaptureAdapter(fixture.object_bytes)
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="reconstructed snapshot checksum mismatch",
    ):
        execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="page-delta-semantic-corruption",
            lookup=forged_lookup,
            cache_root=tmp_path,
            layout=fixture.layout,
            group_slots=(tuple(range(fixture.result_blocks)),),
            expected_span_tokens=fixture.result_tokens,
            arena_bytes=fixture.object_bytes,
            io_workers=4,
        )

    assert adapter.transaction.submit_count > 0
    assert adapter.transaction.state is RestoreState.ABORTED
    assert not adapter.transaction.can_resume


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
    malformed_root = dict(manifest)
    malformed_root["snapshot_sha256"] = "not-a-digest"
    malformed_root.pop("metadata_sha256")
    malformed_root["metadata_sha256"] = hashlib.sha256(
        cache_manifest._canonical_json(malformed_root)
    ).hexdigest()
    malformed_encoded = cache_manifest._canonical_json(malformed_root)
    manifest_path.write_bytes(malformed_encoded)
    malformed_lookup = type(lookup)(
        True,
        "hit",
        manifest_digest=hashlib.sha256(malformed_encoded).hexdigest(),
        _manifest=malformed_root,
        root_kind="page_snapshot",
    )
    transactions_before_malformed_root = len(adapter.transactions)
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError):
        execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-malformed-root-digest",
            lookup=malformed_lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    assert len(adapter.transactions) == transactions_before_malformed_root

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
        match="metadata checksum mismatch",
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
        match="descriptor geometry differs",
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


def test_legacy_chunk_page_restore_rejects_dcp() -> None:
    layout = PageLayout(
        (PageGroup(2, (PageLayer("page", "torch.uint8", (4,), 4),)),)
    )
    with pytest.raises(
        cuda_hybrid.CudaHybridRestoreError,
        match="legacy chunk-based page restore supports only DCP1",
    ):
        execute_cuda_hybrid_restore(
            adapter=object(),
            request_id="legacy-dcp2",
            lookup=SimpleNamespace(is_hit=True, _manifest={}),
            cache_root="unused",
            layout=layout,
            group_slots=((1,),),
            expected_span_tokens=256,
            arena_bytes=64 * 1024 * 1024,
            dcp_degree=2,
            dcp_rank=0,
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
