"""Hot snapshot cache, arena-direct tail reads, and concurrent header checks.

The fixtures come from ``tools/benchmark_flat_history_restore``: a scaled
GLM-5.3-Flash geometry, a one-base-plus-stages ``page-tail-cow-v2`` history,
and the CPU placement adapter that records submitted arena bytes.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from types import SimpleNamespace

import pytest

from sparkcache import spark_cache_cuda as cuda
from sparkcache import spark_context_cache_cuda_hybrid_restore as cuda_hybrid
from sparkcache import spark_context_cache_hybrid as hybrid
from sparkcache.persistent_context_cache import cache_manifest
from sparkcache.persistent_context_cache.cache_manifest import ManifestStore
from sparkcache.spark_context_cache_codec import context_prefix_digest
from sparkcache.spark_context_cache_cuda_placement import RestoreState
from sparkcache.test_periodic_page_snapshot import parse
from tools.benchmark_flat_history_restore import (
    Geometry,
    _capture,
    _identity,
    _snapshot,
    counting,
)

KEY = "spark_cache_hot_snapshot_bytes"
ENV = "SPARK_CONTEXT_CACHE_HOT_SNAPSHOT_BYTES"
TURNS = 6
TURN_TOKENS = 256


class _RecordingArena:
    arena_mode = cuda.ARENA_MAPPED_HOST

    def __init__(self, size: int) -> None:
        self.payload = bytearray(size)
        self.capacity_bytes = size


class _RecordingTransaction:
    """CPU stand-in for the placement ABI that keeps every submitted byte."""

    def __init__(self, arena_bytes: int) -> None:
        self.arenas = (_RecordingArena(arena_bytes), _RecordingArena(arena_bytes))
        self.submissions: list[tuple[int, bytes]] = []
        self.arena_used = 0
        self.slabs = 0
        self.state = RestoreState.PARKED
        self.can_resume = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_args):
        if exc_type is not None:
            self.state = RestoreState.ABORTED
        return None

    def acquire_arena(self, index):
        return self.arenas[index]

    def submit_page_slab(self, *, arena_index, arena_used_bytes, spans):
        arena = self.arenas[arena_index].payload
        self.arena_used += arena_used_bytes
        self.slabs += 1
        for span in spans:
            start = span.arena_offset_bytes
            self.submissions.append((
                span.snapshot_offset_bytes,
                bytes(arena[start : start + span.byte_count]),
            ))

    def finish(self):
        self.state = RestoreState.FINISHED
        self.can_resume = True
        count = len({offset for offset, _ in self.submissions})
        return SimpleNamespace(
            source_bytes=self.arena_used,
            slabs_submitted=self.slabs,
            scatter_kernel_launches=self.slabs,
            slot_uploads=1,
            destination_table_uploads=1,
            device_error=0,
            staged_h2d_bytes=0,
            restored_rows=count,
        )


class _RecordingAdapter:
    def __init__(self, arena_bytes: int) -> None:
        self.transaction = _RecordingTransaction(arena_bytes)

    def begin_parked_page_restore(self, *_args, **_kwargs):
        return self.transaction


def _placed_payload(transaction: _RecordingTransaction, payload_bytes: int) -> bytes:
    placed = bytearray(payload_bytes)
    for offset, data in transaction.submissions:
        placed[offset : offset + len(data)] = data
    return bytes(placed)


class _History:
    """A published chain of TURNS boundaries on the scaled geometry."""

    def __init__(self, root, monkeypatch, *, hot_bytes: int, stage_limit: int = 0):
        self.geometry = Geometry(4096, False)
        monkeypatch.setattr(
            cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", self.geometry.object_bytes
        )
        monkeypatch.setattr(
            cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", self.geometry.object_bytes
        )
        cache_manifest._clear_hot_page_snapshots()
        cache_manifest._clear_authenticated_delta_headers()
        self.root = root
        self.layout = self.geometry.layout()
        self.identity = _identity()
        self.store = ManifestStore(
            root,
            page_delta_stage_limit_count=stage_limit,
            hot_page_snapshot_bytes=hot_bytes,
        )
        self.tokens = tuple(range(TURNS * TURN_TOKENS))
        self.salt = "hot-snapshots"
        self.digests = [
            context_prefix_digest(self.tokens, self.salt, token_count=n * TURN_TOKENS)
            for n in range(1, TURNS + 1)
        ]
        self.snapshots = [
            _snapshot(self.geometry, n * TURN_TOKENS, n) for n in range(1, TURNS + 1)
        ]
        self.store.commit_page_snapshot(
            identity=self.identity,
            context_digest=self.digests[0],
            span_tokens=TURN_TOKENS,
            snapshot=self.snapshots[0],
        )

    def boundary(self, index: int) -> int:
        return (index + 1) * TURN_TOKENS

    def publish(self, index: int, *, verified_base=None):
        base = index * TURN_TOKENS
        result = self.boundary(index)
        return self.store.commit_page_extension(
            identity=self.identity,
            base_context_digest=self.digests[index - 1],
            token_ids=self.tokens,
            identity_salt=self.salt,
            layout=self.layout,
            base_block_counts=self.geometry.counts(base),
            result_block_counts=self.geometry.counts(result),
            base_boundary_tokens=base,
            result_boundary_tokens=result,
            result_snapshot=_capture(self.geometry, base, result, index + 1),
            verified_base_snapshot=verified_base,
        )

    def lookup(self, index: int):
        return self.store.lookup(self.identity, self.digests[index], verify_chunks=False)

    def python_restore(self, index: int):
        boundary = self.boundary(index)
        return self.store.restore_page_snapshot(
            self.lookup(index),
            layout=self.layout,
            result_block_counts=self.geometry.counts(boundary),
            result_boundary_tokens=boundary,
        )

    def cuda_restore(self, index: int, *, io_workers: int = 4):
        boundary = self.boundary(index)
        counts = self.geometry.counts(boundary)
        adapter = _RecordingAdapter(self.geometry.object_bytes)
        result = cuda_hybrid.execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id=f"restore-{index}",
            lookup=self.lookup(index),
            cache_root=self.root,
            layout=self.layout,
            group_slots=tuple(tuple(range(1, count + 1)) for count in counts),
            expected_span_tokens=boundary,
            arena_bytes=self.geometry.object_bytes,
            io_workers=io_workers,
        )
        plan = hybrid.plan_page_snapshot(self.layout, self.snapshots[index], counts)
        placed = _placed_payload(
            adapter.transaction, plan.total_bytes - plan.header_bytes
        )
        return result, placed, adapter.transaction

    def expected_payload(self, index: int) -> bytes:
        counts = self.geometry.counts(self.boundary(index))
        plan = hybrid.plan_page_snapshot(self.layout, self.snapshots[index], counts)
        return self.snapshots[index][plan.header_bytes :]


@pytest.fixture(autouse=True)
def _cpu_arena(monkeypatch):
    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )
    yield
    cache_manifest._clear_hot_page_snapshots()
    cache_manifest._set_hot_page_snapshot_limit(0)


def test_hot_snapshot_budget_defaults_off_and_parses(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert parse().hot_page_snapshot_bytes == 0
    monkeypatch.setenv(ENV, "4294967296")
    assert parse().hot_page_snapshot_bytes == 4294967296
    assert parse({KEY: 1 << 20}).hot_page_snapshot_bytes == 1 << 20
    # The budget is a reader setting; it never changes cache identity.
    assert parse({KEY: 1 << 20}).build_identity(0, 0) == parse({KEY: 0}).build_identity(0, 0)


@pytest.mark.parametrize("value", [-1, True, 1.5, "bad", None])
def test_hot_snapshot_budget_rejects_invalid_settings(value):
    with pytest.raises(RuntimeError, match=KEY):
        parse({KEY: value})


def test_d25_hot_snapshot_serves_publication_and_restore_without_reads(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=64 << 20)
    for index in range(1, TURNS):
        # The publisher's base is the previous turn's result, which the
        # previous publication retained: no object is read for it.
        measured = Counter()
        with counting(measured):
            history.publish(index)
        assert measured["object_reads"] == 0, index
        # The Python reader and the CUDA reader both serve the newest
        # boundary from memory and place exactly the expected bytes.
        measured = Counter()
        with counting(measured):
            assert history.python_restore(index) == history.snapshots[index]
        assert measured["object_reads"] == 0
        result, placed, _transaction = history.cuda_restore(index)
        assert result.hot_snapshot_hit
        assert result.read_source_bytes == 0
        assert placed == history.expected_payload(index)
    stats = cache_manifest.hot_page_snapshot_stats()
    assert stats["entries"] >= 1 and stats["hits"] >= 2 * (TURNS - 1)
    assert stats["resident_bytes"] <= stats["limit_bytes"]

    # A reader that holds nothing takes the disk path, places the same bytes,
    # and retains the verified result for the following restore.
    cache_manifest._clear_hot_page_snapshots()
    cache_manifest._clear_authenticated_delta_headers()
    last = TURNS - 1
    disk, placed, _transaction = history.cuda_restore(last)
    assert not disk.hot_snapshot_hit
    assert disk.read_source_bytes > 0
    assert placed == history.expected_payload(last)
    hot, placed_again, _transaction = history.cuda_restore(last)
    assert hot.hot_snapshot_hit
    assert placed_again == placed
    # The retained copy is keyed by the digest the newest header binds.
    header = cache_manifest._authenticated_page_delta_header(
        tmp_path / "chunks", history.lookup(last)._manifest["delta_stages"][-1]["delta_objects"][0]
    )
    stage = history.lookup(last)._manifest["delta_stages"][-1]
    plan = hybrid.plan_page_delta(
        history.layout, header,
        base_block_counts=stage["base_block_counts"],
        result_block_counts=stage["result_block_counts"],
        base_boundary_tokens=stage["base_committed_tokens"],
        result_boundary_tokens=stage["committed_tokens"],
        total_bytes=stage["delta_encoded_bytes"],
    )
    retained = cache_manifest._hot_page_snapshot(
        plan.result_snapshot_sha256, len(history.snapshots[last])
    )
    assert retained is not None
    assert hashlib.sha256(retained).hexdigest() == plan.result_snapshot_sha256
    assert bytes(retained) == history.snapshots[last]


def test_hot_snapshot_never_serves_another_boundary(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=64 << 20)
    for index in range(1, TURNS):
        history.publish(index)
    cache_manifest._clear_hot_page_snapshots()
    history.cuda_restore(TURNS - 1)
    # Only the newest boundary is resident; an older one must read objects.
    older, placed, _transaction = history.cuda_restore(TURNS - 3)
    assert not older.hot_snapshot_hit
    assert placed == history.expected_payload(TURNS - 3)


def test_hot_snapshot_budget_evicts_oldest_and_disabled_budget_retains_nothing(
    tmp_path, monkeypatch
):
    history = _History(tmp_path, monkeypatch, hot_bytes=0)
    for index in range(1, TURNS):
        history.publish(index)
    assert cache_manifest.hot_page_snapshot_stats()["entries"] == 0
    # A budget that holds one snapshot keeps only the newest one.
    one = len(history.snapshots[-1]) + 1
    cache_manifest._set_hot_page_snapshot_limit(one)
    for index in range(1, TURNS):
        history.python_restore(index)
    stats = cache_manifest.hot_page_snapshot_stats()
    assert stats["entries"] == 1
    assert stats["resident_bytes"] <= one
    assert history.cuda_restore(TURNS - 1)[0].hot_snapshot_hit
    assert not history.cuda_restore(1)[0].hot_snapshot_hit


def test_hot_snapshot_hit_requires_snapshot_root_consistency(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=64 << 20)
    for index in range(1, TURNS):
        history.publish(index)
    lookup = history.lookup(TURNS - 1)
    damaged = dict(lookup._manifest)
    damaged["base_root"] = dict(damaged["base_root"])
    damaged["base_root"]["snapshot_sha256"] = "0" * 64
    damaged_lookup = SimpleNamespace(
        is_hit=True, root_kind="page_delta", _manifest=damaged,
        manifest_digest=lookup.manifest_digest,
    )
    counts = history.geometry.counts(history.boundary(TURNS - 1))
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError):
        cuda_hybrid.execute_cuda_hybrid_restore(
            adapter=_RecordingAdapter(history.geometry.object_bytes),
            request_id="damaged",
            lookup=damaged_lookup,
            cache_root=tmp_path,
            layout=history.layout,
            group_slots=tuple(tuple(range(1, count + 1)) for count in counts),
            expected_span_tokens=history.boundary(TURNS - 1),
            arena_bytes=history.geometry.object_bytes,
        )


def test_direct_tail_reads_fill_the_arena_and_reject_tampering(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=0)
    for index in range(1, TURNS):
        history.publish(index)
    direct_reads = []
    original = cuda_hybrid._read_authenticated_tail_into

    def counted(source, target):
        direct_reads.append((source.path, len(target)))
        return original(source, target)

    monkeypatch.setattr(cuda_hybrid, "_read_authenticated_tail_into", counted)
    result, placed, transaction = history.cuda_restore(TURNS - 1, io_workers=8)
    assert placed == history.expected_payload(TURNS - 1)
    assert transaction.state is RestoreState.FINISHED
    assert direct_reads, "surviving tails are read straight into the arena"
    assert len({path for path, _ in direct_reads}) == len(direct_reads)
    assert result.final_hash_ms >= 0 and result.planning_ms >= 0
    assert result.read_and_hash_ms >= result.planning_ms

    # Damage one byte of a surviving tail of the newest stage. The read lands
    # in the arena, the in-place digest check fails, the transaction aborts,
    # and nothing of that batch is submitted.
    stage = history.lookup(TURNS - 1)._manifest["delta_stages"][-1]
    descriptor = stage["delta_objects"][-1]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    adapter = _RecordingAdapter(history.geometry.object_bytes)
    counts = history.geometry.counts(history.boundary(TURNS - 1))
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError, match="tail SHA-256 mismatch"):
        cuda_hybrid.execute_cuda_hybrid_restore(
            adapter=adapter,
            request_id="tampered",
            lookup=history.lookup(TURNS - 1),
            cache_root=tmp_path,
            layout=history.layout,
            group_slots=tuple(tuple(range(1, count + 1)) for count in counts),
            expected_span_tokens=history.boundary(TURNS - 1),
            arena_bytes=history.geometry.object_bytes,
            io_workers=8,
        )
    assert adapter.transaction.state is RestoreState.ABORTED
    plan = hybrid.plan_page_delta(
        history.layout,
        cache_manifest._authenticated_page_delta_header(
            tmp_path / "chunks", stage["delta_objects"][0]
        ),
        base_block_counts=stage["base_block_counts"],
        result_block_counts=stage["result_block_counts"],
        base_boundary_tokens=stage["base_committed_tokens"],
        result_boundary_tokens=stage["committed_tokens"],
        total_bytes=stage["delta_encoded_bytes"],
    )
    damaged_offset = len(damaged) - 1 + int(descriptor["encoded_start"])
    tampered_tails = [
        tail for tail in plan.tails if tail.source_start <= damaged_offset < tail.source_end
    ]
    assert tampered_tails
    expected_plan = hybrid.plan_page_snapshot(
        history.layout, history.snapshots[TURNS - 1], counts
    )
    for tail in tampered_tails:
        layer = next(
            span for span in expected_plan.spans if span.layer_name == tail.layer_name
        )
        final_start = layer.source_start - expected_plan.header_bytes + tail.destination_byte_offset
        final_end = final_start + tail.source_end - tail.source_start
        for offset, data in adapter.transaction.submissions:
            assert offset + len(data) <= final_start or offset >= final_end


def test_cold_headers_authenticate_concurrently_and_match_warm(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=0)
    for index in range(1, TURNS):
        history.publish(index)
    warm, warm_placed, _transaction = history.cuda_restore(TURNS - 1)
    cache_manifest._clear_authenticated_delta_headers()
    authenticated = []
    original = cuda_hybrid._authenticated_header_from_object

    def counted(source):
        authenticated.append(source.path)
        return original(source)

    monkeypatch.setattr(cuda_hybrid, "_authenticated_header_from_object", counted)
    cold, cold_placed, _transaction = history.cuda_restore(TURNS - 1)
    assert cold_placed == warm_placed == history.expected_payload(TURNS - 1)
    stages = history.lookup(TURNS - 1)._manifest["delta_stages"]
    assert len(authenticated) == len(stages)
    assert len(set(authenticated)) == len(authenticated)
    header_bytes = sum(stage["delta_objects"][0]["bytes"] for stage in stages)
    assert cold.read_source_bytes >= warm.read_source_bytes + header_bytes - (
        history.geometry.object_bytes * len(stages)
    )
    # The authenticated headers are now cached, so a repeat is warm again.
    authenticated.clear()
    history.cuda_restore(TURNS - 1)
    assert authenticated == []


def test_cold_header_authentication_rejects_a_damaged_first_object(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=0)
    for index in range(1, TURNS):
        history.publish(index)
    cache_manifest._clear_authenticated_delta_headers()
    stage = history.lookup(TURNS - 1)._manifest["delta_stages"][2]
    descriptor = stage["delta_objects"][0]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    damaged = bytearray(path.read_bytes())
    damaged[len(damaged) // 2] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError, match="SHA-256 mismatch"):
        history.cuda_restore(TURNS - 1)


def test_base_digest_passthrough_matches_rehash_and_rejects_bad_digests():
    geometry = Geometry(4096, False)
    layout = geometry.layout()
    base_tokens, result_tokens = TURN_TOKENS, 2 * TURN_TOKENS
    base = _snapshot(geometry, base_tokens, 1)
    capture = _capture(geometry, base_tokens, result_tokens, 2)
    arguments = dict(
        base_block_counts=geometry.counts(base_tokens),
        result_block_counts=geometry.counts(result_tokens),
        reused_pages_by_group=capture.reused_pages_by_group,
        base_boundary_tokens=base_tokens,
        result_boundary_tokens=result_tokens,
    )
    rehashed = hybrid.encode_page_delta_from_capture(layout, base, capture, **arguments)
    digest = hashlib.sha256(base).hexdigest()
    passed = hybrid.encode_page_delta_from_capture(
        layout, base, capture, base_snapshot_sha256=digest, **arguments
    )
    assert passed == rehashed
    with pytest.raises(hybrid.HybridCodecError, match="base snapshot digest"):
        hybrid.encode_page_delta_from_capture(
            layout, base, capture, base_snapshot_sha256="nope", **arguments
        )
    delta, result = hybrid.encode_page_delta_from_capture_with_result(
        layout, base, capture, base_snapshot_sha256=digest, **arguments
    )
    assert delta == rehashed
    assert bytes(result) == hybrid.materialize_page_extension_capture(
        layout, base, capture,
        base_block_counts=arguments["base_block_counts"],
        result_block_counts=arguments["result_block_counts"],
        reused_pages_by_group=arguments["reused_pages_by_group"],
    )
    header = hybrid.plan_page_delta(
        layout, hybrid.page_delta_header_prefix(delta),
        base_block_counts=arguments["base_block_counts"],
        result_block_counts=arguments["result_block_counts"],
        base_boundary_tokens=base_tokens,
        result_boundary_tokens=result_tokens,
        total_bytes=len(delta),
    )
    assert hashlib.sha256(result).hexdigest() == header.result_snapshot_sha256
    assert header.base_snapshot_sha256 == digest

    full_result = _snapshot(geometry, result_tokens, 2)
    dense = hybrid.encode_page_delta(
        layout, base, full_result,
        base_block_counts=geometry.counts(base_tokens),
        result_block_counts=geometry.counts(result_tokens),
        base_boundary_tokens=base_tokens, result_boundary_tokens=result_tokens,
    )
    assert dense == hybrid.encode_page_delta(
        layout, base, full_result,
        base_block_counts=geometry.counts(base_tokens),
        result_block_counts=geometry.counts(result_tokens),
        base_boundary_tokens=base_tokens, result_boundary_tokens=result_tokens,
        base_snapshot_sha256=digest,
    )


def test_publisher_binds_the_verified_base_digest_without_rehashing(tmp_path, monkeypatch):
    history = _History(tmp_path, monkeypatch, hot_bytes=0)
    # A flat snapshot root and a flat delta root both yield the digest the
    # reader verified their bytes against, from metadata alone.
    snapshot_root = history.lookup(0)
    assert history.store._publication_snapshot_digest(
        snapshot_root, layout=history.layout,
        block_counts=history.geometry.counts(history.boundary(0)),
        boundary_tokens=history.boundary(0),
    ) == hashlib.sha256(history.snapshots[0]).hexdigest()
    history.publish(1)
    delta_root = history.lookup(1)
    base = history.python_restore(1)
    assert base == history.snapshots[1]
    assert history.store._publication_snapshot_digest(
        delta_root, layout=history.layout,
        block_counts=history.geometry.counts(history.boundary(1)),
        boundary_tokens=history.boundary(1),
    ) == hashlib.sha256(base).hexdigest()

    # The publisher binds that digest into the next header and never hashes
    # the base bytes itself: the counted SHA-256 input is below one base.
    rehashed = []
    original = hybrid.encode_page_delta_from_capture

    def observe(layout, base_snapshot, captured, **kwargs):
        rehashed.append(kwargs.get("base_snapshot_sha256"))
        return original(layout, base_snapshot, captured, **kwargs)

    monkeypatch.setattr(cache_manifest, "encode_page_delta_from_capture", observe, raising=False)
    history.publish(2, verified_base=base)
    assert history.python_restore(2) == history.snapshots[2]
    stage = history.lookup(2)._manifest["delta_stages"][-1]
    header = cache_manifest._authenticated_page_delta_header(
        tmp_path / "chunks", stage["delta_objects"][0]
    )
    plan = hybrid.plan_page_delta(
        history.layout, header,
        base_block_counts=stage["base_block_counts"],
        result_block_counts=stage["result_block_counts"],
        base_boundary_tokens=stage["base_committed_tokens"],
        result_boundary_tokens=stage["committed_tokens"],
        total_bytes=stage["delta_encoded_bytes"],
    )
    assert plan.base_snapshot_sha256 == hashlib.sha256(base).hexdigest()
