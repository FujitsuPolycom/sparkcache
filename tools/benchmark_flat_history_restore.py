"""Measure flat page-history publication and restore work on CPU fixtures.

Status: research-only. The script builds a ``page-tail-cow-v2`` history whose
geometry is a scaled model of GLM-5.3-Flash CSF at TP2, then counts the work
that publication and both restore readers perform. No GPU, CUDA library, or
model is used; the SparkCache CUDA reader runs against an in-memory placement
adapter that only records submitted bytes.

Model geometry (per rank, divided by ``--scale``; default 64):

* one position-indexed full-attention group of 34 layers whose appended pages
  carry 6,732 bytes per token in total; a stored boundary never rewrites an
  earlier full page;
* one recurrent group of 12 layers holding a single checkpoint page per layer,
  76.3 MB in total, which every stored boundary rewrites completely;
* 64 MiB delta and snapshot objects.

These constants come from a 32-turn GLM-5.3-Flash CSF TP2 agent session on two
DGX Sparks: the census of its SparkCache root recorded stage payloads of
103,898,115 bytes (4,096 new tokens), 117,685,247 bytes (6,144) and
131,472,387 bytes (8,192), and the 176,128-token restore placed 1,262,010,532
page bytes. The boundaries below are that session's stored boundaries.

Usage::

    python tools/benchmark_flat_history_restore.py [--scale 64]
        [--snapshot-interval-tokens N] [--stage-limit N]
        [--cold-headers] [--cold-publication-headers] [--recurrent-first]

``--snapshot-interval-tokens`` publishes a complete snapshot whenever a stored
boundary crosses a multiple of N tokens, matching the connector's periodic
full-capture rule. Output is one JSON object. Every byte field is scaled back
to full-size equivalents (fixture bytes multiplied by ``--scale``); seconds
are fixture wall time on the measuring CPU and are not scaled.

The benchmark defaults to unlimited stages (``--stage-limit 0``) to expose
history growth; the connector defaults to four. ``publish_curve`` reports
each boundary's work. Phase timings are inclusive and overlap, so they must
not be summed. ``manifest_write_bytes`` are actual metadata bytes rather than
scaled payload equivalents; ``extent_file_opens`` excludes snapshot-object
reads performed by Python's file API. ``--cold-publication-headers`` clears
the authenticated header cache before each publication; it does not clear
the operating system's file cache.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest  # noqa: E402
import sparkcache.spark_context_cache_cuda_hybrid_restore as cuda_hybrid  # noqa: E402
import sparkcache.spark_context_cache_native_restore as native_restore  # noqa: E402
from sparkcache import spark_context_cache_hybrid as hybrid  # noqa: E402
from sparkcache.persistent_context_cache.cache_manifest import (  # noqa: E402
    CacheIdentity,
    ManifestStore,
)
from sparkcache.spark_context_cache_codec import context_prefix_digest  # noqa: E402
from sparkcache.spark_context_cache_cuda_placement import RestoreState  # noqa: E402

# Stored boundaries of the measured 32-turn session (one linear chain).
SESSION_BOUNDARIES = (
    6144, 12288, 18432, 24576, 28672, 32768, 40960, 45056, 51200, 57344,
    61440, 65536, 73728, 79872, 83968, 92160, 98304, 102400, 108544, 116736,
    122880, 131072, 135168, 141312, 147456, 151552, 159744, 163840, 169984,
    176128, 182272, 186368,
)
FULL_LAYERS = 34
RECURRENT_LAYERS = 12
FULL_BYTES_PER_TOKEN = 6732
RECURRENT_BYTES = 76_323_851
PAGE_TOKENS = 64
OBJECT_BYTES = 64 * 1024 * 1024
PREFETCH_BYTES = 256 * 1024 * 1024


class _CountingHash:
    """hashlib.sha256 stand-in that records every hashed byte."""

    def __init__(self, counter: Counter, original, data=b"", **kwargs):
        self._inner = original(**kwargs)
        self._counter = counter
        if len(data):
            self.update(data)

    def update(self, data):
        self._counter["sha256_bytes"] += memoryview(data).nbytes
        self._inner.update(data)

    def hexdigest(self):
        return self._inner.hexdigest()

    def digest(self):
        return self._inner.digest()

    def copy(self):
        clone = object.__new__(_CountingHash)
        clone._inner = self._inner.copy()
        clone._counter = self._counter
        return clone


@contextlib.contextmanager
def counting(counter: Counter):
    """Count SHA-256 input bytes, object reads, and object writes."""

    original_sha256 = hashlib.sha256
    original_read_bytes = Path.read_bytes
    original_pread = native_restore._pread_exact_into
    original_publish = cache_manifest._publish_immutable_batch
    original_extent = getattr(cache_manifest, "_read_object_extent", None)

    def sha256(data=b"", **kwargs):
        return _CountingHash(counter, original_sha256, data, **kwargs)

    def read_bytes(self):
        payload = original_read_bytes(self)
        if self.suffix == ".spcc":
            counter["object_reads"] += 1
            counter["object_read_bytes"] += len(payload)
        return payload

    def pread(path, expected_bytes, target):
        counter["object_reads"] += 1
        counter["object_read_bytes"] += expected_bytes
        return original_pread(path, expected_bytes, target)

    def extent(pieces):
        counter["object_reads"] += 1
        counter["object_read_bytes"] += sum(piece[3] for piece in pieces)
        return original_extent(pieces)

    original_extent_into = getattr(cache_manifest, "_read_object_extent_into", None)

    def extent_into(pieces, target):
        counter["object_reads"] += 1
        counter["object_read_bytes"] += sum(piece[3] for piece in pieces)
        return original_extent_into(pieces, target)

    def publish(objects):
        counter["object_writes"] += len(objects)
        counter["object_write_bytes"] += sum(len(payload) for _, payload in objects)
        return original_publish(objects)

    hashlib.sha256 = sha256
    Path.read_bytes = read_bytes
    cuda_hybrid._pread_exact_into = pread
    cache_manifest._publish_immutable_batch = publish
    if original_extent is not None:
        cache_manifest._read_object_extent = extent
        cuda_hybrid._read_object_extent = extent
    if original_extent_into is not None:
        cache_manifest._read_object_extent_into = extent_into
        cuda_hybrid._read_object_extent_into = extent_into
    try:
        yield counter
    finally:
        hashlib.sha256 = original_sha256
        Path.read_bytes = original_read_bytes
        cuda_hybrid._pread_exact_into = original_pread
        cache_manifest._publish_immutable_batch = original_publish
        if original_extent is not None:
            cache_manifest._read_object_extent = original_extent
            cuda_hybrid._read_object_extent = original_extent
        if original_extent_into is not None:
            cache_manifest._read_object_extent_into = original_extent_into
            cuda_hybrid._read_object_extent_into = original_extent_into


@contextlib.contextmanager
def publication_phases(counter: Counter):
    """Measure inclusive publication phases and macro extent file opens.

    Nested reconstruction and header calls are counted once per outer call
    of the same phase. Phase times overlap (headers run during reconstruction)
    and must not be summed. Metadata byte counts are actual fixture bytes.
    """
    targets = (
        (ManifestStore, "restore_page_snapshot", "base_reconstruction"),
        (ManifestStore, "lookup", "lookup"),
        (cache_manifest, "_authenticated_page_delta_header", "header_authentication"),
        (hybrid, "encode_page_delta_from_capture", "delta_encode"),
        (cache_manifest, "_canonical_json", "manifest_encode"),
        (cache_manifest, "_publish_immutable", "manifest_write"),
    )
    originals = []
    depths = Counter()

    def wrap(original, name):
        def measured(*args, **kwargs):
            depths[name] += 1
            started = time.perf_counter()
            hashes = counter["sha256_bytes"]
            reads = counter["object_read_bytes"]
            try:
                result = original(*args, **kwargs)
                if name == "manifest_write":
                    counter["manifest_write_bytes"] += len(args[1])
                return result
            finally:
                depths[name] -= 1
                if not depths[name]:
                    counter[f"{name}_seconds"] += time.perf_counter() - started
                    counter[f"{name}_calls"] += 1
                    counter[f"{name}_sha256_bytes"] += counter["sha256_bytes"] - hashes
                    counter[f"{name}_read_bytes"] += counter["object_read_bytes"] - reads
        return measured

    original_open = os.open

    def opened(path, *args, **kwargs):
        if str(path).endswith(".spcc"):
            counter["extent_file_opens"] += 1
        return original_open(path, *args, **kwargs)

    try:
        for owner, attribute, name in targets:
            original = getattr(owner, attribute)
            originals.append((owner, attribute, original))
            setattr(owner, attribute, wrap(original, name))
        os.open = opened
        yield
    finally:
        os.open = original_open
        for owner, attribute, original in reversed(originals):
            setattr(owner, attribute, original)


@dataclass(frozen=True)
class Geometry:
    scale: int
    recurrent_first: bool

    @property
    def full_page_bytes(self) -> int:
        per_layer_token = FULL_BYTES_PER_TOKEN / FULL_LAYERS / self.scale
        return max(1, round(per_layer_token * PAGE_TOKENS))

    @property
    def recurrent_page_bytes(self) -> int:
        return max(1, round(RECURRENT_BYTES / RECURRENT_LAYERS / self.scale))

    @property
    def object_bytes(self) -> int:
        return OBJECT_BYTES // self.scale

    def layout(self) -> hybrid.PageLayout:
        full = hybrid.PageGroup(
            PAGE_TOKENS,
            tuple(
                hybrid.PageLayer(f"full.{index:02d}", "u8",
                                 (self.full_page_bytes,), self.full_page_bytes)
                for index in range(FULL_LAYERS)
            ),
        )
        recurrent = hybrid.PageGroup(
            PAGE_TOKENS,
            tuple(
                hybrid.PageLayer(f"recurrent.{index:02d}", "u8",
                                 (self.recurrent_page_bytes,),
                                 self.recurrent_page_bytes)
                for index in range(RECURRENT_LAYERS)
            ),
            reuse_policy="recurrent_align",
        )
        groups = (recurrent, full) if self.recurrent_first else (full, recurrent)
        return hybrid.PageLayout(groups)

    def counts(self, tokens: int) -> tuple[int, ...]:
        full = -(-tokens // PAGE_TOKENS)
        return (1, full) if self.recurrent_first else (full, 1)


def _page(layer: int, page: int, size: int, salt: int = 0) -> bytes:
    seed = hashlib.blake2b(
        f"{layer}:{page}:{salt}".encode(), digest_size=16
    ).digest()
    return (seed * (size // len(seed) + 1))[:size]


class _Capture:
    """Captured result pages in the layout order used by native capture."""

    def __init__(self, payload: bytes, reused: tuple[int, ...]):
        self._payload = payload
        self.total_bytes = len(payload)
        self.reused_pages_by_group = reused

    def read_range(self, start: int, end: int) -> bytes:
        return self._payload[start:end]


def _snapshot(geometry: Geometry, tokens: int, boundary_index: int) -> bytes:
    layout = geometry.layout()
    counts = geometry.counts(tokens)
    payloads = {}
    for group, count in zip(layout.groups, counts, strict=True):
        for layer_index, layer in enumerate(group.layers):
            if group.reuse_policy == "recurrent_align":
                payloads[layer.name] = _page(
                    100 + layer_index, 0, layer.bytes_per_page, boundary_index
                )
            else:
                payloads[layer.name] = b"".join(
                    _page(layer_index, page, layer.bytes_per_page)
                    for page in range(count)
                )
    return hybrid.encode_page_snapshot(layout, counts, payloads)


def _capture(geometry: Geometry, base_tokens: int, result_tokens: int,
             boundary_index: int) -> _Capture:
    layout = geometry.layout()
    base_counts = geometry.counts(base_tokens)
    result_counts = geometry.counts(result_tokens)
    reused = []
    parts = []
    for group, base, result in zip(layout.groups, base_counts, result_counts,
                                   strict=True):
        if group.reuse_policy == "recurrent_align":
            shared = 0
        else:
            shared = base if base_tokens % PAGE_TOKENS == 0 else base - 1
        reused.append(shared)
        for layer_index, layer in enumerate(group.layers):
            if group.reuse_policy == "recurrent_align":
                parts.append(_page(100 + layer_index, 0, layer.bytes_per_page,
                                   boundary_index))
            else:
                parts.extend(_page(layer_index, page, layer.bytes_per_page)
                             for page in range(shared, result))
    return _Capture(b"".join(parts), tuple(reused))


class _Arena:
    arena_mode = cuda_hybrid.cuda.ARENA_MAPPED_HOST

    def __init__(self, size: int):
        self.payload = bytearray(size)
        self.capacity_bytes = size


class _Transaction:
    def __init__(self, arena_bytes: int):
        self.arenas = (_Arena(arena_bytes), _Arena(arena_bytes))
        self.submitted_bytes = 0
        self.arena_bytes = 0
        self.submit_count = 0
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
        self.submit_count += 1
        self.arena_bytes += arena_used_bytes
        self.submitted_bytes += sum(span.byte_count for span in spans)

    def finish(self):
        self.state = RestoreState.FINISHED
        self.can_resume = True
        return SimpleNamespace(
            source_bytes=self.arena_bytes,
            slabs_submitted=self.submit_count,
            scatter_kernel_launches=self.submit_count,
            slot_uploads=1,
            destination_table_uploads=1,
            device_error=0,
            staged_h2d_bytes=0,
        )


class _Adapter:
    def __init__(self, arena_bytes: int):
        self.transaction = _Transaction(arena_bytes)

    def begin_parked_page_restore(self, *_args, **_kwargs):
        return self.transaction


def _identity(publication_schema: str = "page-tail-cow-v2") -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="benchmark-glm53-flash-pages",
        rope_layout="benchmark-rope",
        tp_degree=2,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
        publication_schema=publication_schema,
    )


def _cuda_restore(geometry: Geometry, store: ManifestStore, root: Path,
                  digest: str, tokens: int, *, io_workers: int = 8) -> dict:
    layout = geometry.layout()
    counts = geometry.counts(tokens)
    lookup = store.lookup(store_identity(store), digest, verify_chunks=False)
    measured = Counter()
    adapter = _Adapter(geometry.object_bytes)
    original_view = cuda_hybrid.cuda.arena_memoryview
    cuda_hybrid.cuda.arena_memoryview = (
        lambda arena, *, length: memoryview(arena.payload)[:length]
    )
    started = time.perf_counter()
    try:
        with counting(measured):
            result = cuda_hybrid.execute_cuda_hybrid_restore(
                adapter=adapter,
                request_id="benchmark",
                lookup=lookup,
                cache_root=root,
                layout=layout,
                group_slots=tuple(tuple(range(1, count + 1)) for count in counts),
                expected_span_tokens=tokens,
                arena_bytes=geometry.object_bytes,
                io_workers=io_workers,
            )
    finally:
        cuda_hybrid.cuda.arena_memoryview = original_view
    seconds = time.perf_counter() - started
    payload = hybrid.page_snapshot_encoded_size(layout, counts) - len(
        hybrid.encode_page_snapshot_header(layout, counts))
    assert adapter.transaction.submitted_bytes == payload
    manifest = lookup._manifest
    return {
        "tokens": tokens,
        "stages": len(manifest.get("delta_stages", ())),
        "seconds": round(seconds, 4),
        "object_reads": measured["object_reads"],
        "object_read_bytes": measured["object_read_bytes"] * geometry.scale,
        "sha256_bytes": measured["sha256_bytes"] * geometry.scale,
        "placed_bytes": payload * geometry.scale,
        "slabs": result.slabs,
        # Phase times of the reader itself, in fixture milliseconds.
        "phase_ms": {
            "read_and_hash": round(result.read_and_hash_ms, 2),
            "planning": round(float(getattr(result, "planning_ms", 0.0)), 2),
            "final_hash": round(float(getattr(result, "final_hash_ms", 0.0)), 2),
            "host_copy": round(result.host_copy_ms, 2),
            "arena_wait": round(result.arena_wait_ms, 2),
            "submit_call": round(result.submit_call_ms, 2),
            "finish": round(result.finish_ms, 2),
        },
        "hot_snapshot_hit": bool(getattr(result, "hot_snapshot_hit", False)),
    }


_SCHEMAS = {
    "sha256": "page-tail-cow-v2",
    "fast": "page-tail-cow-v3-fast",
    "none": "page-tail-cow-v3-none",
}
_STORE_IDENTITIES: dict[int, CacheIdentity] = {}


def store_identity(store: ManifestStore) -> CacheIdentity:
    """Identity the run registered for this store (sha256 schema when none)."""

    return _STORE_IDENTITIES.get(id(store), _identity())


def _clear_header_cache() -> None:
    clear = getattr(cache_manifest, "_clear_authenticated_delta_headers", None)
    if clear is not None:
        clear()


def _clear_hot_snapshots() -> None:
    clear = getattr(cache_manifest, "_clear_hot_page_snapshots", None)
    if clear is not None:
        clear()


def _hot_snapshot_stats() -> dict:
    stats = getattr(cache_manifest, "hot_page_snapshot_stats", None)
    return stats() if stats is not None else {}


def run(geometry: Geometry, root: Path, *, interval_tokens: int,
        cold_headers: bool, stage_limit: int = 0,
        cold_publication_headers: bool = False, hot_snapshot_bytes: int = 0,
        io_workers: int = 8, payload_integrity: str = "sha256") -> dict:
    """Publish the session chain and measure restores along it.

    Byte fields are scaled back to full-size equivalents (fixture bytes times
    ``geometry.scale``). With ``cold_headers`` the process-local caches of
    authenticated delta headers and of hot snapshots are cleared before every
    measured restore, which models a reader that neither published nor
    previously read the stages. ``hot_snapshot_bytes`` is the fixture-size
    budget of the process-local hot snapshot cache (0 disables it); when it
    is set, every CUDA restore point is measured twice, once with the hot
    cache cleared (``disk``) and once immediately after (``hot``).
    """

    layout = geometry.layout()
    identity = _identity(_SCHEMAS[payload_integrity])
    salt = "flat-history-benchmark"
    tokens = tuple(range(SESSION_BOUNDARIES[-1]))
    digests = [context_prefix_digest(tokens, salt, token_count=count)
               for count in SESSION_BOUNDARIES]
    store_options = {"page_delta_stage_limit_count": stage_limit}
    if hot_snapshot_bytes:
        store_options["hot_page_snapshot_bytes"] = hot_snapshot_bytes
    _clear_hot_snapshots()
    store = ManifestStore(root, **store_options)
    _STORE_IDENTITIES[id(store)] = identity
    store.commit_page_snapshot(
        identity=identity,
        context_digest=digests[0],
        span_tokens=SESSION_BOUNDARIES[0],
        snapshot=_snapshot(geometry, SESSION_BOUNDARIES[0], 0),
    )
    publishes = []
    for index in range(1, len(SESSION_BOUNDARIES)):
        base_tokens = SESSION_BOUNDARIES[index - 1]
        result_tokens = SESSION_BOUNDARIES[index]
        if cold_publication_headers:
            _clear_header_cache()
        measured = Counter()
        started = time.perf_counter()
        with counting(measured), publication_phases(measured):
            if interval_tokens and (
                result_tokens // interval_tokens > base_tokens // interval_tokens
            ):
                # Periodic full capture, as selected by the connector.
                store.commit_page_snapshot(
                    identity=identity,
                    context_digest=digests[index],
                    span_tokens=result_tokens,
                    snapshot=_snapshot(geometry, result_tokens, index),
                )
            else:
                capture = _capture(geometry, base_tokens, result_tokens, index)
                # The store authenticates and reconstructs the publication
                # base while retaining its proof for sparse encoding.
                store.commit_page_extension(
                    identity=identity,
                    base_context_digest=digests[index - 1],
                    token_ids=tokens,
                    identity_salt=salt,
                    layout=layout,
                    base_block_counts=geometry.counts(base_tokens),
                    result_block_counts=geometry.counts(result_tokens),
                    base_boundary_tokens=base_tokens,
                    result_boundary_tokens=result_tokens,
                    result_snapshot=capture,
                )
        measured["seconds"] = time.perf_counter() - started
        measured["tokens"] = result_tokens
        measured["new_tokens"] = result_tokens - base_tokens
        measured["stages"] = len(store.lookup(
            identity, digests[index], verify_chunks=False)._manifest.get("delta_stages", ()))
        publishes.append(measured)

    final_tokens = SESSION_BOUNDARIES[-1]
    final_counts = geometry.counts(final_tokens)
    expected = _snapshot(geometry, final_tokens, len(SESSION_BOUNDARIES) - 1)
    lookup = store.lookup(identity, digests[-1], verify_chunks=False)
    stage_bytes = sum(stage["delta_encoded_bytes"]
                      for stage in lookup._manifest.get("delta_stages", ()))

    if cold_headers:
        _clear_header_cache()
        _clear_hot_snapshots()
    python_restore = Counter()
    started = time.perf_counter()
    with counting(python_restore):
        restored = store.restore_page_snapshot(
            lookup, layout=layout, result_block_counts=final_counts,
            result_boundary_tokens=final_tokens,
        )
    python_seconds = time.perf_counter() - started
    assert restored == expected, "Python restore differs from the expected snapshot"
    del restored

    curve = []
    hot_curve = []
    for index in (1, 8, 16, 24, len(SESSION_BOUNDARIES) - 1):
        if cold_headers:
            _clear_header_cache()
        if cold_headers or hot_snapshot_bytes:
            _clear_hot_snapshots()
        curve.append(_cuda_restore(geometry, store, root, digests[index],
                                   SESSION_BOUNDARIES[index],
                                   io_workers=io_workers))
        if hot_snapshot_bytes:
            # The disk restore above retained its result; this one places it.
            hot_curve.append(_cuda_restore(geometry, store, root, digests[index],
                                           SESSION_BOUNDARIES[index],
                                           io_workers=io_workers))

    scale = geometry.scale

    def publish_summary(items) -> dict:
        return {
            "seconds": round(sum(item["seconds"] for item in items), 3),
            "sha256_bytes": sum(item["sha256_bytes"] for item in items) * scale,
            "object_read_bytes": sum(item["object_read_bytes"] for item in items) * scale,
            "object_write_bytes": sum(item["object_write_bytes"] for item in items) * scale,
            "extent_file_opens": sum(item["extent_file_opens"] for item in items),
            "manifest_write_bytes": sum(item["manifest_write_bytes"] for item in items),
            "phases": {
                phase: {
                    "seconds": round(sum(item[f"{phase}_seconds"] for item in items), 6),
                    "calls": sum(item[f"{phase}_calls"] for item in items),
                    "sha256_bytes": sum(item[f"{phase}_sha256_bytes"] for item in items) * scale,
                    "object_read_bytes": sum(item[f"{phase}_read_bytes"] for item in items) * scale,
                }
                for phase in ("base_reconstruction", "lookup", "header_authentication",
                              "delta_encode", "manifest_encode", "manifest_write")
            },
        }

    return {
        "scale": scale,
        "recurrent_first": geometry.recurrent_first,
        "snapshot_interval_tokens": interval_tokens,
        "stage_limit": stage_limit,
        "cold_headers": cold_headers,
        "cold_publication_headers": cold_publication_headers,
        "hot_snapshot_bytes": hot_snapshot_bytes * scale,
        "io_workers": io_workers,
        "hot_snapshot_stats": _hot_snapshot_stats(),
        "payload_integrity": payload_integrity,
        "final_tokens": final_tokens,
        "final_stage_count": len(lookup._manifest.get("delta_stages", ())),
        "final_snapshot_bytes": len(expected) * scale,
        "final_stage_bytes": stage_bytes * scale,
        "publish_all": publish_summary(publishes),
        "publish_last": publish_summary(publishes[-1:]),
        "publish_curve": [
            {"tokens": item["tokens"], "new_tokens": item["new_tokens"],
             "stages": item["stages"], **publish_summary([item])}
            for item in publishes
        ],
        "python_restore_final": {
            "seconds": round(python_seconds, 4),
            "object_reads": python_restore["object_reads"],
            "object_read_bytes": python_restore["object_read_bytes"] * scale,
            "sha256_bytes": python_restore["sha256_bytes"] * scale,
        },
        "cuda_restore_curve": curve,
        "cuda_restore_curve_hot": hot_curve,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scale", type=int, default=64)
    parser.add_argument("--snapshot-interval-tokens", type=int, default=0)
    parser.add_argument("--stage-limit", type=int, default=0,
                        help="maximum delta stages; 0 disables compaction (connector default: 4)")
    parser.add_argument("--hot-snapshot-bytes", type=int, default=0,
                        help="full-size budget of the process-local hot snapshot cache;"
                             " divided by --scale like every other size (0 disables)")
    parser.add_argument("--io-workers", type=int, default=8,
                        help="reader thread count passed to the CUDA restore (connector default: 8)")
    parser.add_argument("--payload-integrity", choices=sorted(_SCHEMAS), default="sha256",
                        help="publication schema: sha256 (page-tail-cow-v2), fast or none (page-tail-cow-v3)")
    parser.add_argument("--cold-headers", action="store_true")
    parser.add_argument("--cold-publication-headers", action="store_true",
                        help="clear authenticated headers before each publication")
    parser.add_argument("--recurrent-first", action="store_true")
    args = parser.parse_args(argv)
    geometry = Geometry(args.scale, args.recurrent_first)
    saved = (cache_manifest._PAGE_DELTA_OBJECT_BYTES,
             cache_manifest._MAX_PAGE_DELTA_OBJECT_BYTES,
             cache_manifest._PAGE_SNAPSHOT_OBJECT_BYTES,
             cache_manifest._MAX_PAGE_SNAPSHOT_OBJECT_BYTES,
             cuda_hybrid._MAX_PAGE_OBJECT_PREFETCH_BYTES)
    cache_manifest._PAGE_DELTA_OBJECT_BYTES = geometry.object_bytes
    cache_manifest._MAX_PAGE_DELTA_OBJECT_BYTES = geometry.object_bytes
    cache_manifest._PAGE_SNAPSHOT_OBJECT_BYTES = geometry.object_bytes
    cache_manifest._MAX_PAGE_SNAPSHOT_OBJECT_BYTES = geometry.object_bytes
    cuda_hybrid._MAX_PAGE_OBJECT_PREFETCH_BYTES = PREFETCH_BYTES // geometry.scale
    try:
        with tempfile.TemporaryDirectory(prefix="sparkcache-flat-") as directory:
            print(json.dumps(run(geometry, Path(directory),
                                 interval_tokens=args.snapshot_interval_tokens,
                                 cold_headers=args.cold_headers,
                                 stage_limit=args.stage_limit,
                                 cold_publication_headers=args.cold_publication_headers,
                                 hot_snapshot_bytes=args.hot_snapshot_bytes // geometry.scale,
                                 io_workers=args.io_workers,
                                 payload_integrity=args.payload_integrity), indent=1))
    finally:
        (cache_manifest._PAGE_DELTA_OBJECT_BYTES,
         cache_manifest._MAX_PAGE_DELTA_OBJECT_BYTES,
         cache_manifest._PAGE_SNAPSHOT_OBJECT_BYTES,
         cache_manifest._MAX_PAGE_SNAPSHOT_OBJECT_BYTES,
         cuda_hybrid._MAX_PAGE_OBJECT_PREFETCH_BYTES) = saved
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
