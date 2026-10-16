"""Payload integrity modes: content-addressed sha256, CRC32 fast, and none.

The ``page-tail-cow-v3-fast`` and ``page-tail-cow-v3-none`` publication schemas
hash identities, manifests and stage headers only. Bulk payload objects and
layer tails carry a CRC32 (fast) or no check (none). Their cache identity
differs from ``page-tail-cow-v2``, so a reader of either format misses the
other cleanly instead of misreading it.
"""

from __future__ import annotations

import hashlib
from collections import Counter

import pytest

from sparkcache import spark_context_cache_hybrid as hybrid
from sparkcache.persistent_context_cache import cache_manifest
from sparkcache.test_hot_snapshots_and_direct_tails import (
    TURNS,
    _History,
)
from sparkcache.test_periodic_page_snapshot import parse
from tools.benchmark_flat_history_restore import _identity, counting

SCHEMAS = {
    "sha256": "page-tail-cow-v2",
    "fast": "page-tail-cow-v3-fast",
    "none": "page-tail-cow-v3-none",
}


class _ModeHistory(_History):
    def __init__(self, root, monkeypatch, *, mode: str, hot_bytes: int = 0):
        self._schema = SCHEMAS[mode]
        super().__init__(root, monkeypatch, hot_bytes=hot_bytes)

    # _History builds its identity through this hook.
    def _make_identity(self):
        return _identity(self._schema)


@pytest.fixture(autouse=True)
def _cpu_arena(monkeypatch):
    from sparkcache import spark_context_cache_cuda_hybrid_restore as cuda_hybrid

    monkeypatch.setattr(
        cuda_hybrid.cuda,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )
    yield
    cache_manifest._clear_hot_page_snapshots()
    cache_manifest._set_hot_page_snapshot_limit(0)


def _publish_all(history):
    for index in range(1, TURNS):
        history.publish(index)


def _parse_block_pages(extra: dict):
    import types

    from sparkcache import spark_context_cache_config as cfg
    from sparkcache.test_spark_context_cache_config import _make_vllm_config

    class FullAttentionSpec:
        block_size = 512
        storage_block_size = 512
        page_size_bytes = 528

    kv_cache_config = types.SimpleNamespace(
        kv_cache_groups=(
            types.SimpleNamespace(
                kv_cache_spec=FullAttentionSpec(),
                is_eagle_group=False,
                layer_names=("full",),
            ),
        )
    )
    values = {
        "spark_cache_model_profile": "deepseek-v4-fp8-hma",
        "spark_cache_publication_schema": "tail-cow-v2",
    }
    values.update(extra)
    vllm, _ = _make_vllm_config(values, tp=1, dcp=1)
    return cfg.parse_connector_config(vllm, vllm.kv_transfer_config, kv_cache_config)


def test_payload_integrity_config_selects_the_schema(monkeypatch):
    monkeypatch.delenv("SPARK_CONTEXT_CACHE_PAYLOAD_INTEGRITY", raising=False)
    # tail-cow-v2 publication defaults to fast; sha256 stays selectable.
    default = _parse_block_pages({})
    assert default.payload_integrity == "fast"
    assert default.publication_schema == "page-tail-cow-v3-fast"
    base = _parse_block_pages({"spark_cache_payload_integrity": "sha256"})
    assert base.publication_schema == "page-tail-cow-v2"
    fast = _parse_block_pages({"spark_cache_payload_integrity": "fast"})
    assert fast.publication_schema == "page-tail-cow-v3-fast"
    none = _parse_block_pages({"spark_cache_payload_integrity": "none"})
    assert none.publication_schema == "page-tail-cow-v3-none"
    # Three distinct cache identities: no mode ever reads another's entries.
    keys = {
        config.build_identity(0, 0).storage_key for config in (base, fast, none)
    }
    assert len(keys) == 3
    with pytest.raises(RuntimeError, match="payload_integrity"):
        _parse_block_pages({"spark_cache_payload_integrity": "crc"})
    with pytest.raises(RuntimeError, match="tail-cow-v2"):
        parse({"spark_cache_payload_integrity": "fast"})


@pytest.mark.parametrize("mode", ["fast", "none"])
def test_v3_round_trip_matches_sha256_bytes_without_payload_digests(tmp_path, monkeypatch, mode):
    reference = _ModeHistory(tmp_path / "sha256", monkeypatch, mode="sha256")
    reference_hashed = Counter()
    with counting(reference_hashed):
        _publish_all(reference)
    history = _ModeHistory(tmp_path / mode, monkeypatch, mode=mode)
    hashed = Counter()
    with counting(hashed):
        _publish_all(history)
    # Publication hashes manifests, headers, identities and token prefixes
    # only; the content-addressed schema additionally hashes every payload
    # several times (object names, tails, delta digest, base and result).
    payload_bytes = sum(len(snapshot) for snapshot in history.snapshots)
    assert reference_hashed["sha256_bytes"] - hashed["sha256_bytes"] > 2 * payload_bytes
    # Every boundary, including the flat snapshot root at index 0.
    for index in range(0, TURNS):
        assert history.python_restore(index) == history.snapshots[index]
        hashed = Counter()
        with counting(hashed):
            result, placed, _transaction = history.cuda_restore(index)
        assert placed == history.expected_payload(index)
        # Only manifest-scale metadata is hashed on a restore.
        assert hashed["sha256_bytes"] < 256 * 1024
        assert result.final_hash_ms >= 0
    manifest = history.lookup(TURNS - 1)._manifest
    stage = manifest["delta_stages"][-1]
    assert "header_sha256" in stage
    descriptor = stage["delta_objects"][0]
    assert "crc32" in descriptor
    assert (descriptor["crc32"] is None) == (mode == "none")
    # Object names are not content addresses.
    path = tmp_path / mode / "chunks" / f"{descriptor['sha256']}.spcc"
    assert hashlib.sha256(path.read_bytes()).hexdigest() != descriptor["sha256"]
    # Digest fields hold boundary tokens.
    assert manifest["base_root"]["snapshot_sha256"] == cache_manifest._boundary_token(
        history.digests[0], history.boundary(0)
    )


def test_fast_detects_a_tampered_tail_and_a_tampered_object(tmp_path, monkeypatch):
    from sparkcache import spark_context_cache_cuda_hybrid_restore as cuda_hybrid

    history = _ModeHistory(tmp_path, monkeypatch, mode="fast")
    _publish_all(history)
    stage = history.lookup(TURNS - 1)._manifest["delta_stages"][-1]
    descriptor = stage["delta_objects"][-1]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    healthy = path.read_bytes()
    damaged = bytearray(healthy)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError, match=r"tail integrity rejection \(CRC32 fast\)"):
        history.cuda_restore(TURNS - 1)
    with pytest.raises(cache_manifest.CacheFormatError):
        history.python_restore(TURNS - 1)
    path.write_bytes(healthy)
    assert history.python_restore(TURNS - 1) == history.snapshots[TURNS - 1]
    # A snapshot object of the base is checked by its CRC32 as well.
    base_descriptor = history.lookup(TURNS - 1)._manifest["base_root"]["snapshot_objects"][0]
    base_path = tmp_path / "chunks" / f"{base_descriptor['sha256']}.spcc"
    base_healthy = base_path.read_bytes()
    base_damaged = bytearray(base_healthy)
    base_damaged[len(base_damaged) // 2] ^= 1
    base_path.write_bytes(base_damaged)
    cache_manifest._clear_hot_page_snapshots()
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError, match="page object CRC32 fast mismatch"):
        history.cuda_restore(TURNS - 1)
    base_path.write_bytes(base_healthy)


@pytest.mark.parametrize("mode", ["fast", "none"])
def test_v3_header_tampering_is_caught_by_the_manifest_bound_digest(tmp_path, monkeypatch, mode):
    from sparkcache import spark_context_cache_cuda_hybrid_restore as cuda_hybrid

    history = _ModeHistory(tmp_path, monkeypatch, mode=mode)
    _publish_all(history)
    stage = history.lookup(TURNS - 1)._manifest["delta_stages"][-1]
    descriptor = stage["delta_objects"][0]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    healthy = path.read_bytes()
    damaged = bytearray(healthy)
    # Flip a byte inside the JSON header (after the 10-byte prefix).
    damaged[16] ^= 1
    path.write_bytes(damaged)
    cache_manifest._clear_authenticated_delta_headers()
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError, match="header"):
        history.cuda_restore(TURNS - 1)
    with pytest.raises(cache_manifest.CacheFormatError, match="header"):
        history.python_restore(TURNS - 1)
    path.write_bytes(healthy)


def test_none_skips_payload_checks_but_keeps_geometry_checks(tmp_path, monkeypatch):
    from sparkcache import spark_context_cache_cuda_hybrid_restore as cuda_hybrid

    history = _ModeHistory(tmp_path, monkeypatch, mode="none")
    _publish_all(history)
    stage = history.lookup(TURNS - 1)._manifest["delta_stages"][-1]
    descriptor = stage["delta_objects"][-1]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    healthy = path.read_bytes()
    damaged = bytearray(healthy)
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    # A flipped payload byte is placed unnoticed: that is what "none" means.
    result, placed, _transaction = history.cuda_restore(TURNS - 1)
    assert placed != history.expected_payload(TURNS - 1)
    assert len(placed) == len(history.expected_payload(TURNS - 1))
    # A truncated object is still rejected by its length.
    path.write_bytes(healthy[:-1])
    with pytest.raises(cuda_hybrid.CudaHybridRestoreError):
        history.cuda_restore(TURNS - 1)
    path.write_bytes(healthy)


def test_readers_of_other_schemas_miss_cleanly(tmp_path, monkeypatch):
    fast = _ModeHistory(tmp_path, monkeypatch, mode="fast")
    _publish_all(fast)
    for other in ("page-tail-cow-v2", "page-tail-cow-v3-none"):
        lookup = fast.store.lookup(_identity(other), fast.digests[TURNS - 1], verify_chunks=False)
        assert not lookup.is_hit


def test_encoders_require_boundary_tokens_without_payload_digests():
    from tools.benchmark_flat_history_restore import Geometry, _capture, _snapshot

    geometry = Geometry(4096, False)
    layout = geometry.layout()
    base = _snapshot(geometry, 256, 1)
    capture = _capture(geometry, 256, 512, 2)
    with pytest.raises(hybrid.HybridCodecError, match="boundary tokens"):
        hybrid.encode_page_delta_from_capture(
            layout, base, capture,
            base_block_counts=geometry.counts(256), result_block_counts=geometry.counts(512),
            reused_pages_by_group=capture.reused_pages_by_group,
            base_boundary_tokens=256, result_boundary_tokens=512, integrity="fast",
        )
    token = hybrid.boundary_token("a" * 64, 512)
    delta = hybrid.encode_page_delta_from_capture(
        layout, base, capture,
        base_block_counts=geometry.counts(256), result_block_counts=geometry.counts(512),
        reused_pages_by_group=capture.reused_pages_by_group,
        base_boundary_tokens=256, result_boundary_tokens=512, integrity="fast",
        base_snapshot_sha256=hybrid.boundary_token("b" * 64, 256),
        result_snapshot_sha256=token,
    )
    plan = hybrid.plan_page_delta(
        layout, hybrid.page_delta_header_prefix(delta),
        base_block_counts=geometry.counts(256), result_block_counts=geometry.counts(512),
        base_boundary_tokens=256, result_boundary_tokens=512, total_bytes=len(delta),
    )
    assert plan.integrity == "fast"
    assert plan.result_snapshot_sha256 == token
    assert all(tail.sha256 is None and tail.crc32 is not None for tail in plan.tails)
