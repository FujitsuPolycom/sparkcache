"""Publication reuses proofs only for the exact verified base boundary."""

import hashlib
import os
from pathlib import Path

import pytest

from sparkcache import spark_context_cache_hybrid as hybrid
from sparkcache.persistent_context_cache import cache_manifest
from sparkcache.persistent_context_cache.test_page_history_reconstruction import _history
from sparkcache.spark_context_cache_codec import context_prefix_digest
from tools.benchmark_flat_history_restore import _Capture


def _extension(store, identity, layout, digest, **kwargs):
    return store.commit_page_extension(
        identity=identity, base_context_digest=digest, token_ids=tuple(range(1536)),
        identity_salt="history", layout=layout, base_block_counts=(5,),
        result_block_counts=(6,), base_boundary_tokens=1280, result_boundary_tokens=1536,
        result_snapshot=_Capture(b"x" * (6 * 512) + b"y" * (6 * 256), (0,)),
        **kwargs,
    )


def test_d25_publication_does_not_rehash_a_verified_composed_base(tmp_path, monkeypatch):
    store, identity, layout, digest, snapshots = _history(tmp_path)
    sha256 = hashlib.sha256
    base_hashes = []

    def counted(data=b"", **kwargs):
        if len(data) == len(snapshots[-1]) and data == snapshots[-1]:
            base_hashes.append(len(data))
        return sha256(data, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", counted)
    _extension(store, identity, layout, digest)
    assert len(base_hashes) == 1
    result_digest = context_prefix_digest(tuple(range(1536)), "history", token_count=1536)
    lookup = store.lookup(identity, result_digest, verify_chunks=False)
    stage = lookup._manifest["delta_stages"][-1]
    header = cache_manifest._authenticated_page_delta_header(
        tmp_path / "chunks", stage["delta_objects"][0])
    plan = hybrid.plan_page_delta(layout, header, base_block_counts=(5,),
        result_block_counts=(6,), base_boundary_tokens=1280, result_boundary_tokens=1536,
        total_bytes=stage["delta_encoded_bytes"])
    assert plan.base_snapshot_sha256 == sha256(snapshots[-1]).hexdigest()
    assert plan.base_snapshot_sha256 != lookup._manifest["base_root"]["snapshot_sha256"]
    assert store.restore_page_snapshot(lookup, layout=layout, result_block_counts=(6,),
        result_boundary_tokens=1536) == hybrid.encode_page_snapshot(
            layout, (6,), {"a": b"x" * (6 * 512), "b": b"y" * (6 * 256)})


def test_d25_supplied_base_bytes_must_match_authenticated_result_digest(tmp_path):
    store, identity, layout, digest, snapshots = _history(tmp_path)
    wrong = bytearray(snapshots[-1])
    wrong[-1] ^= 1
    with pytest.raises(cache_manifest.CacheFormatError, match="base snapshot digest"):
        _extension(store, identity, layout, digest, verified_base_snapshot=wrong)


def test_d25_corrupt_base_is_rejected_before_publication(tmp_path):
    store, identity, layout, digest, _ = _history(tmp_path)
    lookup = store.lookup(identity, digest, verify_chunks=False)
    descriptor = lookup._manifest["delta_stages"][-1]["delta_objects"][-1]
    path = tmp_path / "chunks" / f"{descriptor['sha256']}.spcc"
    damaged = bytearray(path.read_bytes())
    damaged[-1] ^= 1
    path.write_bytes(damaged)
    with pytest.raises(cache_manifest.CacheFormatError, match="checksum"):
        _extension(store, identity, layout, digest)
    result_digest = context_prefix_digest(tuple(range(1536)), "history", token_count=1536)
    assert not store.lookup(identity, result_digest).is_hit


@pytest.mark.parametrize("fail", [False, True])
def test_d25_publication_read_handles_are_bounded_and_closed(tmp_path, monkeypatch, fail):
    monkeypatch.setattr(cache_manifest, "_PUBLICATION_READ_FILES", 2)
    paths = [tmp_path / f"{index}.spcc" for index in range(3)]
    for index, path in enumerate(paths):
        path.write_bytes(bytes([index]) * 16)
    original_open, original_close = os.open, os.close
    active = set()
    opens = []

    def opened(path, *args, **kwargs):
        descriptor = original_open(path, *args, **kwargs)
        active.add(descriptor)
        opens.append(Path(path))
        assert len(active) <= 2
        return descriptor

    def closed(descriptor):
        active.remove(descriptor)
        return original_close(descriptor)

    monkeypatch.setattr(os, "open", opened)
    monkeypatch.setattr(os, "close", closed)
    try:
        with cache_manifest._PublicationReadFiles():
            for index in (0, 1, 0, 2, 1):
                payload = cache_manifest._read_object_extent(((paths[index], 16, 1, 3),))
                assert payload == bytes([index]) * 3
            if fail:
                raise ValueError("publication cancelled")
    except ValueError:
        assert fail
    assert opens == [paths[index] for index in (0, 1, 2, 1)]
    assert not active
    assert cache_manifest._publication_read_files.get() is None


def test_d25_supplied_mutable_base_is_detached_before_encoding(tmp_path, monkeypatch):
    store, identity, layout, digest, snapshots = _history(tmp_path)
    supplied = bytearray(snapshots[-1])
    encode = hybrid.encode_page_delta_from_capture

    def mutate(layout, proof, *args, **kwargs):
        supplied[-1] ^= 1
        assert proof.payload == snapshots[-1]
        assert isinstance(proof.payload, bytes)
        return encode(layout, proof, *args, **kwargs)

    monkeypatch.setattr(hybrid, "encode_page_delta_from_capture", mutate)
    _extension(store, identity, layout, digest, verified_base_snapshot=supplied)
