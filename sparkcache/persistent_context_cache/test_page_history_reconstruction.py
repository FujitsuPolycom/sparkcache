"""Flat history reconstruction retains every intermediate checksum proof."""

import hashlib
import json
import tracemalloc
from collections import Counter
from dataclasses import replace

import pytest

from sparkcache import spark_context_cache_hybrid as hybrid
from sparkcache.persistent_context_cache.cache_manifest import ManifestStore
from sparkcache.persistent_context_cache.test_page_delta_macro_objects import _identity
from sparkcache.spark_context_cache_codec import context_prefix_digest


def _history(tmp_path, stages=3):
    layout = hybrid.PageLayout((hybrid.PageGroup(256, (
        hybrid.PageLayer("a", "u8", (512,), 512),
        hybrid.PageLayer("b", "u8", (256,), 256),
    )),))
    identity = replace(_identity(), publication_schema="page-tail-cow-v2")
    store = ManifestStore(tmp_path)
    tokens = tuple(range((stages + 2) * 256))
    digests = [context_prefix_digest(tokens, "history", token_count=count * 256)
               for count in range(2, stages + 3)]
    snapshots = [hybrid.encode_page_snapshot(layout, (count,), {
        "a": bytes((count,)) * (count * 512), "b": bytes((count + 32,)) * (count * 256),
    }) for count in range(2, stages + 3)]
    store.commit_page_snapshot(identity=identity, context_digest=digests[0],
                               span_tokens=512, snapshot=snapshots[0])
    for index in range(stages):
        store.commit_page_extension(identity=identity, base_context_digest=digests[index],
            token_ids=tokens, identity_salt="history", layout=layout,
            base_block_counts=(index + 2,), result_block_counts=(index + 3,),
            base_boundary_tokens=(index + 2) * 256,
            result_boundary_tokens=(index + 3) * 256, result_snapshot=snapshots[index + 1])
    return store, identity, layout, digests[-1], snapshots


def test_flat_history_hashes_each_intermediate_snapshot_once(tmp_path, monkeypatch):
    store, identity, layout, digest, snapshots = _history(tmp_path)
    counts = Counter()
    original = hashlib.sha256

    class SnapshotHash:
        def __init__(self, data):
            self.inner = original(data)
            self.size = len(data)

        def update(self, data):
            self.inner.update(data)
            self.size += len(data)

        def hexdigest(self):
            if self.size > 512:
                counts[self.size] += 1
            return self.inner.hexdigest()

    def counted(data=b"", *args, **kwargs):
        if bytes(data[:6]) == hybrid._MAGIC:
            return SnapshotHash(data)
        return original(data, *args, **kwargs)

    monkeypatch.setattr(hashlib, "sha256", counted)
    restored = store.restore_page_snapshot(store.lookup(identity, digest, verify_chunks=False),
        layout=layout, result_block_counts=(5,), result_boundary_tokens=1280)
    assert restored == snapshots[-1]
    assert [counts[len(snapshot)] for snapshot in snapshots[1:]] == [1, 1, 1]


def test_flat_history_does_not_materialize_complete_intermediate_snapshots(tmp_path, monkeypatch):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=8)
    monkeypatch.setattr(hybrid, "_apply_verified_page_delta",
                        lambda *args, **kwargs: pytest.fail("complete intermediate snapshot joined"))
    restored = store.restore_page_snapshot(store.lookup(identity, digest, verify_chunks=False),
        layout=layout, result_block_counts=(10,), result_boundary_tokens=2560)
    assert type(restored) is bytes
    assert restored == snapshots[-1]


def test_private_layer_history_matches_scalar_with_mixed_groups_and_zero_tails():
    layout = hybrid.PageLayout((
        hybrid.PageGroup(256, (
            hybrid.PageLayer("a", "u8", (32,), 32),
            hybrid.PageLayer("b", "u8", (17,), 17),
        )),
        hybrid.PageGroup(1, (hybrid.PageLayer("recurrent", "u8", (13,), 13),)),
    ))
    counts = [(2, 1), (3, 1), (4, 1), (4, 1)]
    payloads = [
        {"a": b"a" * 64, "b": b"b" * 34, "recurrent": b"r" * 13},
        {"a": b"a" * 96, "b": b"b" * 51, "recurrent": b"s" * 13},
        {"a": b"z" * 128, "b": b"b" * 68, "recurrent": b"s" * 13},
        {"a": b"z" * 128, "b": b"b" * 68, "recurrent": b"t" * 13},
    ]
    snapshots = [hybrid.encode_page_snapshot(layout, count, payload)
                 for count, payload in zip(counts, payloads, strict=True)]
    mutable_base = bytearray(snapshots[0])
    reconstruction = hybrid._PageHistoryReconstruction(layout, mutable_base, counts[0], 256)
    mutable_base[-1] ^= 1
    for index in range(1, len(snapshots)):
        arguments = dict(base_block_counts=counts[index - 1], result_block_counts=counts[index],
                         base_boundary_tokens=index * 256, result_boundary_tokens=(index + 1) * 256)
        delta = hybrid.encode_page_delta(layout, snapshots[index - 1], snapshots[index], **arguments)
        scalar = hybrid.apply_page_delta(layout, snapshots[index - 1], delta, **arguments)
        mutable_delta = bytearray(delta)
        reconstruction.apply(mutable_delta, **arguments)
        mutable_delta[-1] ^= 1
        observed = reconstruction.finish()
        assert type(observed) is bytes
        assert observed == scalar == snapshots[index]
    assert reconstruction.finish() == snapshots[-1]


def test_failed_private_history_cannot_return_or_reuse_unverified_buffers(tmp_path):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=1)
    stage = store.lookup(identity, digest, verify_chunks=False)._manifest["delta_stages"][0]
    delta = store._read_page_delta_objects(stage["delta_objects"],
        encoded_bytes=stage["delta_encoded_bytes"], encoded_sha256=stage["delta_sha256"])
    delta = bytes(delta).replace(hashlib.sha256(snapshots[1]).hexdigest().encode(), b"f" * 64)
    reconstruction = hybrid._PageHistoryReconstruction(layout, snapshots[0], (2,), 512)
    arguments = dict(base_block_counts=(2,), result_block_counts=(3,),
                     base_boundary_tokens=512, result_boundary_tokens=768)
    with pytest.raises(hybrid.HybridCodecError, match="result checksum mismatch"):
        reconstruction.apply(delta, **arguments)
    with pytest.raises(hybrid.HybridCodecError, match="unverified stage"):
        reconstruction.finish()
    with pytest.raises(hybrid.HybridCodecError, match="unverified stage"):
        reconstruction.apply(delta, **arguments)


def test_delta_application_assembles_views_without_decoding_full_base(tmp_path, monkeypatch):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=1)
    lookup = store.lookup(identity, digest, verify_chunks=False)
    stage = lookup._manifest["delta_stages"][0]
    delta = store._read_page_delta_objects(stage["delta_objects"],
        encoded_bytes=stage["delta_encoded_bytes"], encoded_sha256=stage["delta_sha256"])
    monkeypatch.setattr(hybrid, "decode_page_snapshot",
                        lambda *args: pytest.fail("restore copied the full base into layer buffers"))
    restored = hybrid.apply_page_delta(layout, snapshots[0], delta,
        base_block_counts=(2,), result_block_counts=(3,),
        base_boundary_tokens=512, result_boundary_tokens=768)
    assert restored == snapshots[1]


def test_snapshot_encoding_does_not_copy_a_joined_payload_again_for_its_header():
    layout = hybrid.PageLayout((hybrid.PageGroup(256, (
        hybrid.PageLayer("a", "u8", (131072,), 131072),
        hybrid.PageLayer("b", "u8", (131072,), 131072),
    )),))
    payloads = {"a": b"a" * 131072, "b": b"b" * 131072}
    tracemalloc.start()
    try:
        encoded = hybrid.encode_page_snapshot(layout, (1,), payloads)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert hybrid.decode_page_snapshot(layout, encoded, (1,)) == payloads
    assert peak < len(encoded) * 1.5


def test_captured_delta_encoding_does_not_copy_a_joined_payload_again_for_its_header():
    layout = hybrid.PageLayout((hybrid.PageGroup(256, (
        hybrid.PageLayer("a", "u8", (131072,), 131072),
        hybrid.PageLayer("b", "u8", (131072,), 131072),
    )),))
    base = hybrid.encode_page_snapshot(layout, (1,), {"a": b"a" * 131072, "b": b"b" * 131072})
    captured = b"c" * 131072 + b"d" * 131072
    tracemalloc.start()
    try:
        encoded = hybrid.encode_page_delta_from_capture(layout, base, captured,
            base_block_counts=(1,), result_block_counts=(2,), reused_pages_by_group=(1,),
            base_boundary_tokens=256, result_boundary_tokens=512)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    restored = hybrid.apply_page_delta(layout, base, encoded,
        base_block_counts=(1,), result_block_counts=(2,),
        base_boundary_tokens=256, result_boundary_tokens=512)
    assert hybrid.decode_page_snapshot(layout, restored, (2,)) == {
        "a": b"a" * 131072 + b"c" * 131072,
        "b": b"b" * 131072 + b"d" * 131072,
    }
    assert peak < len(encoded) * 2.5


def test_verified_snapshot_proof_does_not_alias_mutable_input(tmp_path):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=1)
    mutable = bytearray(snapshots[0])
    proof = hybrid._verify_page_snapshot_bytes(mutable)
    mutable[-1] ^= 1
    assert isinstance(proof.payload, bytes)
    assert proof.payload == snapshots[0]
    stage = store.lookup(identity, digest, verify_chunks=False)._manifest["delta_stages"][0]
    delta = store._read_page_delta_objects(stage["delta_objects"],
        encoded_bytes=stage["delta_encoded_bytes"], encoded_sha256=stage["delta_sha256"])
    result = hybrid._apply_verified_page_delta(layout, proof, delta,
        base_block_counts=(2,), result_block_counts=(3,),
        base_boundary_tokens=512, result_boundary_tokens=768)
    assert result.payload == snapshots[1]
    assert result.sha256 == hashlib.sha256(snapshots[1]).hexdigest()


@pytest.mark.parametrize("alter_result", [False, True])
def test_flat_history_rejects_wrong_intermediate_proof_even_when_final_stage_overwrites_it(tmp_path, alter_result):
    store, identity, layout, digest, snapshots = _history(tmp_path, stages=2)
    path = tmp_path / "manifests" / identity.storage_key / f"{digest}.json"
    manifest = json.loads(path.read_bytes())
    fields = [(1, "base_snapshot_sha256")]
    if alter_result:
        fields.append((0, "result_snapshot_sha256"))
    for index, field in fields:
        stage = manifest["delta_stages"][index]
        assert len(stage["delta_objects"]) == 1
        descriptor = stage["delta_objects"][0]
        encoded = (tmp_path / "chunks" / f"{descriptor['sha256']}.spcc").read_bytes()
        needle = f'"{field}":"{hashlib.sha256(snapshots[1]).hexdigest()}"'.encode()
        assert encoded.count(needle) == 1
        encoded = encoded.replace(needle, f'"{field}":"{"f" * 64}"'.encode())
        checksum = hashlib.sha256(encoded).hexdigest()
        (tmp_path / "chunks" / f"{checksum}.spcc").write_bytes(encoded)
        stage["delta_sha256"] = descriptor["sha256"] = checksum
    manifest.pop("metadata_sha256")
    manifest["metadata_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")))
    lookup = store.lookup(identity, digest, verify_chunks=False)
    assert lookup.is_hit
    with pytest.raises(hybrid.HybridCodecError, match="result checksum mismatch" if alter_result else "base differs"):
        store.restore_page_snapshot(lookup, layout=layout,
                                    result_block_counts=(4,), result_boundary_tokens=1024)
