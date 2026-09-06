"""GPU-free storage work and authentication regression tests."""

import dataclasses
import json
from collections import Counter
from pathlib import Path
from unittest import mock

import pytest

from sparkcache.persistent_context_cache import cache_manifest as m
from sparkcache.persistent_context_cache.test_cache_manifest import _chunk, _identity
from sparkcache.spark_context_cache_codec import context_prefix_digest
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
)


def _publish(path, payload, batch):
    if batch:
        m._publish_immutable_batch([(path, payload)])
    else:
        m._publish_immutable(path, payload)


@pytest.mark.parametrize("batch", [False, True])
def test_matching_immutable_payload_is_not_staged(tmp_path, batch):
    path = tmp_path / "entry.json"
    payload = b"authenticated-payload"
    path.write_bytes(payload)
    original_open = Path.open
    staged = []

    def record_open(path, mode="r", *args, **kwargs):
        if "x" in mode:
            staged.append(path)
        return original_open(path, mode, *args, **kwargs)

    with mock.patch.object(Path, "open", record_open):
        _publish(path, payload, batch)
    assert not staged
    assert path.read_bytes() == payload


@pytest.mark.parametrize("batch", [False, True])
def test_deduplication_flushes_adopted_data_before_directory(tmp_path, batch):
    path = tmp_path / "entry.json"
    path.write_bytes(b"value")
    events = []
    with (
        mock.patch.object(m.os, "fsync", side_effect=lambda fd: events.append("data")),
        mock.patch.object(
            m, "_fsync_directory", side_effect=lambda path: events.append("directory")
        ),
    ):
        _publish(path, b"value", batch)
    assert events == ["data", "directory"]


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("barrier", ["data", "directory"])
def test_deduplication_durability_failure_cannot_succeed(tmp_path, batch, barrier):
    path = tmp_path / "entry.json"
    path.write_bytes(b"value")
    owner, name = (m.os, "fsync") if barrier == "data" else (m, "_fsync_directory")
    with (
        mock.patch.object(owner, name, side_effect=OSError("barrier failed")),
        pytest.raises((m.CommitConflict, OSError)),
    ):
        _publish(path, b"value", batch)
    assert path.read_bytes() == b"value"
    assert list(tmp_path.glob(".*.writing-*")) == []


@pytest.mark.parametrize("batch", [False, True])
@pytest.mark.parametrize("same_payload", [False, True])
def test_immutable_publication_rechecks_competing_link(tmp_path, batch, same_payload):
    path = tmp_path / "entry.json"
    payload = b"value"
    competing = payload if same_payload else b"other"
    original_link = m.os.link

    def competing_link(source, destination):
        destination.write_bytes(competing)
        original_link(source, destination)

    with mock.patch.object(m.os, "link", competing_link):
        if same_payload:
            _publish(path, payload, batch)
        else:
            with pytest.raises(m.CommitConflict):
                _publish(path, payload, batch)
    assert path.read_bytes() == competing
    assert list(tmp_path.glob(".*.writing-*")) == []


@pytest.mark.parametrize("competing_link", [False, True])
def test_content_addressed_batch_repairs_corruption(tmp_path, competing_link):
    payload = b"value"
    path = tmp_path / f"{m._sha256(payload)}.spcc"
    if not competing_link:
        path.write_bytes(b"other")
    original_link = m.os.link

    def corrupt_link(source, destination):
        destination.write_bytes(b"other")
        original_link(source, destination)

    with mock.patch.object(
        m.os, "link", corrupt_link if competing_link else original_link
    ):
        m._publish_immutable_batch([(path, payload)])
    assert path.read_bytes() == payload
    assert list(tmp_path.glob(".*.writing-*")) == []


def test_batch_stages_identical_content_addressed_paths_once(tmp_path):
    payload = b"value"
    path = tmp_path / f"{m._sha256(payload)}.spcc"
    original_open = Path.open
    staged = []

    def record_open(path, mode="r", *args, **kwargs):
        if "x" in mode:
            staged.append(path)
        return original_open(path, mode, *args, **kwargs)

    with mock.patch.object(Path, "open", record_open):
        m._publish_immutable_batch([(path, payload), (path, payload)])
    assert len(staged) == 1
    assert path.read_bytes() == payload


def _page_fixture(tmp_path, schema="page-tail-cow-v2"):
    store = m.ManifestStore(tmp_path)
    identity = dataclasses.replace(
        _identity(),
        record_schema=("target_ckv", "logical_positions"),
        publication_schema=schema,
    )
    layout = PageLayout((PageGroup(256, (PageLayer("page", "u8", (64,), 64),)),))
    tokens = tuple(range(768))
    salt = "page-publication-authentication"

    def digest(n):
        return context_prefix_digest(tokens, salt, token_count=n * 256)

    def snapshot(n):
        return encode_page_snapshot(layout, (n,), {"page": b"A" * 64 * n})

    def extend(n):
        return store.commit_page_extension(
            identity=identity,
            base_context_digest=digest(n - 1),
            token_ids=tokens,
            identity_salt=salt,
            layout=layout,
            base_block_counts=(n - 1,),
            result_block_counts=(n,),
            base_boundary_tokens=(n - 1) * 256,
            result_boundary_tokens=n * 256,
            result_snapshot=snapshot(n),
        )

    store.commit_page_snapshot(
        identity=identity, context_digest=digest(1), span_tokens=256, snapshot=snapshot(1)
    )
    return store, identity, layout, digest, snapshot, extend


@pytest.mark.parametrize("schema", ["page-tail-cow-v1", "page-tail-cow-v2"])
@pytest.mark.parametrize("base_chunks", [1, 2])
def test_page_extension_reads_each_authenticated_base_object_once(
    tmp_path, schema, base_chunks
):
    store, identity, layout, digest, snapshot, extend = _page_fixture(tmp_path, schema)
    if base_chunks == 2:
        extend(2)
    base_paths = set((tmp_path / "chunks").glob("*.spcc"))
    reads = Counter()
    original_read = Path.read_bytes

    def record_read(path):
        if path in base_paths:
            reads[path] += 1
        return original_read(path)

    with mock.patch.object(Path, "read_bytes", record_read):
        extend(base_chunks + 1)

    assert set(reads) == base_paths
    assert all(count == 1 for count in reads.values()), reads
    restored = store.restore_page_snapshot(
        store.lookup(identity, digest(base_chunks + 1)),
        layout=layout,
        result_block_counts=(base_chunks + 1,),
        result_boundary_tokens=(base_chunks + 1) * 256,
    )
    assert restored == snapshot(base_chunks + 1)


@pytest.mark.parametrize("schema", ["page-tail-cow-v1", "page-tail-cow-v2"])
@pytest.mark.parametrize("corrupt_object", ["base", "delta"])
def test_page_extension_rejects_same_size_corruption_before_publication(
    tmp_path, schema, corrupt_object
):
    store, identity, _layout, digest, _snapshot, extend = _page_fixture(tmp_path, schema)
    base_paths = set((tmp_path / "chunks").glob("*.spcc"))
    extend(2)
    paths = (
        base_paths
        if corrupt_object == "base"
        else set((tmp_path / "chunks").glob("*.spcc")) - base_paths
    )
    path = next(iter(paths))
    payload = bytearray(path.read_bytes())
    payload[-1] ^= 1
    path.write_bytes(payload)
    with pytest.raises(m.CacheFormatError):
        extend(3)
    assert not store.lookup(identity, digest(3)).is_hit


def _alias_fixture(tmp_path):
    store = m.ManifestStore(tmp_path)
    identity = _identity()
    tokens = tuple(range(65536))
    salt = "alias-maintenance-sharing"
    digest = context_prefix_digest(tokens, salt, token_count=len(tokens))
    store.commit(
        identity=identity,
        context_digest=digest,
        chunks=tuple(_chunk(n * 256, (n + 1) * 256) for n in range(256)),
        span_tokens=len(tokens),
    )
    receipt = store.publish_prefix_aliases(
        identity=identity,
        source_context_digest=digest,
        token_ids=tokens,
        identity_salt=salt,
        storage_mode="per_token_rows",
    )
    return store, identity, receipt


def test_maintenance_authenticates_each_shared_alias_segment_once_per_pass(tmp_path):
    store, _identity_value, receipt = _alias_fixture(tmp_path)
    assert receipt.aliases_published > 1
    original_read = Path.read_bytes
    reads = Counter()

    def record_read(path):
        if path.suffix == ".spix":
            reads[path] += 1
        return original_read(path)

    with mock.patch.object(Path, "read_bytes", record_read):
        report = store.maintain(
            m.CapacityPolicy(max_bytes=10**9, low_watermark_bytes=10**9)
        )
    assert report.capacity_satisfied
    assert len(reads) == receipt.segments_published
    assert all(count == 1 for count in reads.values()), reads


def test_maintenance_does_not_reuse_segment_authentication_across_passes(tmp_path):
    store, identity, receipt = _alias_fixture(tmp_path)
    policy = m.CapacityPolicy(max_bytes=10**9, low_watermark_bytes=10**9)
    store.maintain(policy)
    paths = tuple((tmp_path / "prefix-index" / identity.storage_key).glob("*.spix"))
    for path in paths:
        path.write_bytes(path.read_bytes() + b"corrupt")
    report = store.maintain(policy)
    assert report.aliases_evicted == receipt.aliases_published
    # The exact manifest remains a valid retention root for shared chunks.
    assert report.orphan_chunks_deleted == 0


def test_segment_memoization_preserves_alias_geometry_validation(tmp_path):
    store, identity, receipt = _alias_fixture(tmp_path)
    paths = sorted((tmp_path / "prefix-aliases" / identity.storage_key).glob("*.json"))
    cache = {}
    valid = store._capacity_alias_entry(paths[0], cache)
    assert valid.valid
    alias = json.loads(paths[0].read_bytes())
    alias["committed_tokens"] += 256
    alias.pop("metadata_sha256")
    alias["metadata_sha256"] = m._sha256(m._canonical_json(alias))
    paths[0].write_bytes(m._canonical_json(alias))
    assert not store._capacity_alias_entry(paths[0], cache).valid
    assert receipt.aliases_published > 1
