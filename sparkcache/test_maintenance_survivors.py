"""A maintenance inventory qualifies offers without a second filesystem walk."""

from collections import Counter
from pathlib import Path

import pytest

from sparkcache.persistent_context_cache.cache_manifest import (
    CapacityPolicy, EntryKey, LookupResult, ManifestStore,
)
from sparkcache.persistent_context_cache.test_cache_manifest import _chunk, _identity
from sparkcache.persistent_context_cache.test_prefix_aliases import _publish_source
from sparkcache.test_spark_context_cache_connector import _make_connector


POLICY = CapacityPolicy(max_bytes=10**9, low_watermark_bytes=10**9)


@pytest.fixture
def connector(tmp_path):
    connector = _make_connector(tmp_path, 0)
    identity = _identity()
    connector._identity = lambda rank: identity
    connector._capacity_policy = POLICY
    yield connector
    connector.shutdown()


def populate(connector, roots=12, chunks=8):
    identity = connector._identity(0)
    payloads = tuple(_chunk(index * 256, (index + 1) * 256) for index in range(chunks))
    digests = {f"{index + 1:064x}" for index in range(roots)}
    for digest in digests:
        connector._store.commit(identity=identity, context_digest=digest, chunks=payloads)
    connector._held.update(digests)
    return digests


def test_normal_maintenance_reads_each_root_once_and_stats_unique_chunks(connector, monkeypatch):
    digests = populate(connector)
    counts = Counter()
    read_bytes, stat = Path.read_bytes, Path.stat

    def read(path):
        if path.suffix == ".json":
            counts["roots"] += 1
        return read_bytes(path)

    def metadata(path, *args, **kwargs):
        if path.parent.name == "chunks" and path.suffix == ".spcc":
            counts["chunks"] += 1
        return stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_bytes", read)
    monkeypatch.setattr(Path, "stat", metadata)
    monkeypatch.setattr(connector, "_lookup_reusable",
                        lambda *args, **kwargs: pytest.fail("inventory was probed twice"))
    report = connector._maintain_capacity(force=True)
    assert report.surviving_entries is not None
    assert set(connector._held) == digests
    assert counts["roots"] == 12
    # Enumeration tests file type and obtains allocated/logical size once.
    assert counts["chunks"] <= 2 * 8


@pytest.mark.parametrize("damage", ["missing", "truncated", "same_size"])
def test_shared_payload_metadata_qualifies_every_offer(connector, damage):
    digests = populate(connector, roots=3, chunks=1)
    chunk = next((Path(connector._root) / "chunks").glob("*.spcc"))
    encoded = chunk.read_bytes()
    if damage == "missing":
        chunk.unlink()
    elif damage == "truncated":
        chunk.write_bytes(encoded[:-1])
    else:
        chunk.write_bytes(encoded[:-1] + bytes([encoded[-1] ^ 1]))
    connector._maintain_capacity(force=True)
    assert set(connector._held) == (digests if damage == "same_size" else set())
    # Metadata qualification never substitutes for restore integrity checks.
    for digest in digests:
        assert not connector._store.lookup(connector._identity(0), digest).is_hit


def test_corrupt_exact_root_shadows_valid_alias_until_removed(tmp_path, monkeypatch):
    store, identity, _, digest, _ = _publish_source(tmp_path, chunk_count=4,
                                                 prefix_tokens=(1024,))
    exact = store._manifest_path(identity, digest)
    exact.write_bytes(b"{}")
    unlink = Path.unlink

    def refuse_exact(path, *args, **kwargs):
        if path == exact:
            raise PermissionError("root retained")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", refuse_exact)
    report = store.maintain(POLICY)
    assert report.surviving_entries == ()
    assert not store.lookup(identity, digest, verify_chunks=False,
                            verify_chunk_metadata=True, storage_mode="per_token_rows").is_hit
    monkeypatch.setattr(Path, "unlink", unlink)
    report = store.maintain(POLICY)
    assert EntryKey(identity.storage_key, digest, "prefix_alias") in report.surviving_entries
    assert store.lookup(identity, digest, storage_mode="per_token_rows").is_hit


def test_alias_offers_are_excluded_from_block_page_connector(connector, tmp_path):
    store, identity, _, digest, _ = _publish_source(tmp_path, chunk_count=4,
                                                 prefix_tokens=(1024,))
    store._manifest_path(identity, digest).unlink()
    connector._identity = lambda rank: identity
    connector._held.add(digest)
    connector._storage_mode = "block_pages"
    connector._maintain_capacity(force=True)
    assert digest not in connector._held
    connector._storage_mode = "per_token_rows"
    connector._held.add(digest)
    connector._maintain_capacity(force=True)
    assert digest in connector._held


@pytest.mark.parametrize("mutation", ["add", "replace_inventory", "republish_same_digest"])
def test_inventory_mutations_are_not_withdrawn_from_a_stale_pass(connector, monkeypatch, mutation):
    stale = "a" * 64
    fresh = "b" * 64
    connector._held.add(stale)
    maintain = connector._store.maintain

    def race(*args, **kwargs):
        report = maintain(*args, **kwargs)
        with connector._store_cv:
            if mutation == "add":
                connector._held.add(fresh)
            elif mutation == "replace_inventory":
                connector._held = {fresh}
            else:
                connector._held.remove(stale)
                connector._held.add(stale)
        return report

    monkeypatch.setattr(connector._store, "maintain", race)
    connector._maintain_capacity(force=True)
    assert fresh in connector._held if mutation != "republish_same_digest" else stale in connector._held
    assert connector.counters["capacity_stale_inventory_snapshots"] == 1
    monkeypatch.setattr(connector._store, "maintain", maintain)
    connector._maintain_capacity(force=True)
    assert not connector._held


def test_fallback_probe_preserves_same_digest_republication(connector, monkeypatch):
    digest = "a" * 64
    connector._held.add(digest)

    def changed(*args, **kwargs):
        with connector._store_cv:
            connector._held.remove(digest)
            connector._held.add(digest)
        return LookupResult(False, "absent"), False

    monkeypatch.setattr(connector, "_lookup_reusable", changed)
    connector._reconcile_held_capacity()
    assert digest in connector._held
    assert connector.counters["capacity_stale_inventory_snapshots"] == 1


def test_other_store_publications_are_observed_on_each_inventory(connector):
    root = Path(connector._root)
    assert connector._maintain_capacity(force=True).surviving_entries == ()
    other = ManifestStore(root)
    digest = "a" * 64
    identity = connector._identity(0)
    other.commit(identity=identity, context_digest=digest, chunks=[_chunk()])
    connector._held.add(digest)
    report = connector._maintain_capacity(force=True)
    assert report.surviving_entries == (EntryKey(identity.storage_key, digest),)
    assert digest in connector._held
    other._manifest_path(identity, digest).write_bytes(b"{}")
    connector._maintain_capacity(force=True)
    assert digest not in connector._held


def test_empty_offer_inventory_does_not_require_initialized_model_identity(connector, monkeypatch):
    monkeypatch.setattr(connector, "_identity",
                        lambda rank: pytest.fail("model identity is not initialized"))
    assert connector._maintain_capacity(force=True).surviving_entries == ()
    connector._reconcile_held_capacity()
