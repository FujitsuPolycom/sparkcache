"""Deletion pacing preserves cache references and publication barriers."""

from unittest import mock

import pytest

from sparkcache.persistent_context_cache import cache_manifest as manifest
from sparkcache.persistent_context_cache.cache_manifest import (
    CapacityPolicy,
    EntryKey,
    ManifestStore,
)
from sparkcache.persistent_context_cache.test_cache_manifest import (
    _identity,
    _variant_chunk,
)


def populate(root, count=3):
    store = ManifestStore(root)
    identity = _identity()
    digests = [f"{index + 1:064x}" for index in range(count)]
    for digest in digests:
        store.commit(
            identity=identity,
            context_digest=digest,
            chunks=[_variant_chunk(digest.encode())],
        )
    return store, identity, digests


def test_budget_drains_orphans_before_evicting_more_roots(tmp_path):
    store, _, _ = populate(tmp_path)
    policy = CapacityPolicy(
        max_bytes=1, low_watermark_bytes=1, maintenance_max_deletions=1
    )
    first = store.maintain(policy)
    assert first.manifests_evicted == first.deletion_attempts == 1
    assert first.work_pending
    second = store.maintain(policy)
    assert second.manifests_evicted == 0
    assert second.chunks_deleted == second.deletion_attempts == 1
    for _ in range(8):
        result = store.maintain(policy)
        assert result.deletion_attempts <= 1
        if result.capacity_satisfied and not result.work_pending:
            break
    else:
        pytest.fail("bounded passes did not converge")
    assert result.bytes_after == 0


def test_budget_preserves_protected_shared_payload(tmp_path):
    store, identity, digests = populate(tmp_path, 1)
    sibling = "f" * 64
    store.commit(
        identity=identity,
        context_digest=sibling,
        chunks=[_variant_chunk(digests[0].encode())],
    )
    policy = CapacityPolicy(
        max_bytes=1, low_watermark_bytes=1, maintenance_max_deletions=1
    )
    report = store.maintain(
        policy, protected_entries=[EntryKey(identity.storage_key, sibling)]
    )
    assert report.manifests_evicted == 1
    assert report.chunks_deleted == 0
    assert not report.capacity_satisfied
    lookup = store.lookup(identity, sibling)
    assert lookup.is_hit
    assert store.restore(lookup) == (_variant_chunk(digests[0].encode()),)
    assert report.bytes_after == sum(
        manifest._allocated_bytes(p.stat())
        for folder in ("manifests", "chunks")
        for p in (tmp_path / folder).rglob("*")
        if p.is_file()
    )


def test_budget_counts_failed_unlink_attempts(tmp_path):
    store, _, _ = populate(tmp_path)
    policy = CapacityPolicy(
        max_bytes=1, low_watermark_bytes=1, maintenance_max_deletions=1
    )
    with mock.patch("pathlib.Path.unlink", side_effect=OSError("busy")) as unlink:
        report = store.maintain(policy)
    assert unlink.call_count == report.deletion_attempts == 1
    assert report.manifests_evicted == report.chunks_deleted == 0
    assert report.bytes_before == report.bytes_after
    assert report.work_pending


def test_cooldown_skips_inventory_and_does_not_extend_itself(tmp_path):
    store, _, _ = populate(tmp_path)
    policy = CapacityPolicy(
        max_bytes=1,
        low_watermark_bytes=1,
        maintenance_max_deletions=1,
        maintenance_interval_ms=1000,
    )
    with mock.patch.object(manifest.time, "monotonic", return_value=10.0):
        store.maintain(policy)
    with (
        mock.patch.object(manifest.time, "monotonic", return_value=10.9),
        mock.patch.object(store, "_capacity_entry", side_effect=AssertionError("scan")),
    ):
        report = store.maintain(policy)
    assert report.skipped_cooldown and not report.skipped_busy
    assert not report.capacity_satisfied
    assert report.deletion_attempts == 0
    with mock.patch.object(manifest.time, "monotonic", return_value=11.0):
        resumed = store.maintain(policy)
    assert not resumed.skipped_cooldown
    assert resumed.chunks_deleted == 1


def test_root_barrier_failure_preserves_payload_and_starts_cooldown(tmp_path):
    store, _, _ = populate(tmp_path, 1)
    policy = CapacityPolicy(
        max_bytes=1,
        low_watermark_bytes=1,
        maintenance_max_deletions=2,
        maintenance_interval_ms=1000,
    )
    with (
        mock.patch.object(manifest.time, "monotonic", return_value=10.0),
        mock.patch.object(manifest, "_fsync_directory", side_effect=OSError("barrier")),
    ):
        with pytest.raises(OSError, match="barrier"):
            store.maintain(policy)
    assert len(list((tmp_path / "chunks").iterdir())) == 1
    with mock.patch.object(manifest.time, "monotonic", return_value=10.1):
        assert store.maintain(policy).skipped_cooldown


@pytest.mark.parametrize(
    "field", ["maintenance_max_deletions", "maintenance_interval_ms"]
)
@pytest.mark.parametrize("value", [-1, True, 1.5, "1"])
def test_maintenance_controls_reject_invalid_values(field, value):
    with pytest.raises(ValueError, match=field):
        CapacityPolicy(**{field: value})


def test_budgeted_alias_cleanup_keeps_protected_descriptor_chain(tmp_path):
    from sparkcache.persistent_context_cache.test_prefix_aliases import _publish_source
    from sparkcache.spark_context_cache_codec import context_prefix_digest

    store, identity, tokens, _, chunks = _publish_source(
        tmp_path, chunk_count=20, prefix_tokens=(1024, 4096)
    )
    digest = context_prefix_digest(tokens, identity.storage_key, token_count=4096)
    protected = [EntryKey(identity.storage_key, digest, "prefix_alias")]
    policy = CapacityPolicy(
        max_bytes=1, low_watermark_bytes=1, maintenance_max_deletions=2
    )
    for _ in range(20):
        report = store.maintain(policy, protected_entries=protected)
        assert report.deletion_attempts <= 2
        lookup = store.lookup(identity, digest, storage_mode="per_token_rows")
        assert lookup.is_hit, lookup.reason
        assert store.restore(lookup) == chunks[:16]
        if report.deletion_attempts == 0:
            break
    else:
        pytest.fail("cleanup did not settle around the protected alias")
    for _ in range(30):
        report = store.maintain(policy)
        assert report.deletion_attempts <= 2
        if report.capacity_satisfied and not report.work_pending:
            break
    else:
        pytest.fail("descriptor and payload cleanup did not converge")
    assert report.bytes_after == 0
