"""Bounded flat page histories preserve every published boundary."""

from dataclasses import replace

import pytest

from sparkcache import spark_context_cache_hybrid as hybrid
from sparkcache.persistent_context_cache.cache_manifest import ManifestStore
from sparkcache.persistent_context_cache.test_page_delta_macro_objects import _identity
from sparkcache.test_periodic_page_snapshot import parse
from tools.benchmark_flat_history_restore import (
    Geometry, _snapshot, _capture, _cuda_restore, _identity as benchmark_identity,
)
from sparkcache.spark_context_cache_codec import context_prefix_digest


KEY = "spark_cache_page_delta_stage_limit_count"
ENV = "SPARK_CONTEXT_CACHE_PAGE_DELTA_STAGE_LIMIT_COUNT"


def test_stage_limit_default_override_and_off(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    assert parse().page_delta_stage_limit_count == 4
    monkeypatch.setenv(ENV, "8")
    assert parse().page_delta_stage_limit_count == 8
    assert parse({KEY: 0}).page_delta_stage_limit_count == 0
    assert parse({KEY: 2}).build_identity(0, 0) == parse({KEY: 0}).build_identity(0, 0)


@pytest.mark.parametrize("value", [-1, True, False, 1.5, "bad", None])
def test_stage_limit_rejects_invalid_settings(value):
    with pytest.raises(RuntimeError, match=KEY):
        parse({KEY: value})


@pytest.mark.parametrize("limit", [0, 1, 4])
@pytest.mark.parametrize("sparse", [False, True])
@pytest.mark.parametrize("recurrent_first", [False, True])
def test_d24_stage_limit_preserves_every_boundary(tmp_path, monkeypatch, limit, sparse, recurrent_first):
    from sparkcache.persistent_context_cache import cache_manifest

    geometry = Geometry(4096, recurrent_first)
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", geometry.object_bytes)
    monkeypatch.setattr(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", geometry.object_bytes)
    layout = geometry.layout()
    identity = benchmark_identity()
    store = ManifestStore(tmp_path, page_delta_stage_limit_count=limit)
    tokens = tuple(range(12 * 256))
    salt = "stage-limit"
    digests = [context_prefix_digest(tokens, salt, token_count=n * 256)
               for n in range(1, 13)]
    snapshots = [_snapshot(geometry, n * 256, n) for n in range(1, 13)]
    store.commit_page_snapshot(identity=identity, context_digest=digests[0],
                               span_tokens=256, snapshot=snapshots[0])
    for index in range(1, 12):
        receipt = store.commit_page_extension(
            identity=identity, base_context_digest=digests[index - 1],
            token_ids=tokens, identity_salt=salt, layout=layout,
            base_block_counts=geometry.counts(index * 256),
            result_block_counts=geometry.counts((index + 1) * 256),
            base_boundary_tokens=index * 256, result_boundary_tokens=(index + 1) * 256,
            result_snapshot=(_capture(geometry, index * 256, (index + 1) * 256, index + 1)
                             if sparse else snapshots[index]),
        )
        stages = index % (limit + 1) if limit else index
        lookup = store.lookup(identity, digests[index], verify_chunks=False)
        assert len(lookup._manifest.get("delta_stages", ())) == stages
        assert lookup.root_kind == ("page_delta" if stages else "page_snapshot")
        assert receipt.publication.kind == ("page_delta" if stages else "page_snapshot")
    assert store.publication_telemetry_snapshot().committed_publications == 12
    # A fresh reader must reconstruct even boundaries preceding compaction.
    reader = ManifestStore(tmp_path)
    for index, expected in enumerate(snapshots):
        boundary = (index + 1) * 256
        lookup = reader.lookup(identity, digests[index], verify_chunks=False)
        assert reader.restore_page_snapshot(
            lookup, layout=layout, result_block_counts=geometry.counts(boundary),
            result_boundary_tokens=boundary,
        ) == expected
        # Exercise the CUDA reader with its CPU placement adapter as well.
        _cuda_restore(geometry, reader, tmp_path, digests[index], boundary)


def test_stage_limit_does_not_change_v1_publication(tmp_path):
    store = ManifestStore(tmp_path, page_delta_stage_limit_count=1)
    identity = replace(_identity(), publication_schema="page-tail-cow-v1")
    layout = hybrid.PageLayout((hybrid.PageGroup(256, (
        hybrid.PageLayer("a", "u8", (1,), 1),)),))
    tokens = tuple(range(1024))
    digest = context_prefix_digest(tokens, "v1", token_count=256)
    store.commit_page_snapshot(identity=identity, context_digest=digest, span_tokens=256,
                               snapshot=hybrid.encode_page_snapshot(layout, (1,), {"a": b"a"}))
    for count in range(2, 5):
        store.commit_page_extension(identity=identity, base_context_digest=digest,
            token_ids=tokens, identity_salt="v1", layout=layout,
            base_block_counts=(count - 1,), result_block_counts=(count,),
            base_boundary_tokens=(count - 1) * 256, result_boundary_tokens=count * 256,
            result_snapshot=hybrid.encode_page_snapshot(layout, (count,), {"a": b"a" * count}))
        digest = context_prefix_digest(tokens, "v1", token_count=count * 256)
        assert store.lookup(identity, digest).root_kind == "page_delta"
