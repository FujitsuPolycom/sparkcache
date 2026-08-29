from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    CommitConflict,
    ContextChunk,
    ManifestStore,
    StateRecord,
)
from sparkcache.spark_context_cache_codec import context_prefix_digest


def _identity() -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="rows-v1",
        rope_layout="rope-v1",
        tp_degree=1,
        dcp_degree=1,
        chunk_tokens=256,
    )


def _chunk(index: int, *, label: bytes = b"source") -> ContextChunk:
    start = index * 256
    records = {
        StateRecord.TARGET_CKV: label + b"-target-" + str(index).encode(),
        StateRecord.SPARSE_INDEXER: label + b"-index-" + str(index).encode(),
        StateRecord.MTP_DRAFT_KV: label + b"-draft-" + str(index).encode(),
        StateRecord.BOUNDARY_HIDDEN: label + b"-hidden-" + str(index).encode(),
        StateRecord.LOGICAL_POSITIONS: label + b"-positions-" + str(index).encode(),
    }
    return ContextChunk(start, start + 256, records)


def _publish_source(
    root: Path,
    *,
    chunk_count: int,
    prefix_tokens: tuple[int, ...] | None = None,
) -> tuple[ManifestStore, CacheIdentity, tuple[int, ...], str, tuple[ContextChunk, ...]]:
    store = ManifestStore(root)
    identity = _identity()
    tokens = tuple(range(chunk_count * 256))
    salt = identity.storage_key
    source_digest = context_prefix_digest(tokens, salt, token_count=len(tokens))
    chunks = tuple(_chunk(index) for index in range(chunk_count))
    store.commit(
        identity=identity,
        context_digest=source_digest,
        chunks=chunks,
        span_tokens=len(tokens),
    )
    store.publish_prefix_aliases(
        identity=identity,
        source_context_digest=source_digest,
        token_ids=tokens,
        identity_salt=salt,
        prefix_tokens=prefix_tokens,
        storage_mode="per_token_rows",
    )
    return store, identity, tokens, source_digest, chunks


def test_sparse_aliases_use_bounded_linear_descriptor_segments(tmp_path: Path) -> None:
    store, identity, tokens, source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=40,
    )

    receipt = store.publish_prefix_aliases(
        identity=identity,
        source_context_digest=source_digest,
        token_ids=tokens,
        identity_salt=identity.storage_key,
        storage_mode="per_token_rows",
    )

    assert receipt.aliases_published == 3
    assert receipt.segments_published == 3
    aliases = sorted((tmp_path / "prefix-aliases" / identity.storage_key).glob("*.json"))
    segments = sorted((tmp_path / "prefix-index" / identity.storage_key).glob("*.spix"))
    assert len(aliases) == 3
    assert len(segments) == 3
    decoded_segments = [json.loads(path.read_bytes()) for path in segments]
    assert sum(len(segment["descriptors"]) for segment in decoded_segments) == 40
    assert all(1 <= len(segment["descriptors"]) <= 16 for segment in decoded_segments)
    assert {segment["schema"] for segment in decoded_segments} == {
        "sparkcache-prefix-descriptor-segment/v1"
    }
    assert {json.loads(path.read_bytes())["schema"] for path in aliases} == {
        "sparkcache-prefix-alias/v1"
    }


def test_alias_lookup_flattens_to_existing_manifest_and_restores(tmp_path: Path) -> None:
    store, identity, tokens, _source_digest, chunks = _publish_source(
        tmp_path,
        chunk_count=20,
        prefix_tokens=(1024, 4096),
    )
    prefix_digest = context_prefix_digest(tokens, identity.storage_key, token_count=1024)

    assert not store.lookup(identity, prefix_digest).is_hit
    lookup = store.lookup(
        identity,
        prefix_digest,
        storage_mode="per_token_rows",
    )

    assert lookup.is_hit, lookup.reason
    assert lookup._manifest is not None
    assert lookup._manifest["committed_tokens"] == 1024
    assert len(lookup._manifest["chunks"]) == 4
    assert store.restore(lookup) == chunks[:4]


def test_alias_publication_rejects_opaque_page_storage_and_unbound_tokens(
    tmp_path: Path,
) -> None:
    store, identity, tokens, source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=2,
        prefix_tokens=(256,),
    )
    with pytest.raises(ValueError, match="per_token_rows"):
        store.publish_prefix_aliases(
            identity=identity,
            source_context_digest=source_digest,
            token_ids=tokens,
            identity_salt=identity.storage_key,
            prefix_tokens=(256,),
            storage_mode="block_pages_v1",
        )
    with pytest.raises(CommitConflict, match="source digest disagrees"):
        store.publish_prefix_aliases(
            identity=identity,
            source_context_digest=source_digest,
            token_ids=(*tokens[:-1], 999),
            identity_salt=identity.storage_key,
            prefix_tokens=(256,),
            storage_mode="per_token_rows",
        )


@pytest.mark.parametrize("damage", ["alias", "tail_segment", "missing_parent"])
def test_damaged_or_incomplete_alias_metadata_is_a_cache_miss(
    tmp_path: Path,
    damage: str,
) -> None:
    store, identity, tokens, _source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=32,
        prefix_tokens=(6144,),
    )
    digest = context_prefix_digest(tokens, identity.storage_key, token_count=6144)
    alias_path = tmp_path / "prefix-aliases" / identity.storage_key / f"{digest}.json"
    alias = json.loads(alias_path.read_bytes())
    tail_path = (
        tmp_path
        / "prefix-index"
        / identity.storage_key
        / f"{alias['tail_segment_sha256']}.spix"
    )
    if damage == "alias":
        alias["unexpected"] = True
        alias_path.write_text(json.dumps(alias), encoding="ascii")
    elif damage == "tail_segment":
        tail_path.write_bytes(tail_path.read_bytes() + b"damage")
    else:
        tail = json.loads(tail_path.read_bytes())
        parent_path = (
            tmp_path
            / "prefix-index"
            / identity.storage_key
            / f"{tail['parent_sha256']}.spix"
        )
        parent_path.unlink()

    lookup = store.lookup(identity, digest, storage_mode="per_token_rows")

    assert not lookup.is_hit
    assert lookup.reason in {"absent", "corrupt"}


def test_prefix_publication_is_idempotent(tmp_path: Path) -> None:
    store, identity, tokens, source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=20,
        prefix_tokens=(1024, 4096),
    )
    before = {
        path.relative_to(tmp_path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.glob("prefix-*/*/*")
    }

    receipt = store.publish_prefix_aliases(
        identity=identity,
        source_context_digest=source_digest,
        token_ids=tokens,
        identity_salt=identity.storage_key,
        prefix_tokens=(1024, 4096),
        storage_mode="per_token_rows",
    )
    after = {
        path.relative_to(tmp_path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in tmp_path.glob("prefix-*/*/*")
    }

    assert receipt.aliases_published == 2
    assert receipt.segments_published == 1
    assert after == before


def test_exact_manifest_wins_over_prefix_alias(tmp_path: Path) -> None:
    store, identity, tokens, _source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=2,
        prefix_tokens=(256,),
    )
    prefix_digest = context_prefix_digest(tokens, identity.storage_key, token_count=256)
    exact = _chunk(0, label=b"exact")
    store.commit(
        identity=identity,
        context_digest=prefix_digest,
        chunks=(exact,),
        span_tokens=256,
    )

    lookup = store.lookup(
        identity,
        prefix_digest,
        storage_mode="per_token_rows",
    )

    assert lookup.is_hit, lookup.reason
    assert store.restore(lookup) == (exact,)


def test_corrupt_exact_manifest_does_not_fall_through_to_alias(tmp_path: Path) -> None:
    store, identity, tokens, _source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=2,
        prefix_tokens=(256,),
    )
    prefix_digest = context_prefix_digest(tokens, identity.storage_key, token_count=256)
    store.commit(
        identity=identity,
        context_digest=prefix_digest,
        chunks=(_chunk(0, label=b"exact"),),
        span_tokens=256,
    )
    exact_path = tmp_path / "manifests" / identity.storage_key / f"{prefix_digest}.json"
    exact_path.write_bytes(b"damaged")

    lookup = store.lookup(
        identity,
        prefix_digest,
        storage_mode="per_token_rows",
    )

    assert not lookup.is_hit
    assert lookup.reason == "corrupt"
    assert store.invalidate(identity, prefix_digest)
    alias = store.lookup(
        identity,
        prefix_digest,
        storage_mode="per_token_rows",
    )
    assert alias.is_hit, alias.reason


def test_invalidated_damaged_alias_can_be_republished(tmp_path: Path) -> None:
    store, identity, tokens, source_digest, _chunks = _publish_source(
        tmp_path,
        chunk_count=2,
        prefix_tokens=(256,),
    )
    prefix_digest = context_prefix_digest(tokens, identity.storage_key, token_count=256)
    alias_path = (
        tmp_path / "prefix-aliases" / identity.storage_key / f"{prefix_digest}.json"
    )
    alias_path.write_bytes(b"damaged")
    assert store.lookup(
        identity,
        prefix_digest,
        storage_mode="per_token_rows",
    ).reason == "corrupt"

    assert store.invalidate(identity, prefix_digest)
    store.publish_prefix_aliases(
        identity=identity,
        source_context_digest=source_digest,
        token_ids=tokens,
        identity_salt=identity.storage_key,
        prefix_tokens=(256,),
        storage_mode="per_token_rows",
    )

    repaired = store.lookup(
        identity,
        prefix_digest,
        storage_mode="per_token_rows",
    )
    assert repaired.is_hit, repaired.reason
