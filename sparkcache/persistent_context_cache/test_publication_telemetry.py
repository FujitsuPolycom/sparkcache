from __future__ import annotations

import dataclasses
import hashlib
from pathlib import Path
from unittest import mock

import pytest

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ContextChunk,
    ManifestStore,
    StateRecord,
)


def _identity(*, publication_schema: str = "") -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-layout",
        rope_layout="test-rope",
        tp_degree=1,
        dcp_degree=1,
        chunk_tokens=256,
        publication_schema=publication_schema,
    )


def _chunk(start: int, label: bytes) -> ContextChunk:
    return ContextChunk(
        start,
        start + 256,
        {
            StateRecord.TARGET_CKV: b"target-" + label,
            StateRecord.SPARSE_INDEXER: b"index-" + label,
            StateRecord.MTP_DRAFT_KV: b"draft-" + label,
            StateRecord.BOUNDARY_HIDDEN: b"hidden-" + label,
            StateRecord.LOGICAL_POSITIONS: b"positions-" + label,
        },
    )


def _digest(label: bytes) -> str:
    return hashlib.sha256(label).hexdigest()


def test_complete_snapshot_receipt_distinguishes_deduplication(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    first = store.commit(
        identity=_identity(),
        context_digest=_digest(b"complete"),
        chunks=(_chunk(0, b"same"),),
    )
    second = store.commit(
        identity=_identity(),
        context_digest=_digest(b"complete"),
        chunks=(_chunk(0, b"same"),),
    )

    assert first.publication is not None
    assert second.publication is not None
    assert first.publication.kind == "complete_snapshot"
    assert first.publication.outcome == "committed"
    assert first.publication.logical_payload_bytes > 0
    assert first.publication.unique_object_bytes == first.encoded_bytes
    assert first.publication.deduplicated_bytes == 0
    assert second.publication.logical_payload_bytes == first.publication.logical_payload_bytes
    assert second.publication.unique_object_bytes == 0
    assert second.publication.deduplicated_bytes == first.encoded_bytes
    assert second.publication.staged_write_bytes == first.encoded_bytes
    assert second.publication.format_compact().startswith(
        "sparkcache: publish kind=complete_snapshot outcome=committed "
    )


def test_abort_reports_unreferenced_unique_objects_without_a_commit(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    transaction = store.begin_context(
        identity=_identity(),
        context_digest=_digest(b"aborted"),
    )
    chunk_receipt = transaction.append_chunk(_chunk(0, b"orphan"))
    transaction.abort()

    counters = store.publication_telemetry_snapshot()
    assert counters.publication_attempts == 1
    assert counters.committed_publications == 0
    assert counters.aborted_publications == 1
    assert counters.unique_object_bytes == chunk_receipt.encoded_bytes
    assert counters.committed_unique_object_bytes == 0
    assert counters.uncommitted_unique_object_bytes == chunk_receipt.encoded_bytes
    assert counters.staged_write_bytes == chunk_receipt.encoded_bytes


def test_failed_manifest_publication_preserves_attempted_write_accounting(
    tmp_path: Path,
) -> None:
    store = ManifestStore(tmp_path)
    transaction = store.begin_context(
        identity=_identity(),
        context_digest=_digest(b"failed"),
    )
    chunk_receipt = transaction.append_chunk(_chunk(0, b"failed"))
    with mock.patch.object(
        cache_manifest.os,
        "link",
        side_effect=OSError("synthetic publication failure"),
    ):
        with pytest.raises(OSError, match="synthetic publication failure"):
            transaction.commit_manifest()

    counters = store.publication_telemetry_snapshot()
    assert counters.failed_publications == 1
    assert counters.committed_publications == 0
    assert counters.unique_object_bytes == chunk_receipt.encoded_bytes
    assert counters.uncommitted_unique_object_bytes == chunk_receipt.encoded_bytes
    assert counters.staged_write_bytes > chunk_receipt.encoded_bytes
    transaction.abort()


def test_row_tail_receipt_separates_extension_from_reused_base(tmp_path: Path) -> None:
    from sparkcache.spark_context_cache_codec import context_prefix_digest

    identity = _identity(publication_schema="tail-cow-v1")
    tokens = tuple(range(512))
    salt = "publication-telemetry-tail"
    base_digest = context_prefix_digest(tokens, salt, token_count=256)
    store = ManifestStore(tmp_path)
    base_receipt = store.commit(
        identity=identity,
        context_digest=base_digest,
        chunks=(_chunk(0, b"base"),),
    )
    tail_receipt = store.commit_extension(
        identity=identity,
        base_context_digest=base_digest,
        token_ids=tokens,
        identity_salt=salt,
        tail_chunks=(_chunk(256, b"tail"),),
    )

    assert base_receipt.publication is not None
    assert tail_receipt.publication is not None
    assert tail_receipt.publication.kind == "row_tail"
    assert tail_receipt.publication.logical_payload_bytes > 0
    assert (
        tail_receipt.publication.reused_base_bytes
        == base_receipt.publication.logical_payload_bytes
    )
    assert tail_receipt.publication.committed_unique_object_bytes > 0


def test_page_delta_receipt_counts_delta_and_reused_snapshot(tmp_path: Path) -> None:
    from sparkcache.spark_context_cache_codec import context_prefix_digest
    from sparkcache.spark_context_cache_hybrid import (
        PageGroup,
        PageLayer,
        PageLayout,
        encode_page_delta,
        encode_page_snapshot,
    )

    identity = dataclasses.replace(
        _identity(publication_schema="page-tail-cow-v1"),
        record_schema=("target_ckv", "logical_positions"),
    )
    layout = PageLayout((PageGroup(128, (PageLayer("page", "u8", (64,), 64),)),))
    base_snapshot = encode_page_snapshot(layout, (2,), {"page": b"A" * 128})
    result_snapshot = encode_page_snapshot(
        layout,
        (3,),
        {"page": b"A" * 128 + b"B" * 64},
    )
    tokens = tuple(range(512))
    salt = "publication-telemetry-page"
    base_digest = context_prefix_digest(tokens, salt, token_count=256)
    store = ManifestStore(tmp_path)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=base_digest,
        span_tokens=256,
        snapshot=base_snapshot,
    )
    delta_receipt = store.commit_page_extension(
        identity=identity,
        base_context_digest=base_digest,
        token_ids=tokens,
        identity_salt=salt,
        layout=layout,
        base_block_counts=(2,),
        result_block_counts=(3,),
        base_boundary_tokens=256,
        result_boundary_tokens=512,
        result_snapshot=result_snapshot,
    )

    assert delta_receipt.publication is not None
    assert delta_receipt.publication.kind == "page_delta"
    expected_delta = encode_page_delta(
        layout,
        base_snapshot,
        result_snapshot,
        base_block_counts=(2,),
        result_block_counts=(3,),
        base_boundary_tokens=256,
        result_boundary_tokens=512,
    )
    assert delta_receipt.publication.logical_payload_bytes == len(expected_delta)
    assert delta_receipt.publication.reused_base_bytes == len(base_snapshot)


def test_counters_are_monotonic_and_have_stable_operator_output(tmp_path: Path) -> None:
    store = ManifestStore(tmp_path)
    snapshots = [store.publication_telemetry_snapshot()]
    for index in range(2):
        store.commit(
            identity=_identity(),
            context_digest=_digest(b"monotonic"),
            chunks=(_chunk(0, b"monotonic"),),
        )
        snapshots.append(store.publication_telemetry_snapshot())

    numeric_fields = (
        "publication_attempts",
        "committed_publications",
        "logical_payload_bytes",
        "unique_object_bytes",
        "staged_write_bytes",
        "deduplicated_bytes",
    )
    for before, after in zip(snapshots, snapshots[1:]):
        for field_name in numeric_fields:
            assert getattr(after, field_name) >= getattr(before, field_name)
    final = snapshots[-1]
    assert final.as_dict()["schema"] == "sparkcache-publication-telemetry/v1"
    line = final.format_compact()
    assert line.startswith("sparkcache: publication commits=2 ")
    assert "staged=" in line
    assert "dedup=" in line
