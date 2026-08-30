from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
import sparkcache.spark_context_cache_native_hybrid_restore as native_hybrid
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ManifestStore,
)
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
    plan_page_snapshot,
)
from sparkcache.spark_context_cache_native_hybrid_restore import (
    build_page_copy_spans,
    build_page_object_spans,
    execute_native_hybrid_restore,
    plan_page_slabs,
)
from sparkcache.spark_context_cache_native_placement import RestoreState


def test_page_copy_spans_cover_payload_once_in_layout_order() -> None:
    layout = PageLayout(
        (
            PageGroup(
                256,
                (
                    PageLayer("a", "torch.uint8", (4,), 4),
                    PageLayer("b", "torch.uint8", (2,), 2),
                ),
            ),
            PageGroup(1, (PageLayer("state", "torch.uint8", (3,), 3),)),
        )
    )
    encoded = encode_page_snapshot(
        layout,
        (2, 1),
        {"a": bytes(8), "b": bytes(4), "state": bytes(3)},
    )
    plan = plan_page_snapshot(layout, encoded, (2, 1))

    spans = build_page_copy_spans(plan)

    assert [span.destination_index for span in spans] == [0, 1, 2]
    assert [span.snapshot_offset_bytes for span in spans] == [0, 8, 12]
    assert [span.destination_byte_offset for span in spans] == [0, 0, 0]
    assert sum(span.byte_count for span in spans) == 15
    assert spans[0].arena_offset_bytes == 0

    slabs = plan_page_slabs(plan, arena_bytes=7)
    assert [(slab.payload_start, slab.payload_end) for slab in slabs] == [
        (0, 7),
        (7, 14),
        (14, 15),
    ]
    flattened = [span for slab in slabs for span in slab.spans]
    assert [span.snapshot_offset_bytes for span in flattened] == [0, 7, 8, 12, 14]
    assert sum(span.byte_count for span in flattened) == 15


def test_flat_macro_objects_authenticate_before_direct_page_submission(
    tmp_path, monkeypatch
) -> None:
    layout = PageLayout((PageGroup(2, (PageLayer("page", "torch.uint8", (4,), 4),)),))
    encoded = encode_page_snapshot(layout, (4,), {"page": bytes(range(16))})
    plan = plan_page_snapshot(layout, encoded, (4,))
    cut = plan.header_bytes + 5
    identity = CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="test-block-pages-v1",
        rope_layout="test-rope-v1",
        tp_degree=1,
        dcp_degree=1,
        record_schema=("target_ckv", "logical_positions"),
    )
    digest = hashlib.sha256(b"native-flat-macro").hexdigest()
    store = ManifestStore(tmp_path)
    monkeypatch.setattr(cache_manifest, "_PAGE_SNAPSHOT_OBJECT_BYTES", cut)
    store.commit_page_snapshot(
        identity=identity,
        context_digest=digest,
        span_tokens=256,
        snapshot=encoded,
    )
    lookup = store.lookup(identity, digest, verify_chunks=False)
    assert lookup.is_hit
    manifest = lookup._manifest
    assert manifest is not None
    descriptors = manifest["snapshot_objects"]
    chunk_root = tmp_path / "chunks"

    class Arena:
        def __init__(self, size: int) -> None:
            self.payload = bytearray(size)

    class Transaction:
        def __init__(self) -> None:
            self.arenas = (Arena(cut), Arena(cut))
            self.submissions = []
            self.state = RestoreState.PARKED
            self.can_resume = False

        def __enter__(self):
            return self

        def __exit__(self, exc_type, *_args):
            if exc_type is not None:
                self.state = RestoreState.ABORTED
                self.can_resume = False
            return None

        def acquire_arena(self, index):
            return self.arenas[index]

        def submit_page_slab(self, *, arena_index, arena_used_bytes, spans):
            source = self.arenas[arena_index].payload
            self.submissions.extend(
                (
                    span.snapshot_offset_bytes,
                    bytes(
                        source[
                            span.arena_offset_bytes : span.arena_offset_bytes
                            + span.byte_count
                        ]
                    ),
                )
                for span in spans
            )

        def finish(self):
            self.state = RestoreState.FINISHED
            self.can_resume = True
            return SimpleNamespace(
                source_bytes=len(encoded) - plan.header_bytes,
                slabs_submitted=2,
                scatter_kernel_launches=2,
                slot_uploads=1,
                destination_table_uploads=1,
                device_error=0,
                staged_h2d_bytes=0,
            )

    class Adapter:
        def __init__(self) -> None:
            self.transactions = []

        def begin_parked_page_restore(self, *_args, **_kwargs):
            transaction = Transaction()
            self.transactions.append(transaction)
            return transaction

    adapter = Adapter()
    monkeypatch.setattr(
        native_hybrid.native,
        "arena_memoryview",
        lambda arena, *, length: memoryview(arena.payload)[:length],
    )

    result = execute_native_hybrid_restore(
        adapter=adapter,
        request_id="flat-macro-native",
        lookup=lookup,
        cache_root=tmp_path,
        layout=layout,
        group_slots=((3, 4, 5, 6),),
        expected_span_tokens=256,
        arena_bytes=cut,
    )

    transaction = adapter.transactions[-1]
    restored_payload = b"".join(
        payload for _offset, payload in sorted(transaction.submissions)
    )
    assert restored_payload == encoded[plan.header_bytes :]
    assert result.source_bytes == len(encoded)
    assert result.slabs == 2

    forged_lookup = type(lookup)(
        True,
        "hit",
        manifest_digest="0" * 64,
        _manifest=manifest,
        root_kind="page_snapshot",
    )
    transactions_before_forgery = len(adapter.transactions)
    with pytest.raises(
        native_hybrid.NativeHybridRestoreError,
        match="identity is not authenticated",
    ):
        execute_native_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-forged-lookup",
            lookup=forged_lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    assert len(adapter.transactions) == transactions_before_forgery

    damaged_path = chunk_root / f"{descriptors[1]['sha256']}.spcc"
    healthy = damaged_path.read_bytes()
    damaged = bytearray(healthy)
    damaged[-1] ^= 0xFF
    damaged_path.write_bytes(damaged)
    with pytest.raises(
        native_hybrid.NativeHybridRestoreError,
        match="SHA-256 mismatch",
    ):
        execute_native_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-corrupt",
            lookup=lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    damaged_transaction = adapter.transactions[-1]
    assert damaged_transaction.submissions
    assert damaged_transaction.state is RestoreState.ABORTED
    assert not damaged_transaction.can_resume

    damaged_path.write_bytes(healthy)
    manifest_path = tmp_path / "manifests" / identity.storage_key / f"{digest}.json"
    wrong_root = dict(manifest)
    wrong_root["snapshot_sha256"] = "0" * 64
    wrong_root.pop("metadata_sha256")
    wrong_root["metadata_sha256"] = hashlib.sha256(
        cache_manifest._canonical_json(wrong_root)
    ).hexdigest()
    manifest_path.write_bytes(cache_manifest._canonical_json(wrong_root))
    wrong_lookup = store.lookup(identity, digest, verify_chunks=False)
    assert wrong_lookup.is_hit
    with pytest.raises(
        native_hybrid.NativeHybridRestoreError,
        match="snapshot checksum mismatch",
    ):
        execute_native_hybrid_restore(
            adapter=adapter,
            request_id="flat-macro-wrong-root-digest",
            lookup=wrong_lookup,
            cache_root=tmp_path,
            layout=layout,
            group_slots=((3, 4, 5, 6),),
            expected_span_tokens=256,
            arena_bytes=cut,
        )
    root_digest_transaction = adapter.transactions[-1]
    assert len(root_digest_transaction.submissions) == 2
    assert root_digest_transaction.state is RestoreState.ABORTED
    assert not root_digest_transaction.can_resume


def test_page_object_spans_exclude_header_and_cover_payload_contiguously() -> None:
    layout = PageLayout((PageGroup(2, (PageLayer("page", "torch.uint8", (4,), 4),)),))
    encoded = encode_page_snapshot(layout, (4,), {"page": bytes(range(16))})
    plan = plan_page_snapshot(layout, encoded, (4,))
    cut = plan.header_bytes + 5

    first = build_page_object_spans(
        plan,
        encoded_start=0,
        encoded_end=cut,
    )
    second = build_page_object_spans(
        plan,
        encoded_start=cut,
        encoded_end=len(encoded),
    )

    assert [span.snapshot_offset_bytes for span in (*first, *second)] == [0, 5]
    assert sum(span.byte_count for span in (*first, *second)) == 16
    assert first[0].arena_offset_bytes == plan.header_bytes
    assert second[0].arena_offset_bytes == 0
