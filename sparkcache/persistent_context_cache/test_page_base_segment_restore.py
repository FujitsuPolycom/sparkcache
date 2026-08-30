from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import tempfile
import threading

import pytest

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
from sparkcache.page_base_read_flights import (
    PageBaseReadError,
    PageBaseReadFlightKey,
    PageBaseReadFlights,
)
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheFormatError,
    CacheIdentity,
    ContextChunk,
    ManifestStore,
    StateRecord,
)
from sparkcache.spark_context_cache_codec import context_prefix_digest, pack_positions
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
    split_snapshot,
)


def _identity() -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="nvfp4-ds-mla-v1",
        rope_layout="glm53-hybrid-v1",
        tp_degree=4,
        dcp_degree=1,
        chunk_tokens=256,
        record_schema=("target_ckv", "logical_positions"),
        publication_schema="page-tail-cow-v1",
    )


def _layout() -> PageLayout:
    return PageLayout((PageGroup(256, (PageLayer("page", "u8", (32,), 32),)),))


def _snapshot(layout: PageLayout, labels: bytes) -> bytes:
    return encode_page_snapshot(
        layout,
        (len(labels),),
        {"page": b"".join(bytes((label,)) * 32 for label in labels)},
    )


def _commit_flat_base(
    store: ManifestStore,
    *,
    identity: CacheIdentity,
    layout: PageLayout,
    salt: str,
    macro: bool,
) -> tuple[tuple[int, ...], str, bytes]:
    tokens = tuple(range(256))
    digest = context_prefix_digest(tokens, salt, token_count=256)
    snapshot = _snapshot(layout, b"A")
    if macro:
        store.commit_page_snapshot(
            identity=identity,
            context_digest=digest,
            span_tokens=256,
            snapshot=snapshot,
        )
    else:
        store.commit(
            identity=identity,
            context_digest=digest,
            span_tokens=256,
            chunks=(
                ContextChunk(
                    0,
                    256,
                    {
                        StateRecord.LOGICAL_POSITIONS: pack_positions(tokens),
                        StateRecord.TARGET_CKV: snapshot,
                    },
                ),
            ),
        )
    return tokens, digest, snapshot


def _commit_extension(
    store: ManifestStore,
    *,
    identity: CacheIdentity,
    layout: PageLayout,
    salt: str,
    base_tokens: tuple[int, ...],
    base_digest: str,
    base_blocks: int,
    result_blocks: int,
    label: int,
) -> tuple[str, bytes, int]:
    result_tokens = result_blocks * 256
    tokens = base_tokens + tuple(
        label * 10_000 + index for index in range(result_tokens - len(base_tokens))
    )
    labels = bytes((65 + (index + label) % 26 for index in range(result_blocks)))
    labels = bytes((65,)) * base_blocks + labels[base_blocks:]
    snapshot = _snapshot(layout, labels)
    store.commit_page_extension(
        identity=identity,
        base_context_digest=base_digest,
        token_ids=tokens,
        identity_salt=salt,
        layout=layout,
        base_block_counts=(base_blocks,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=len(base_tokens),
        result_boundary_tokens=result_tokens,
        result_snapshot=snapshot,
    )
    return (
        context_prefix_digest(tokens, salt, token_count=result_tokens),
        snapshot,
        result_tokens,
    )


def _flight_key(evidence: object) -> PageBaseReadFlightKey:
    return PageBaseReadFlightKey(
        worker_generation="test-worker",
        storage_mode="block_pages_v1",
        evidence=evidence,  # type: ignore[arg-type]
    )


def _convert_result_to_v1(
    root: Path,
    store: ManifestStore,
    identity: CacheIdentity,
    digest: str,
) -> None:
    manifest_path = root / "manifests" / identity.storage_key / f"{digest}.json"
    manifest = json.loads(manifest_path.read_bytes())
    encoded_delta = store._read_page_delta_objects(
        manifest["delta_objects"],
        encoded_bytes=manifest["delta_encoded_bytes"],
        encoded_sha256=manifest["delta_sha256"],
    )
    parts = split_snapshot(
        encoded_delta,
        manifest["committed_tokens"] // identity.chunk_tokens,
    )
    descriptors: list[dict[str, object]] = []
    for index, part in enumerate(parts):
        chunk = ContextChunk(
            index * identity.chunk_tokens,
            (index + 1) * identity.chunk_tokens,
            {
                StateRecord.LOGICAL_POSITIONS: pack_positions(
                    range(
                        index * identity.chunk_tokens,
                        (index + 1) * identity.chunk_tokens,
                    )
                ),
                StateRecord.TARGET_CKV: bytes(part),
            },
        )
        encoded = cache_manifest._encode_chunk(chunk)
        chunk_digest = hashlib.sha256(encoded).hexdigest()
        cache_manifest._publish_immutable(
            root / "chunks" / f"{chunk_digest}.spcc",
            encoded,
        )
        descriptors.append(
            {
                "sha256": chunk_digest,
                "bytes": len(encoded),
                "logical_start": chunk.logical_start,
                "logical_end": chunk.logical_end,
            }
        )
    legacy = {
        key: value
        for key, value in manifest.items()
        if key
        not in {
            "delta_objects",
            "delta_encoded_bytes",
            "delta_object_bytes",
            "delta_sha256",
            "logical_chunk_tokens",
            "metadata_sha256",
        }
    }
    legacy["schema"] = "sparkcache-page-delta-manifest/v1"
    legacy["delta_chunks"] = descriptors
    legacy["metadata_sha256"] = hashlib.sha256(
        cache_manifest._canonical_json(legacy)
    ).hexdigest()
    manifest_path.write_bytes(cache_manifest._canonical_json(legacy))


def test_c16_distinct_roots_share_one_flat_macro_base_read() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ManifestStore(root)
        identity = _identity()
        layout = _layout()
        salt = "c16-shared-flat-macro-base"
        base_tokens, base_digest, _base_snapshot = _commit_flat_base(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            macro=True,
        )
        results = [
            _commit_extension(
                store,
                identity=identity,
                layout=layout,
                salt=salt,
                base_tokens=base_tokens,
                base_digest=base_digest,
                base_blocks=1,
                result_blocks=2 + index % 2,
                label=index + 1,
            )
            for index in range(16)
        ]
        lookups = [store.lookup(identity, digest, verify_chunks=False) for digest, _, _ in results]
        evidences = [
            store.page_delta_base_read_evidence(
                lookup,
                layout=layout,
                result_block_counts=(2 + index % 2,),
                result_boundary_tokens=results[index][2],
            )
            for index, lookup in enumerate(lookups)
        ]
        assert len(set(evidences)) == 1
        key = _flight_key(evidences[0])
        flights = PageBaseReadFlights()
        request_ids = tuple(f"request-{index}" for index in range(16))
        assert flights.register_cohort(key, request_ids).member_ids == request_ids

        base_reads = 0
        private_delta_reads = 0
        counter_lock = threading.Lock()
        original_base = store._read_page_snapshot_objects
        original_delta = store._read_page_delta_objects

        def read_base_objects(*args: object, **kwargs: object) -> bytearray:
            nonlocal base_reads
            with counter_lock:
                base_reads += 1
            return original_base(*args, **kwargs)  # type: ignore[arg-type]

        def read_delta_objects(*args: object, **kwargs: object) -> bytearray:
            nonlocal private_delta_reads
            with counter_lock:
                private_delta_reads += 1
            return original_delta(*args, **kwargs)  # type: ignore[arg-type]

        def restore(index: int) -> bytes | bytearray:
            request_id = request_ids[index]
            return store.restore_page_snapshot(
                lookups[index],
                layout=layout,
                result_block_counts=(2 + index % 2,),
                result_boundary_tokens=results[index][2],
                base_reader=lambda evidence, reader: flights.resolve(
                    request_id,
                    _flight_key(evidence),
                    reader,
                ),
            )

        with (
            pytest.MonkeyPatch.context() as patch,
            ThreadPoolExecutor(max_workers=2) as executor,
        ):
            patch.setattr(store, "_read_page_snapshot_objects", read_base_objects)
            patch.setattr(store, "_read_page_delta_objects", read_delta_objects)
            restored = list(executor.map(restore, range(16)))

        assert restored == [snapshot for _digest, snapshot, _tokens in results]
        assert base_reads == 1
        assert private_delta_reads == 16
        assert flights.snapshot().retained_bytes == 0
        summaries = flights.take_summaries()
        assert len(summaries) == 1
        assert summaries[0]["participants"] == 16
        assert summaries[0]["physical_base_reads"] == 1
        assert summaries[0]["avoided_base_reads"] == 15


def test_v1_and_v2_results_share_one_legacy_flat_base() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ManifestStore(root)
        identity = _identity()
        layout = _layout()
        salt = "mixed-result-schemas"
        base_tokens, base_digest, _snapshot_bytes = _commit_flat_base(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            macro=False,
        )
        results = [
            _commit_extension(
                store,
                identity=identity,
                layout=layout,
                salt=salt,
                base_tokens=base_tokens,
                base_digest=base_digest,
                base_blocks=1,
                result_blocks=2,
                label=index + 1,
            )
            for index in range(2)
        ]
        _convert_result_to_v1(root, store, identity, results[0][0])
        lookups = [store.lookup(identity, item[0], verify_chunks=False) for item in results]
        evidences = [
            store.page_delta_base_read_evidence(
                lookup,
                layout=layout,
                result_block_counts=(2,),
                result_boundary_tokens=512,
            )
            for lookup in lookups
        ]
        assert len(set(evidences)) == 1
        assert evidences[0].base_root_kind == "manifest"
        flights = PageBaseReadFlights()
        key = _flight_key(evidences[0])
        flights.register_cohort(key, ("v1", "v2"))

        restored = [
            store.restore_page_snapshot(
                lookup,
                layout=layout,
                result_block_counts=(2,),
                result_boundary_tokens=512,
                base_reader=lambda evidence, reader, request_id=request_id: (
                    flights.resolve(request_id, _flight_key(evidence), reader)
                ),
            )
            for request_id, lookup in zip(("v1", "v2"), lookups, strict=True)
        ]
        assert restored == [item[1] for item in results]
        assert flights.take_summaries()[0]["avoided_base_reads"] == 1


def test_flat_base_declared_geometry_must_match_authenticated_object_bytes() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ManifestStore(root)
        identity = _identity()
        layout = _layout()
        salt = "base-geometry-mismatch"
        base_tokens, base_digest, _snapshot_bytes = _commit_flat_base(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            macro=True,
        )
        result_digest, _result_snapshot, _result_tokens = _commit_extension(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            base_tokens=base_tokens,
            base_digest=base_digest,
            base_blocks=1,
            result_blocks=2,
            label=1,
        )
        manifest_path = (
            root / "manifests" / identity.storage_key / f"{result_digest}.json"
        )
        manifest = json.loads(manifest_path.read_bytes())
        manifest["base_block_counts"] = [2]
        manifest.pop("metadata_sha256")
        manifest["metadata_sha256"] = hashlib.sha256(
            cache_manifest._canonical_json(manifest)
        ).hexdigest()
        manifest_path.write_bytes(cache_manifest._canonical_json(manifest))
        lookup = store.lookup(identity, result_digest, verify_chunks=False)
        assert lookup.is_hit, lookup.reason

        with pytest.raises(CacheFormatError, match="page snapshot base geometry"):
            store.page_delta_base_read_evidence(
                lookup,
                layout=layout,
                result_block_counts=(2,),
                result_boundary_tokens=512,
            )


@pytest.mark.parametrize("base_schema", ["v1", "v2"])
def test_nested_page_delta_base_schemas_are_shareable(base_schema: str) -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ManifestStore(root)
        identity = _identity()
        layout = _layout()
        salt = f"nested-base-{base_schema}"
        initial_tokens, initial_digest, _initial_snapshot = _commit_flat_base(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            macro=True,
        )
        base_digest, base_snapshot, _ = _commit_extension(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            base_tokens=initial_tokens,
            base_digest=initial_digest,
            base_blocks=1,
            result_blocks=2,
            label=1,
        )
        base_tokens = initial_tokens + tuple(10_000 + index for index in range(256))
        if base_schema == "v1":
            _convert_result_to_v1(root, store, identity, base_digest)
        results = [
            _commit_extension(
                store,
                identity=identity,
                layout=layout,
                salt=salt,
                base_tokens=base_tokens,
                base_digest=base_digest,
                base_blocks=2,
                result_blocks=3 + index,
                label=index + 2,
            )
            for index in range(2)
        ]
        lookups = [store.lookup(identity, item[0], verify_chunks=False) for item in results]
        evidences = [
            store.page_delta_base_read_evidence(
                lookup,
                layout=layout,
                result_block_counts=(3 + index,),
                result_boundary_tokens=item[2],
            )
            for index, (lookup, item) in enumerate(zip(lookups, results, strict=True))
        ]
        assert len(set(evidences)) == 1
        assert evidences[0].base_root_kind == "page_delta"
        flights = PageBaseReadFlights()
        key = _flight_key(evidences[0])
        flights.register_cohort(key, ("left", "right"))
        restored = [
            store.restore_page_snapshot(
                lookup,
                layout=layout,
                result_block_counts=(3 + index,),
                result_boundary_tokens=results[index][2],
                base_reader=lambda evidence, reader, request_id=request_id: (
                    flights.resolve(request_id, _flight_key(evidence), reader)
                ),
            )
            for index, (request_id, lookup) in enumerate(
                zip(("left", "right"), lookups, strict=True)
            )
        ]
        assert restored == [item[1] for item in results]
        assert base_snapshot != restored[0]
        assert flights.take_summaries()[0]["physical_base_reads"] == 1


def test_corrupt_base_rejects_cohort_and_private_delta_corruption_is_local() -> None:
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        store = ManifestStore(root)
        identity = _identity()
        layout = _layout()
        salt = "corruption-isolation"
        base_tokens, base_digest, _base_snapshot = _commit_flat_base(
            store,
            identity=identity,
            layout=layout,
            salt=salt,
            macro=True,
        )
        results = [
            _commit_extension(
                store,
                identity=identity,
                layout=layout,
                salt=salt,
                base_tokens=base_tokens,
                base_digest=base_digest,
                base_blocks=1,
                result_blocks=2,
                label=index + 1,
            )
            for index in range(3)
        ]
        lookups = [store.lookup(identity, item[0], verify_chunks=False) for item in results]
        evidence = store.page_delta_base_read_evidence(
            lookups[0],
            layout=layout,
            result_block_counts=(2,),
            result_boundary_tokens=512,
        )
        key = _flight_key(evidence)

        corrupt_delta_manifest = lookups[0]._manifest
        assert corrupt_delta_manifest is not None
        delta_path = root / "chunks" / f"{corrupt_delta_manifest['delta_objects'][0]['sha256']}.spcc"
        delta_payload = bytearray(delta_path.read_bytes())
        delta_payload[-1] ^= 1
        delta_path.write_bytes(delta_payload)

        flights = PageBaseReadFlights()
        flights.register_cohort(key, ("corrupt-delta", "good-a", "good-b"))
        outcomes: list[bytes | bytearray | Exception] = []
        for request_id, lookup in zip(
            ("corrupt-delta", "good-a", "good-b"),
            lookups,
            strict=True,
        ):
            try:
                outcomes.append(
                    store.restore_page_snapshot(
                        lookup,
                        layout=layout,
                        result_block_counts=(2,),
                        result_boundary_tokens=512,
                        base_reader=lambda ev, reader, rid=request_id: flights.resolve(
                            rid,
                            _flight_key(ev),
                            reader,
                        ),
                    )
                )
            except Exception as error:
                outcomes.append(error)
        assert isinstance(outcomes[0], CacheFormatError)
        assert outcomes[1:] == [results[1][1], results[2][1]]
        assert flights.take_summaries()[0]["outcome"] == "verified"

        base_manifest = lookups[1]._manifest["base_root"]  # type: ignore[index]
        base_path = root / "chunks" / f"{base_manifest['snapshot_objects'][0]['sha256']}.spcc"
        base_payload = bytearray(base_path.read_bytes())
        base_payload[-1] ^= 1
        base_path.write_bytes(base_payload)
        base_flights = PageBaseReadFlights()
        base_flights.register_cohort(key, ("base-a", "base-b"))
        for request_id, lookup in zip(("base-a", "base-b"), lookups[1:], strict=True):
            with pytest.raises(PageBaseReadError):
                store.restore_page_snapshot(
                    lookup,
                    layout=layout,
                    result_block_counts=(2,),
                    result_boundary_tokens=512,
                    base_reader=lambda ev, reader, rid=request_id: base_flights.resolve(
                        rid,
                        _flight_key(ev),
                        reader,
                    ),
                )
        assert base_flights.take_summaries()[0]["outcome"] == "recompute"
