from __future__ import annotations

import dataclasses
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
from sparkcache.persistent_context_cache.cache_manifest import (
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


@dataclasses.dataclass(frozen=True)
class _Fixture:
    identity: CacheIdentity
    layout: PageLayout
    tokens: tuple[int, ...]
    salt: str
    base_digest: str
    result_digest: str
    result_snapshot: bytes
    result_tokens: int
    result_blocks: int


def _commit_fixture(store: ManifestStore, *, result_blocks: int = 16) -> _Fixture:
    identity = _identity()
    layout = PageLayout(
        (PageGroup(256, (PageLayer("page", "u8", (1024,), 1024),)),)
    )
    result_tokens = result_blocks * 256
    tokens = tuple(range(result_tokens))
    salt = "page-delta-macro-object-test"
    base_digest = context_prefix_digest(tokens, salt, token_count=256)
    result_digest = context_prefix_digest(tokens, salt, token_count=result_tokens)
    base_snapshot = encode_page_snapshot(layout, (1,), {"page": b"A" * 1024})
    result_payload = b"A" * 1024 + b"".join(
        bytes((index % 251,)) * 1024 for index in range(1, result_blocks)
    )
    result_snapshot = encode_page_snapshot(
        layout,
        (result_blocks,),
        {"page": result_payload},
    )
    store.commit(
        identity=identity,
        context_digest=base_digest,
        chunks=(
            ContextChunk(
                0,
                256,
                {
                    StateRecord.LOGICAL_POSITIONS: pack_positions(range(256)),
                    StateRecord.TARGET_CKV: base_snapshot,
                },
            ),
        ),
        span_tokens=256,
    )
    store.commit_page_extension(
        identity=identity,
        base_context_digest=base_digest,
        token_ids=tokens,
        identity_salt=salt,
        layout=layout,
        base_block_counts=(1,),
        result_block_counts=(result_blocks,),
        base_boundary_tokens=256,
        result_boundary_tokens=result_tokens,
        result_snapshot=result_snapshot,
    )
    return _Fixture(
        identity=identity,
        layout=layout,
        tokens=tokens,
        salt=salt,
        base_digest=base_digest,
        result_digest=result_digest,
        result_snapshot=result_snapshot,
        result_tokens=result_tokens,
        result_blocks=result_blocks,
    )


def _manifest(root: Path, fixture: _Fixture) -> tuple[Path, dict[str, object]]:
    path = (
        root
        / "manifests"
        / fixture.identity.storage_key
        / f"{fixture.result_digest}.json"
    )
    return path, json.loads(path.read_bytes())


class PageDeltaMacroObjectTests(unittest.TestCase):
    def test_v2_uses_fewer_physical_objects_than_logical_token_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            published_batch_sizes: list[int] = []
            original_publish = cache_manifest._publish_immutable_batch

            def record_publish(objects: list[tuple[Path, bytes]]) -> None:
                published_batch_sizes.append(len(objects))
                original_publish(objects)

            with (
                mock.patch.object(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", 2048),
                mock.patch.object(
                    cache_manifest,
                    "_publish_immutable_batch",
                    side_effect=record_publish,
                ),
            ):
                fixture = _commit_fixture(store)

            _path, manifest = _manifest(root, fixture)
            objects = manifest["delta_objects"]
            self.assertEqual(manifest["schema"], "sparkcache-page-delta-manifest/v2")
            self.assertEqual(manifest["logical_chunk_tokens"], 256)
            self.assertLess(len(objects), fixture.result_tokens // 256)
            self.assertLess(
                manifest["delta_encoded_bytes"],
                len(fixture.result_snapshot),
            )
            base_descriptor = manifest["base_root"]["chunks"][0]
            self.assertTrue(
                (root / "chunks" / f"{base_descriptor['sha256']}.spcc").is_file()
            )
            self.assertLessEqual(
                max(published_batch_sizes),
                cache_manifest._PAGE_DELTA_WRITE_BATCH_SIZE,
            )
            self.assertEqual(
                [item["encoded_start"] for item in objects],
                [0, *[item["encoded_end"] for item in objects[:-1]]],
            )
            lookup = store.lookup(fixture.identity, fixture.result_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(
                store.restore_page_snapshot(
                    lookup,
                    layout=fixture.layout,
                    result_block_counts=(fixture.result_blocks,),
                    result_boundary_tokens=fixture.result_tokens,
                ),
                fixture.result_snapshot,
            )

    def test_live_scale_payload_needs_at_most_24_macro_objects(self) -> None:
        encoded_bytes = 1_575_821_491

        object_count = (
            encoded_bytes + cache_manifest._PAGE_DELTA_OBJECT_BYTES - 1
        ) // cache_manifest._PAGE_DELTA_OBJECT_BYTES

        self.assertEqual(object_count, 24)
        self.assertLess(object_count, 1_024)

    def test_corrupt_macro_object_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            with mock.patch.object(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", 2048):
                fixture = _commit_fixture(store)
            _path, manifest = _manifest(root, fixture)
            descriptor = manifest["delta_objects"][0]
            object_path = root / "chunks" / f"{descriptor['sha256']}.spcc"
            encoded = object_path.read_bytes()
            object_path.write_bytes(encoded[:-1] + bytes((encoded[-1] ^ 0xFF,)))

            lookup = store.lookup(fixture.identity, fixture.result_digest)

            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")

    def test_corrupt_macro_descriptor_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            with mock.patch.object(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", 2048):
                fixture = _commit_fixture(store)
            manifest_path, manifest = _manifest(root, fixture)
            manifest["delta_objects"][0]["encoded_end"] += 1
            manifest_path.write_bytes(cache_manifest._canonical_json(manifest))

            lookup = store.lookup(fixture.identity, fixture.result_digest)

            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")

    def test_restore_reads_only_one_bounded_object_batch_at_a_time(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            with mock.patch.object(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", 2048):
                fixture = _commit_fixture(store, result_blocks=24)
            _path, manifest = _manifest(root, fixture)
            self.assertGreater(
                len(manifest["delta_objects"]),
                cache_manifest._PAGE_DELTA_READ_BATCH_SIZE,
            )
            observed_batch_sizes: list[int] = []
            original = cache_manifest._read_page_delta_object_batch

            def record_batch(
                object_root: Path,
                descriptors: tuple[dict[str, object], ...],
            ) -> tuple[bytes, ...]:
                observed_batch_sizes.append(len(descriptors))
                return original(object_root, descriptors)

            lookup = store.lookup(
                fixture.identity,
                fixture.result_digest,
                verify_chunks=False,
            )
            with mock.patch.object(
                cache_manifest,
                "_read_page_delta_object_batch",
                side_effect=record_batch,
            ):
                restored = store.restore_page_snapshot(
                    lookup,
                    layout=fixture.layout,
                    result_block_counts=(fixture.result_blocks,),
                    result_boundary_tokens=fixture.result_tokens,
                )

            self.assertEqual(restored, fixture.result_snapshot)
            self.assertGreater(len(observed_batch_sizes), 1)
            self.assertLessEqual(
                max(observed_batch_sizes),
                cache_manifest._PAGE_DELTA_READ_BATCH_SIZE,
            )

    def test_v1_page_delta_remains_readable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            with mock.patch.object(cache_manifest, "_PAGE_DELTA_OBJECT_BYTES", 2048):
                fixture = _commit_fixture(store)
            manifest_path, manifest = _manifest(root, fixture)
            encoded_delta = store._read_page_delta_objects(
                manifest["delta_objects"],
                encoded_bytes=manifest["delta_encoded_bytes"],
                encoded_sha256=manifest["delta_sha256"],
            )
            parts = split_snapshot(
                encoded_delta,
                fixture.result_tokens // fixture.identity.chunk_tokens,
            )
            descriptors = []
            for index, part in enumerate(parts):
                chunk = ContextChunk(
                    index * fixture.identity.chunk_tokens,
                    (index + 1) * fixture.identity.chunk_tokens,
                    {
                        StateRecord.LOGICAL_POSITIONS: pack_positions(
                            range(
                                index * fixture.identity.chunk_tokens,
                                (index + 1) * fixture.identity.chunk_tokens,
                            )
                        ),
                        StateRecord.TARGET_CKV: bytes(part),
                    },
                )
                encoded = cache_manifest._encode_chunk(chunk)
                digest = hashlib.sha256(encoded).hexdigest()
                cache_manifest._publish_immutable(
                    root / "chunks" / f"{digest}.spcc",
                    encoded,
                )
                descriptors.append(
                    {
                        "sha256": digest,
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

            lookup = store.lookup(fixture.identity, fixture.result_digest)

            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(
                store.restore_page_snapshot(
                    lookup,
                    layout=fixture.layout,
                    result_block_counts=(fixture.result_blocks,),
                    result_boundary_tokens=fixture.result_tokens,
                ),
                fixture.result_snapshot,
            )


if __name__ == "__main__":
    unittest.main()
