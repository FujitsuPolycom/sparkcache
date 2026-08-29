from __future__ import annotations

import dataclasses
import gc
import hashlib
import json
import multiprocessing
import os
import tempfile
import threading
import time
import unittest
import weakref
from pathlib import Path
from typing import Any
from unittest import mock

import sparkcache.persistent_context_cache.cache_manifest as cache_manifest
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    CapacityPolicy,
    CommitConflict,
    ContextChunk,
    IncompleteEntry,
    ManifestStore,
    StateRecord,
)


def _identity() -> CacheIdentity:
    return CacheIdentity(
        target_checkpoint="1" * 64,
        draft_checkpoint="2" * 64,
        quantization_layout="nvfp4-ds-mla-v1",
        rope_layout="glm52-bf16-rope-v1",
        tp_degree=4,
        dcp_degree=1,
        chunk_tokens=256,
    )


def _tail_identity(**changes: Any) -> CacheIdentity:
    return dataclasses.replace(
        _identity(),
        publication_schema="tail-cow-v1",
        **changes,
    )


def _chunk(start: int = 0, end: int = 256) -> ContextChunk:
    return ContextChunk(
        logical_start=start,
        logical_end=end,
        records={
            StateRecord.TARGET_CKV: b"target-ckv-and-scales",
            StateRecord.SPARSE_INDEXER: b"sparse-indexer",
            StateRecord.MTP_DRAFT_KV: b"mtp-draft-kv",
            StateRecord.BOUNDARY_HIDDEN: b"boundary-hidden",
            StateRecord.LOGICAL_POSITIONS: b"logical-positions",
        },
    )


def _variant_chunk(label: bytes) -> ContextChunk:
    records = dict(_chunk().records)
    records[StateRecord.TARGET_CKV] = b"target-" + label
    return ContextChunk(0, 256, records)


def _hold_open_transaction(
    root: str,
    ready: Any,
    release: Any,
) -> None:
    store = ManifestStore(root)
    transaction = store.begin_context(
        identity=_identity(),
        context_digest=hashlib.sha256(b"subprocess-open").hexdigest(),
    )
    transaction.append_chunk(_chunk())
    ready.set()
    release.wait(timeout=10)
    transaction.abort()


def _clear_once_in_subprocess(
    root: str,
    token: str,
    start: Any,
    results: Any,
) -> None:
    start.wait(timeout=10)
    try:
        results.put(("ok", ManifestStore(root).clear_once(token)))
    except BaseException as error:
        results.put(("error", type(error).__name__, str(error)))


class ManifestStoreTests(unittest.TestCase):
    def test_tail_publication_uses_a_distinct_cache_namespace(self) -> None:
        snapshot_identity = _identity()
        tail_identity = _tail_identity()

        self.assertNotEqual(snapshot_identity.storage_key, tail_identity.storage_key)
        self.assertNotIn("publication_schema", snapshot_identity.to_wire())
        self.assertEqual(
            tail_identity.to_wire()["publication_schema"],
            "tail-cow-v1",
        )

    def test_extension_publishes_only_tail_payload_and_restores_full_context(
        self,
    ) -> None:
        from sparkcache.spark_context_cache_codec import context_prefix_digest

        identity = _tail_identity()
        token_ids = tuple(range(512))
        salt = "tail-publication-test"
        base_digest = context_prefix_digest(token_ids, salt, token_count=256)
        result_digest = context_prefix_digest(token_ids, salt, token_count=512)
        base = _variant_chunk(b"base")
        tail = ContextChunk(
            256,
            512,
            dict(_variant_chunk(b"tail").records),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=identity,
                context_digest=base_digest,
                chunks=[base],
                span_tokens=256,
            )
            base_chunk = next((root / "chunks").glob("*.spcc"))

            receipt = store.commit_extension(
                identity=identity,
                base_context_digest=base_digest,
                token_ids=token_ids,
                identity_salt=salt,
                tail_chunks=[tail],
            )

            self.assertEqual(receipt.committed_tokens, 512)
            self.assertEqual(len(tuple((root / "chunks").glob("*.spcc"))), 2)
            self.assertTrue(base_chunk.exists())
            encoded = (
                root / "manifests" / identity.storage_key / f"{result_digest}.json"
            ).read_bytes()
            manifest = json.loads(encoded)
            self.assertEqual(manifest["schema"], "sparkcache-tail-manifest/v1")
            self.assertEqual(len(manifest["tail_chunks"]), 1)
            lookup = store.lookup(identity, result_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (base, tail))

    def test_extension_replaces_partial_terminal_without_mutating_base(self) -> None:
        from sparkcache.spark_context_cache_codec import context_prefix_digest

        identity = _tail_identity()
        token_ids = tuple(range(512))
        salt = "partial-terminal-test"
        base_digest = context_prefix_digest(token_ids, salt, token_count=300)
        result_digest = context_prefix_digest(token_ids, salt, token_count=512)
        first = _variant_chunk(b"first")
        partial = ContextChunk(256, 300, dict(_variant_chunk(b"partial").records))
        replacement = ContextChunk(
            256,
            512,
            dict(_variant_chunk(b"replacement").records),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            store.commit(
                identity=identity,
                context_digest=base_digest,
                chunks=[first, partial],
                span_tokens=300,
            )

            store.commit_extension(
                identity=identity,
                base_context_digest=base_digest,
                token_ids=token_ids,
                identity_salt=salt,
                tail_chunks=[replacement],
            )

            base = store.lookup(identity, base_digest)
            extended = store.lookup(identity, result_digest)
            self.assertEqual(store.restore(base), (first, partial))
            self.assertEqual(store.restore(extended), (first, replacement))

    def test_extension_gc_preserves_old_alias_and_shared_base_chunks(self) -> None:
        from sparkcache.spark_context_cache_codec import context_prefix_digest

        identity = _tail_identity()
        token_ids = tuple(range(768))
        salt = "tail-reference-gc-test"
        base_digest = context_prefix_digest(token_ids, salt, token_count=512)
        alias_digest = context_prefix_digest(token_ids, salt, token_count=256)
        result_digest = context_prefix_digest(token_ids, salt, token_count=768)
        chunks = (
            _variant_chunk(b"base-zero"),
            ContextChunk(256, 512, dict(_variant_chunk(b"base-one").records)),
        )
        tail = ContextChunk(512, 768, dict(_variant_chunk(b"tail-two").records))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=identity,
                context_digest=base_digest,
                chunks=chunks,
                span_tokens=512,
            )
            store.publish_prefix_aliases(
                identity=identity,
                source_context_digest=base_digest,
                token_ids=token_ids,
                identity_salt=salt,
                prefix_tokens=(256,),
                storage_mode="per_token_rows",
            )
            store.commit_extension(
                identity=identity,
                base_context_digest=base_digest,
                token_ids=token_ids,
                identity_salt=salt,
                tail_chunks=(tail,),
            )
            (root / "manifests" / identity.storage_key / f"{base_digest}.json").unlink()

            report = store.maintain(
                CapacityPolicy(max_bytes=10**9, low_watermark_bytes=10**9)
            )

            self.assertEqual(report.orphan_chunks_deleted, 0)
            alias = store.lookup(
                identity,
                alias_digest,
                storage_mode="per_token_rows",
            )
            extension = store.lookup(identity, result_digest)
            self.assertEqual(store.restore(alias), (chunks[0],))
            self.assertEqual(store.restore(extension), (*chunks, tail))

    def test_extension_rejects_corrupt_descriptor_chain_and_wrong_rank(self) -> None:
        from sparkcache.spark_context_cache_codec import context_prefix_digest

        identity = _tail_identity(tp_shard_rank=0)
        other_rank = dataclasses.replace(identity, tp_shard_rank=1)
        token_ids = tuple(range(512))
        salt = "tail-integrity-test"
        base_digest = context_prefix_digest(token_ids, salt, token_count=256)
        result_digest = context_prefix_digest(token_ids, salt, token_count=512)
        tail = ContextChunk(256, 512, dict(_variant_chunk(b"tail").records))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=identity,
                context_digest=base_digest,
                chunks=[_variant_chunk(b"base")],
            )
            store.commit_extension(
                identity=identity,
                base_context_digest=base_digest,
                token_ids=token_ids,
                identity_salt=salt,
                tail_chunks=(tail,),
            )
            manifest_path = (
                root / "manifests" / identity.storage_key / f"{result_digest}.json"
            )
            manifest = json.loads(manifest_path.read_bytes())
            segment = (
                root
                / "prefix-index"
                / identity.storage_key
                / f"{manifest['base_tail_segment_sha256']}.spix"
            )
            segment.write_bytes(segment.read_bytes() + b"corrupt")

            damaged = store.lookup(identity, result_digest)
            self.assertFalse(damaged.is_hit)
            self.assertEqual(damaged.reason, "corrupt")

            wrong_root = (
                root / "manifests" / other_rank.storage_key / manifest_path.name
            )
            wrong_root.parent.mkdir(parents=True)
            wrong_root.write_bytes(manifest_path.read_bytes())
            wrong_rank = store.lookup(other_rank, result_digest)
            self.assertFalse(wrong_rank.is_hit)
            self.assertEqual(wrong_rank.reason, "incompatible")

    def test_extension_derives_and_checks_token_commitments_internally(self) -> None:
        from sparkcache.spark_context_cache_codec import context_prefix_digest

        identity = _tail_identity()
        original = tuple(range(512))
        changed = (*original[:255], 9999, *original[256:])
        salt = "tail-token-commitment-test"
        base_digest = context_prefix_digest(original, salt, token_count=256)
        tail = ContextChunk(256, 512, dict(_variant_chunk(b"tail").records))
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            store.commit(
                identity=identity,
                context_digest=base_digest,
                chunks=[_variant_chunk(b"base")],
            )

            with self.assertRaisesRegex(CommitConflict, "token sequence"):
                store.commit_extension(
                    identity=identity,
                    base_context_digest=base_digest,
                    token_ids=changed,
                    identity_salt=salt,
                    tail_chunks=(tail,),
                )

    def test_clear_once_preserves_markers_and_non_cache_siblings(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "cache" / "rank-0"
            model = parent / "models" / "checkpoint.bin"
            jit = parent / "jit" / "kernel.bin"
            model.parent.mkdir()
            jit.parent.mkdir()
            model.write_bytes(b"model")
            jit.write_bytes(b"jit")
            for name in (
                "chunks",
                "manifests",
                "prefix-aliases",
                "prefix-index",
            ):
                path = root / name / "nested" / "payload"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(name.encode())
            operator_file = root / "operator-note.txt"
            operator_file.write_text("preserve", encoding="utf-8")
            store = ManifestStore(root)

            self.assertTrue(store.clear_once("rollout-2026-08-29"))

            for name in (
                "chunks",
                "manifests",
                "prefix-aliases",
                "prefix-index",
            ):
                self.assertFalse((root / name).exists())
            self.assertEqual(model.read_bytes(), b"model")
            self.assertEqual(jit.read_bytes(), b"jit")
            self.assertEqual(operator_file.read_text(encoding="utf-8"), "preserve")
            marker_directory = root / ".sparkcache-clear-once"
            markers = tuple(marker_directory.glob("*.json"))
            self.assertEqual(len(markers), 1)
            marker = json.loads(markers[0].read_bytes())
            self.assertEqual(marker["schema"], "sparkcache-clear-once/v1")
            self.assertNotIn("rollout-2026-08-29", markers[0].read_text())

            later_chunk = root / "chunks" / "later.spcc"
            later_chunk.parent.mkdir()
            later_chunk.write_bytes(b"later")
            self.assertFalse(store.clear_once("rollout-2026-08-29"))
            self.assertEqual(later_chunk.read_bytes(), b"later")

            self.assertTrue(store.clear_once("rollout-2026-08-30"))
            self.assertFalse(later_chunk.exists())
            self.assertEqual(len(tuple(marker_directory.glob("*.json"))), 2)

    def test_clear_once_records_nothing_after_failed_removal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache" / "rank-0"
            chunk = root / "chunks" / "payload.spcc"
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(b"payload")
            store = ManifestStore(root)
            with mock.patch.object(
                cache_manifest,
                "_remove_tree_without_following_links",
                side_effect=OSError("synthetic removal failure"),
            ):
                with self.assertRaisesRegex(OSError, "synthetic removal failure"):
                    store.clear_once("failed-clear")

            marker_directory = root / ".sparkcache-clear-once"
            self.assertFalse(tuple(marker_directory.glob("*.json")))
            self.assertEqual(chunk.read_bytes(), b"payload")
            self.assertTrue(store.clear_once("failed-clear"))

    def test_clear_once_unlinks_owned_symlink_without_following_it(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "cache" / "rank-0"
            target = parent / "models"
            target.mkdir()
            model = target / "checkpoint.bin"
            model.write_bytes(b"model")
            root.mkdir(parents=True)
            try:
                (root / "chunks").symlink_to(target, target_is_directory=True)
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")

            self.assertTrue(ManifestStore(root).clear_once("symlink-child"))

            self.assertFalse((root / "chunks").exists())
            self.assertEqual(model.read_bytes(), b"model")

    def test_clear_once_rejects_symlinked_marker_directory_before_deletion(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            root = parent / "cache" / "rank-0"
            chunk = root / "chunks" / "payload.spcc"
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(b"payload")
            target = parent / "marker-target"
            target.mkdir()
            try:
                (root / ".sparkcache-clear-once").symlink_to(
                    target,
                    target_is_directory=True,
                )
            except OSError as error:
                self.skipTest(f"directory symlinks unavailable: {error}")

            with self.assertRaisesRegex(OSError, "marker directory is a symlink"):
                ManifestStore(root).clear_once("symlink-marker")

            self.assertEqual(chunk.read_bytes(), b"payload")
            self.assertFalse(tuple(target.iterdir()))

    def test_clear_once_same_process_concurrency_removes_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache" / "rank-0"
            chunk = root / "chunks" / "payload.spcc"
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(b"payload")
            barrier = threading.Barrier(8)
            results: list[bool] = []
            errors: list[BaseException] = []

            def clear() -> None:
                try:
                    barrier.wait(timeout=5)
                    results.append(ManifestStore(root).clear_once("concurrent"))
                except BaseException as error:
                    errors.append(error)

            threads = [threading.Thread(target=clear) for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=10)

            self.assertFalse(errors)
            self.assertEqual(results.count(True), 1)
            self.assertEqual(results.count(False), 7)

    @unittest.skipIf(cache_manifest._fcntl is None, "requires POSIX flock")
    def test_clear_once_cross_process_concurrency_removes_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache" / "rank-0"
            chunk = root / "chunks" / "payload.spcc"
            chunk.parent.mkdir(parents=True)
            chunk.write_bytes(b"payload")
            context = multiprocessing.get_context("spawn")
            start = context.Event()
            results = context.Queue()
            processes = [
                context.Process(
                    target=_clear_once_in_subprocess,
                    args=(str(root), "multiprocess", start, results),
                )
                for _ in range(2)
            ]
            for process in processes:
                process.start()
            start.set()
            outcomes = [results.get(timeout=15) for _ in processes]
            for process in processes:
                process.join(timeout=15)
                self.assertEqual(process.exitcode, 0)

            self.assertEqual(sorted(outcomes), [("ok", False), ("ok", True)])

    def test_clear_once_lock_wait_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache" / "rank-0"
            guard = cache_manifest._RootGuard(
                root,
                shared=False,
                blocking=True,
            )
            guard.__enter__()
            try:
                started = time.monotonic()
                with self.assertRaises(BlockingIOError):
                    ManifestStore(root).clear_once(
                        "bounded-wait",
                        lock_timeout_seconds=0.01,
                    )
                self.assertLess(time.monotonic() - started, 1.0)
            finally:
                guard.__exit__(None, None, None)

    def test_root_lock_timeout_is_one_budget_across_both_lock_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "cache" / "rank-0"
            guard = cache_manifest._RootGuard(
                root,
                shared=False,
                blocking=True,
                timeout_seconds=30.0,
            )
            clock = [100.0]
            process_lock = mock.Mock()

            def acquire(**kwargs: Any) -> bool:
                self.assertEqual(kwargs["timeout_seconds"], 30.0)
                # Simulate process-local contention consuming more than the
                # complete acquisition deadline before returning ownership.
                clock[0] = 131.0
                return True

            process_lock.acquire.side_effect = acquire
            guard._process_lock = process_lock
            file_lock = mock.Mock()
            file_lock.LOCK_SH = 1
            file_lock.LOCK_EX = 2
            file_lock.LOCK_NB = 4

            with (
                mock.patch.object(
                    cache_manifest.time,
                    "monotonic",
                    side_effect=lambda: clock[0],
                ),
                mock.patch.object(cache_manifest.time, "sleep") as sleep,
                mock.patch.object(cache_manifest, "_fcntl", file_lock),
            ):
                with self.assertRaisesRegex(
                    BlockingIOError,
                    "manifest root is busy",
                ):
                    guard.__enter__()

            file_lock.flock.assert_not_called()
            sleep.assert_not_called()
            process_lock.release.assert_called_once_with(shared=False)

    def test_capacity_policy_rejects_invalid_geometry(self) -> None:
        for kwargs in (
            {"max_bytes": True},
            {"max_bytes": -1},
            {"max_bytes": 0, "low_watermark_bytes": 1},
            {"max_bytes": 10, "low_watermark_bytes": 0},
            {"max_bytes": 10, "low_watermark_bytes": 11},
            {"ttl_seconds": -1},
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                CapacityPolicy(**kwargs)

    def test_capacity_counts_unique_chunks_and_evicts_lru_to_low_watermark(
        self,
    ) -> None:
        identity = _identity()
        digests = [
            hashlib.sha256(f"entry-{index}".encode()).hexdigest() for index in range(3)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            for index, digest in enumerate(digests):
                store.commit(
                    identity=identity,
                    context_digest=digest,
                    chunks=[_variant_chunk(str(index).encode())],
                )
                path = root / "manifests" / identity.storage_key / f"{digest}.json"
                mtime_ns = (index + 1) * 10**9
                os.utime(path, ns=(mtime_ns, mtime_ns))
            chunk_bytes = sum(
                cache_manifest._allocated_bytes(path.stat())
                for path in (root / "chunks").iterdir()
            )
            newest_manifest = (
                root / "manifests" / identity.storage_key / f"{digests[-1]}.json"
            )
            keep_bytes = chunk_bytes // 3 + cache_manifest._allocated_bytes(
                newest_manifest.stat()
            )
            report = store.maintain(
                CapacityPolicy(
                    max_bytes=keep_bytes + 1,
                    low_watermark_bytes=keep_bytes,
                ),
                now_ns=10 * 10**9,
            )

            self.assertTrue(report.capacity_satisfied)
            self.assertEqual(report.manifests_evicted, 2)
            self.assertFalse(store.lookup(identity, digests[0]).is_hit)
            self.assertFalse(store.lookup(identity, digests[1]).is_hit)
            newest = store.lookup(identity, digests[2])
            self.assertTrue(newest.is_hit, newest.reason)
            self.assertEqual(store.restore(newest), (_variant_chunk(b"2"),))
            self.assertLessEqual(report.bytes_after, keep_bytes + 1)

    def test_verified_touch_refreshes_ttl_recency(self) -> None:
        identity = _identity()
        old_digest = hashlib.sha256(b"ttl-old").hexdigest()
        touched_digest = hashlib.sha256(b"ttl-touched").hexdigest()
        second = 10**9
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            for digest, label in ((old_digest, b"old"), (touched_digest, b"new")):
                store.commit(
                    identity=identity,
                    context_digest=digest,
                    chunks=[_variant_chunk(label)],
                )
                path = root / "manifests" / identity.storage_key / f"{digest}.json"
                os.utime(path, ns=(second, second))
            self.assertTrue(
                store.touch(
                    identity,
                    touched_digest,
                    now_ns=20 * second,
                    minimum_interval_seconds=0,
                )
            )

            report = store.maintain(
                CapacityPolicy(ttl_seconds=10),
                now_ns=25 * second,
            )

            self.assertEqual(report.manifests_evicted, 1)
            self.assertFalse(store.lookup(identity, old_digest).is_hit)
            self.assertTrue(store.lookup(identity, touched_digest).is_hit)

    def test_pressure_reclaims_to_low_after_orphan_cleanup_crosses_high(
        self,
    ) -> None:
        identity = _identity()
        digests = [
            hashlib.sha256(f"hysteresis-{index}".encode()).hexdigest()
            for index in range(2)
        ]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            for index, digest in enumerate(digests):
                store.commit(
                    identity=identity,
                    context_digest=digest,
                    chunks=[_variant_chunk(f"live-{index}".encode())],
                )
                manifest = root / "manifests" / identity.storage_key / f"{digest}.json"
                mtime_ns = (index + 1) * 10**9
                os.utime(manifest, ns=(mtime_ns, mtime_ns))

            live_bytes = sum(
                cache_manifest._allocated_bytes(path.stat())
                for directory in (root / "manifests", root / "chunks")
                for path in directory.glob("**/*")
                if path.is_file()
            )
            transaction = store.begin_context(
                identity=identity,
                context_digest=hashlib.sha256(b"hysteresis-orphan").hexdigest(),
            )
            transaction.append_chunk(_variant_chunk(b"orphan"))
            transaction.abort()
            bytes_with_orphan = sum(
                cache_manifest._allocated_bytes(path.stat())
                for directory in (root / "manifests", root / "chunks")
                for path in directory.glob("**/*")
                if path.is_file()
            )
            orphan_bytes = bytes_with_orphan - live_bytes
            self.assertGreater(orphan_bytes, 0)

            report = store.maintain(
                CapacityPolicy(
                    max_bytes=live_bytes + orphan_bytes // 2,
                    low_watermark_bytes=live_bytes - 1,
                )
            )

            self.assertEqual(report.bytes_before, bytes_with_orphan)
            self.assertEqual(report.orphan_chunks_deleted, 1)
            self.assertEqual(report.manifests_evicted, 1)
            self.assertLessEqual(report.bytes_after, live_bytes - 1)
            self.assertFalse(store.lookup(identity, digests[0]).is_hit)
            self.assertTrue(store.lookup(identity, digests[1]).is_hit)

    def test_eviction_preserves_chunks_shared_across_namespaces(self) -> None:
        first_identity = _identity()
        second_identity = dataclasses.replace(
            first_identity,
            quantization_layout="different-layout",
        )
        first_digest = hashlib.sha256(b"shared-first").hexdigest()
        second_digest = hashlib.sha256(b"shared-second").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            for identity, digest in (
                (first_identity, first_digest),
                (second_identity, second_digest),
            ):
                store.commit(
                    identity=identity,
                    context_digest=digest,
                    chunks=[_chunk()],
                )
            manifests = sorted((root / "manifests").glob("*/*.json"))
            os.utime(manifests[0], ns=(1, 1))
            os.utime(manifests[1], ns=(2, 2))
            chunk_path = next((root / "chunks").glob("*.spcc"))
            one_entry_bytes = cache_manifest._allocated_bytes(
                chunk_path.stat()
            ) + cache_manifest._allocated_bytes(manifests[1].stat())

            first_report = store.maintain(
                CapacityPolicy(
                    max_bytes=one_entry_bytes + 1,
                    low_watermark_bytes=one_entry_bytes,
                ),
                now_ns=3,
            )

            self.assertEqual(first_report.manifests_evicted, 1)
            self.assertTrue(chunk_path.exists())
            survivors = [
                (identity, digest)
                for identity, digest in (
                    (first_identity, first_digest),
                    (second_identity, second_digest),
                )
                if store.lookup(identity, digest).is_hit
            ]
            self.assertEqual(len(survivors), 1)
            lookup = store.lookup(*survivors[0])
            self.assertEqual(store.restore(lookup), (_chunk(),))

            second_report = store.maintain(
                CapacityPolicy(max_bytes=1, low_watermark_bytes=1),
                now_ns=4,
            )
            self.assertEqual(second_report.manifests_evicted, 1)
            self.assertFalse(chunk_path.exists())

    def test_maintenance_skips_open_transaction_then_collects_orphan(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=hashlib.sha256(b"open-orphan").hexdigest(),
            )
            transaction.append_chunk(_chunk())
            chunk_path = next((root / "chunks").glob("*.spcc"))

            busy = store.maintain(CapacityPolicy(max_bytes=1, low_watermark_bytes=1))
            self.assertTrue(busy.skipped_busy)
            self.assertTrue(chunk_path.exists())

            transaction.abort()
            manifest_directory = root / "manifests" / _identity().storage_key
            manifest_directory.mkdir(parents=True)
            temporary_manifest = manifest_directory / ".manifest.json.writing-crash"
            temporary_manifest.write_bytes(b"partial")
            collected = store.maintain(
                CapacityPolicy(max_bytes=1, low_watermark_bytes=1)
            )
            self.assertEqual(collected.orphan_chunks_deleted, 1)
            self.assertFalse(chunk_path.exists())
            self.assertFalse(temporary_manifest.exists())

    @unittest.skipIf(os.name == "nt", "POSIX flock behavior")
    def test_posix_maintenance_skips_transaction_in_another_process(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            context = multiprocessing.get_context("fork")
            ready = context.Event()
            release = context.Event()
            child = context.Process(
                target=_hold_open_transaction,
                args=(directory, ready, release),
            )
            child.start()
            self.assertTrue(ready.wait(timeout=10))
            try:
                busy = ManifestStore(directory).maintain(
                    CapacityPolicy(max_bytes=1, low_watermark_bytes=1)
                )
                self.assertTrue(busy.skipped_busy)
            finally:
                release.set()
                child.join(timeout=10)
            self.assertEqual(child.exitcode, 0)
            collected = ManifestStore(directory).maintain(
                CapacityPolicy(max_bytes=1, low_watermark_bytes=1)
            )
            self.assertEqual(collected.orphan_chunks_deleted, 1)

    def test_maintenance_never_reads_or_hashes_chunk_payloads(self) -> None:
        digest = hashlib.sha256(b"metadata-only-maintenance").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=digest,
                chunks=[_chunk()],
            )
            original_read = Path.read_bytes

            def reject_chunk_read(path: Path) -> bytes:
                if path.suffix == ".spcc":
                    raise AssertionError("maintenance read a chunk payload")
                return original_read(path)

            with mock.patch.object(Path, "read_bytes", reject_chunk_read):
                report = store.maintain(
                    CapacityPolicy(max_bytes=1, low_watermark_bytes=1)
                )
            self.assertEqual(report.manifests_evicted, 1)

    def test_manifest_fsync_failure_never_deletes_referenced_chunk(self) -> None:
        digest = hashlib.sha256(b"eviction-fsync-failure").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=digest,
                chunks=[_chunk()],
            )
            chunk_path = next((root / "chunks").glob("*.spcc"))
            manifest_directory = root / "manifests" / _identity().storage_key

            def fail_manifest_fsync(path: Path) -> None:
                if path == manifest_directory:
                    raise OSError("simulated eviction barrier failure")

            with (
                mock.patch.object(
                    cache_manifest,
                    "_fsync_directory",
                    side_effect=fail_manifest_fsync,
                ),
                self.assertRaisesRegex(OSError, "eviction barrier failure"),
            ):
                store.maintain(CapacityPolicy(max_bytes=1, low_watermark_bytes=1))

            self.assertTrue(chunk_path.exists())
            retry = store.maintain(CapacityPolicy(max_bytes=1, low_watermark_bytes=1))
            self.assertEqual(retry.orphan_chunks_deleted, 1)
            self.assertFalse(chunk_path.exists())

    def test_streaming_macro_batch_uses_one_chunk_directory_barrier(self) -> None:
        context_digest = hashlib.sha256(b"streamed-macro-batch").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            chunk_directory = root / "chunks"
            chunk_directory.mkdir()
            store = ManifestStore(root)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=768,
            )
            barriers: list[Path] = []

            with mock.patch.object(
                cache_manifest,
                "_fsync_directory",
                side_effect=lambda path: barriers.append(Path(path)),
            ):
                receipts = transaction.append_chunks(
                    (
                        _chunk(0, 256),
                        _chunk(256, 512),
                        _chunk(512, 768),
                    )
                )

            self.assertEqual(len(receipts), 3)
            self.assertEqual(barriers, [chunk_directory])

            transaction.commit_manifest()
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(
                store.restore(lookup),
                (
                    _chunk(0, 256),
                    _chunk(256, 512),
                    _chunk(512, 768),
                ),
            )

    def test_compatibility_commit_publishes_bounded_macro_batches(self) -> None:
        context_digest = hashlib.sha256(b"compatibility-macro-batch").hexdigest()
        chunks = tuple(_chunk(index * 256, (index + 1) * 256) for index in range(17))

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            batch_sizes: list[int] = []
            original_publish = cache_manifest._publish_immutable_batch

            def record_batch(objects: list[tuple[Path, bytes]]) -> None:
                batch_sizes.append(len(objects))
                original_publish(objects)

            with mock.patch.object(
                cache_manifest,
                "_publish_immutable_batch",
                side_effect=record_batch,
            ):
                store.commit(
                    identity=_identity(),
                    context_digest=context_digest,
                    chunks=chunks,
                    span_tokens=17 * 256,
                )

            self.assertEqual(batch_sizes, [8, 8, 1])
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), chunks)

    def test_streaming_macro_batch_retry_is_idempotent(self) -> None:
        context_digest = hashlib.sha256(b"macro-batch-retry").hexdigest()
        chunks = (_chunk(0, 256), _chunk(256, 512), _chunk(512, 768))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=768,
            )

            first = transaction.append_chunks(chunks)
            second = transaction.append_chunks(chunks)
            self.assertEqual(second, first)

            transaction.commit_manifest()
            self.assertEqual(transaction.append_chunks(chunks), first)
            self.assertEqual(len(tuple((root / "chunks").glob("*.spcc"))), 3)
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), chunks)

    def test_streaming_macro_batch_conflict_fails_before_partial_append(self) -> None:
        context_digest = hashlib.sha256(b"macro-batch-conflict").hexdigest()
        first = _chunk(0, 256)
        second = _chunk(256, 512)
        replacement_records = dict(second.records)
        replacement_records[StateRecord.TARGET_CKV] = b"different-target"
        conflicting_second = ContextChunk(256, 512, replacement_records)

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=512,
            )
            transaction.append_chunks((first, second))

            with self.assertRaisesRegex(CommitConflict, "logical range"):
                transaction.append_chunks((first, conflicting_second))

            transaction.commit_manifest()
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (first, second))

    def test_streaming_transaction_keeps_partial_publication_invisible(self) -> None:
        context_digest = hashlib.sha256(b"partial-stream").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            transaction = store.begin(
                identity=_identity(),
                context_digest=context_digest,
            )
            transaction.append_chunk(_chunk())

            self.assertEqual(len(tuple((root / "chunks").glob("*.spcc"))), 1)
            self.assertFalse(store.lookup(_identity(), context_digest).is_hit)

            transaction.abort()
            self.assertFalse(store.lookup(_identity(), context_digest).is_hit)

    def test_streaming_transaction_commits_without_retaining_chunk_payloads(
        self,
    ) -> None:
        context_digest = hashlib.sha256(b"streamed-two-chunks").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
            )
            first = _chunk(0, 256)
            first_reference = weakref.ref(first)
            first_receipt = transaction.append_chunk(first)
            del first
            gc.collect()

            self.assertIsNone(first_reference())
            self.assertEqual(first_receipt.logical_start, 0)
            self.assertEqual(first_receipt.logical_end, 256)

            transaction.append_chunk(_chunk(256, 384))
            receipt = transaction.commit_manifest()
            self.assertEqual(receipt.committed_tokens, 384)

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(
                [
                    (chunk.logical_start, chunk.logical_end)
                    for chunk in store.restore(lookup)
                ],
                [(0, 256), (256, 384)],
            )

    def test_streaming_duplicate_publication_is_idempotent(self) -> None:
        context_digest = hashlib.sha256(b"idempotent-stream").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            receipts = []
            for _ in range(2):
                transaction = store.begin_context(
                    identity=_identity(),
                    context_digest=context_digest,
                )
                transaction.append_chunk(_chunk())
                receipts.append(transaction.commit_manifest())

            self.assertEqual(receipts[0], receipts[1])
            self.assertEqual(
                len(tuple((Path(directory) / "chunks").glob("*.spcc"))),
                1,
            )
            self.assertEqual(
                len(tuple((Path(directory) / "manifests").rglob("*.json"))),
                1,
            )

    def test_streaming_concurrent_identical_append_is_serialized_and_idempotent(
        self,
    ) -> None:
        context_digest = hashlib.sha256(b"concurrent-identical-append").hexdigest()
        entered_publish = threading.Event()
        release_publish = threading.Event()
        second_started = threading.Event()
        original_publish = cache_manifest._publish_immutable_batch
        chunk_publish_calls = 0
        receipts = []
        errors = []

        def blocking_publish(objects: tuple[tuple[Path, bytes], ...]) -> None:
            nonlocal chunk_publish_calls
            if objects:
                chunk_publish_calls += 1
                entered_publish.set()
                if not release_publish.wait(timeout=5):
                    raise TimeoutError("test did not release chunk publication")
            original_publish(objects)

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=256,
            )

            def append(*, announce: bool = False) -> None:
                if announce:
                    second_started.set()
                try:
                    receipts.append(transaction.append_chunk(_chunk()))
                except BaseException as error:  # pragma: no cover - assertion data
                    errors.append(error)

            with mock.patch.object(
                cache_manifest,
                "_publish_immutable_batch",
                blocking_publish,
            ):
                first = threading.Thread(target=append)
                second = threading.Thread(
                    target=append,
                    kwargs={"announce": True},
                )
                first.start()
                self.assertTrue(entered_publish.wait(timeout=5))
                second.start()
                self.assertTrue(second_started.wait(timeout=5))
                release_publish.set()
                first.join(timeout=5)
                second.join(timeout=5)

            self.assertFalse(first.is_alive())
            self.assertFalse(second.is_alive())
            self.assertEqual(errors, [])
            self.assertEqual(len(receipts), 2)
            self.assertEqual(receipts[0], receipts[1])
            self.assertEqual(chunk_publish_calls, 1)

            committed = transaction.commit_manifest()
            self.assertEqual(transaction.commit_manifest(), committed)
            self.assertEqual(transaction.append_chunk(_chunk()), receipts[0])
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (_chunk(),))

    def test_streaming_conflicting_retry_for_same_range_fails_closed(self) -> None:
        context_digest = hashlib.sha256(b"conflicting-range-retry").hexdigest()
        replacement_records = dict(_chunk().records)
        replacement_records[StateRecord.TARGET_CKV] = b"different-target"
        replacement = ContextChunk(0, 256, replacement_records)

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=256,
            )
            transaction.append_chunk(_chunk())
            with self.assertRaisesRegex(
                CommitConflict,
                "logical range",
            ):
                transaction.append_chunk(replacement)

            transaction.commit_manifest()
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (_chunk(),))

    def test_streaming_conflict_cannot_replace_committed_manifest(self) -> None:
        context_digest = hashlib.sha256(b"conflicting-stream").hexdigest()
        replacement_records = dict(_chunk().records)
        replacement_records[StateRecord.TARGET_CKV] = b"different-target"
        replacement = ContextChunk(
            logical_start=0,
            logical_end=256,
            records=replacement_records,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            original = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
            )
            original.append_chunk(_chunk())
            original_receipt = original.commit_manifest()

            conflict = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
            )
            conflict.append_chunk(replacement)
            with self.assertRaises(CommitConflict):
                conflict.commit_manifest()
            conflict.abort()

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.manifest_digest, original_receipt.manifest_digest)
            self.assertEqual(store.restore(lookup), (_chunk(),))

    def test_streaming_transaction_rejects_invalid_lifecycle_and_geometry(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)

            empty = store.begin_context(
                identity=_identity(),
                context_digest="a" * 64,
            )
            with self.assertRaisesRegex(ValueError, "at least one context chunk"):
                empty.commit_manifest()
            empty.abort()
            with self.assertRaisesRegex(RuntimeError, "aborted"):
                empty.append_chunk(_chunk())

            partial = store.begin_context(
                identity=_identity(),
                context_digest="b" * 64,
            )
            partial.append_chunk(_chunk(0, 128))
            with self.assertRaisesRegex(
                ValueError,
                "only the final context chunk may be partial",
            ):
                partial.append_chunk(_chunk(128, 256))
            partial.abort()

            committed = store.begin_context(
                identity=_identity(),
                context_digest="c" * 64,
            )
            committed.append_chunk(_chunk())
            committed_receipt = committed.commit_manifest()
            self.assertEqual(committed.commit_manifest(), committed_receipt)
            with self.assertRaisesRegex(RuntimeError, "committed"):
                committed.abort()

    def test_declared_span_prevents_partial_manifest_visibility(self) -> None:
        context_digest = hashlib.sha256(b"declared-span").hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            transaction = store.begin_context(
                identity=_identity(),
                context_digest=context_digest,
                span_tokens=512,
            )
            transaction.append_chunk(_chunk(0, 256))

            with self.assertRaisesRegex(IncompleteEntry, "declared context span"):
                transaction.commit_manifest()
            self.assertFalse(store.lookup(_identity(), context_digest).is_hit)

            transaction.append_chunk(_chunk(256, 512))
            receipt = transaction.commit_manifest()
            self.assertEqual(receipt.committed_tokens, 512)
            self.assertTrue(store.lookup(_identity(), context_digest).is_hit)

    def test_checkpoint_identity_requires_content_digests(self) -> None:
        for field, value in (
            ("target_checkpoint", "mutable/model/path"),
            ("draft_checkpoint", "latest"),
            ("target_checkpoint", "A" * 64),
            ("draft_checkpoint", None),
        ):
            with self.subTest(field=field, value=value):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{field} must be a 64-character lowercase SHA-256",
                ):
                    dataclasses.replace(_identity(), **{field: value})

    def test_successful_commit_durably_publishes_chunks_before_manifest(
        self,
    ) -> None:
        events: list[tuple[str, Path]] = []
        original_link = cache_manifest.os.link

        def recording_link(source: Path, destination: Path) -> None:
            original_link(source, destination)
            events.append(("link", Path(destination)))

        def recording_directory_fsync(path: Path) -> None:
            events.append(("fsync-directory", Path(path)))

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            with (
                mock.patch.object(cache_manifest.os, "link", recording_link),
                mock.patch.object(
                    cache_manifest,
                    "_fsync_directory",
                    recording_directory_fsync,
                ),
            ):
                store.commit(
                    identity=_identity(),
                    context_digest="0" * 64,
                    chunks=(_chunk(),),
                )

        chunk_link = next(
            index
            for index, event in enumerate(events)
            if event[0] == "link" and event[1].suffix == ".spcc"
        )
        manifest_link = next(
            index
            for index, event in enumerate(events)
            if event[0] == "link" and event[1].suffix == ".json"
        )
        chunk_fsync = next(
            index
            for index, event in enumerate(events)
            if index > chunk_link
            and event == ("fsync-directory", events[chunk_link][1].parent)
        )
        manifest_fsync = next(
            index
            for index, event in enumerate(events)
            if index > manifest_link
            and event == ("fsync-directory", events[manifest_link][1].parent)
        )
        self.assertLess(chunk_link, chunk_fsync)
        self.assertLess(chunk_fsync, manifest_link)
        self.assertLess(manifest_link, manifest_fsync)

    def test_manifest_directory_fsync_failure_fails_the_commit(self) -> None:
        identity = _identity()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "chunks").mkdir()
            manifest_directory = root / "manifests" / identity.storage_key
            manifest_directory.mkdir(parents=True)
            store = ManifestStore(root)

            def fail_manifest_barrier(path: Path) -> None:
                if Path(path) == manifest_directory:
                    raise OSError("simulated directory fsync failure")

            with (
                mock.patch.object(
                    cache_manifest,
                    "_fsync_directory",
                    fail_manifest_barrier,
                ),
                self.assertRaisesRegex(OSError, "directory fsync failure"),
            ):
                store.commit(
                    identity=identity,
                    context_digest="0" * 64,
                    chunks=(_chunk(),),
                )

    def test_complete_entry_round_trips_byte_exactly(self) -> None:
        context_digest = hashlib.sha256(b"tokens-0-through-255").hexdigest()
        expected = _chunk()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            receipt = store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[expected],
            )

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.manifest_digest, receipt.manifest_digest)
            self.assertEqual(store.restore(lookup), (expected,))

    def test_manifest_probe_then_restore_reads_each_chunk_once(self) -> None:
        """The optimized load path probes metadata, then verifies once."""
        context_digest = hashlib.sha256(b"single-pass-restore").hexdigest()
        expected = _chunk()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[expected],
            )
            chunk_path = next((Path(directory) / "chunks").glob("*.spcc"))
            original_read_bytes = Path.read_bytes
            chunk_reads = 0

            def counted_read_bytes(path: Path) -> bytes:
                nonlocal chunk_reads
                if path == chunk_path:
                    chunk_reads += 1
                return original_read_bytes(path)

            with mock.patch.object(Path, "read_bytes", counted_read_bytes):
                lookup = store.lookup(_identity(), context_digest, verify_chunks=False)
                self.assertTrue(lookup.is_hit, lookup.reason)
                self.assertEqual(store.restore(lookup), (expected,))

            self.assertEqual(chunk_reads, 1)

    def test_restore_hashes_the_authenticated_chunk_only_once(self) -> None:
        """The outer descriptor hash makes per-record rehashing redundant."""
        context_digest = hashlib.sha256(b"single-hash-restore").hexdigest()
        expected = _chunk()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[expected],
            )
            lookup = store.lookup(_identity(), context_digest, verify_chunks=False)
            original_sha256 = cache_manifest._sha256
            hashed_lengths: list[int] = []

            def counted_sha256(
                value: bytes | bytearray | memoryview,
            ) -> str:
                hashed_lengths.append(len(value))
                return original_sha256(value)

            with mock.patch.object(cache_manifest, "_sha256", counted_sha256):
                self.assertEqual(store.restore(lookup), (expected,))

            chunk_bytes = (
                next((Path(directory) / "chunks").glob("*.spcc")).stat().st_size
            )
            self.assertEqual(hashed_lengths, [chunk_bytes])

    def test_standalone_decoder_still_checks_record_digests(self) -> None:
        encoded = bytearray(cache_manifest._encode_chunk(_chunk()))
        prefix = cache_manifest._CHUNK_PREFIX
        _, _, header_length = prefix.unpack_from(encoded)
        payload_start = prefix.size + header_length
        encoded[payload_start] ^= 0x01

        with self.assertRaises(cache_manifest.CacheFormatError):
            cache_manifest._decode_chunk(bytes(encoded))

    def test_new_encoder_is_byte_identical_to_frozen_v1_encoder(self) -> None:
        """Changing assembly mechanics must not change the on-disk ABI."""
        chunk = _chunk()
        payload = bytearray()
        descriptors = []
        for kind in sorted(chunk.records, key=lambda item: item.value):
            value = chunk.records[kind]
            offset = len(payload)
            payload.extend(value)
            descriptors.append(
                {
                    "kind": kind.value,
                    "offset": offset,
                    "length": len(value),
                    "sha256": hashlib.sha256(value).hexdigest(),
                }
            )
        header = cache_manifest._canonical_json(
            {
                "format_abi": cache_manifest.FORMAT_ABI,
                "logical_start": chunk.logical_start,
                "logical_end": chunk.logical_end,
                "records": descriptors,
            }
        )
        legacy = (
            cache_manifest._CHUNK_PREFIX.pack(
                cache_manifest._CHUNK_MAGIC,
                cache_manifest.FORMAT_ABI,
                len(header),
            )
            + header
            + payload
        )
        encoded = cache_manifest._encode_chunk(chunk)

        self.assertEqual(encoded, legacy)
        self.assertEqual(cache_manifest._decode_chunk(legacy), chunk)

    def test_duplicate_writer_cannot_replace_a_committed_context(self) -> None:
        context_digest = hashlib.sha256(b"same-logical-context").hexdigest()
        original = _chunk()
        replacement = ContextChunk(
            logical_start=0,
            logical_end=256,
            records={
                **original.records,
                StateRecord.BOUNDARY_HIDDEN: b"different-boundary",
            },
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[original],
            )

            with self.assertRaises(CommitConflict):
                store.commit(
                    identity=_identity(),
                    context_digest=context_digest,
                    chunks=[replacement],
                )

            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (original,))

    def test_missing_mtp_state_is_rejected_before_publication(self) -> None:
        records = dict(_chunk().records)
        del records[StateRecord.MTP_DRAFT_KV]
        incomplete = ContextChunk(
            logical_start=0,
            logical_end=256,
            records=records,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            with self.assertRaises(IncompleteEntry):
                store.commit(
                    identity=_identity(),
                    context_digest="c" * 64,
                    chunks=(incomplete,),
                )

    def test_live_forward_policy_permits_absent_boundary_hidden(self) -> None:
        records = dict(_chunk().records)
        del records[StateRecord.BOUNDARY_HIDDEN]
        chunk = ContextChunk(logical_start=0, logical_end=256, records=records)
        identity = dataclasses.replace(
            _identity(),
            boundary_hidden_policy="live_forward",
            dcp_degree=4,
            dcp_shard_rank=2,
        )

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            receipt = store.commit(
                identity=identity, context_digest="d" * 64, chunks=(chunk,)
            )
            self.assertEqual(receipt.committed_tokens, 256)
            lookup = store.lookup(identity, "d" * 64)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(store.restore(lookup), (chunk,))
            strict_lookup = store.lookup(
                dataclasses.replace(identity, boundary_hidden_policy="persisted"),
                "d" * 64,
            )
            self.assertFalse(strict_lookup.is_hit)

    def test_shard_rank_isolates_entries_between_ranks(self) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            rank1 = dataclasses.replace(_identity(), dcp_degree=4, dcp_shard_rank=1)
            store.commit(identity=rank1, context_digest="e" * 64, chunks=(chunk,))
            rank2 = dataclasses.replace(rank1, dcp_shard_rank=2)
            self.assertFalse(store.lookup(rank2, "e" * 64).is_hit)
            self.assertTrue(store.lookup(rank1, "e" * 64).is_hit)

    def test_invalidate_purges_corrupt_chunks_so_republish_succeeds(
        self,
    ) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            identity = _identity()
            digest = "f" * 64
            store.commit(identity=identity, context_digest=digest, chunks=(chunk,))
            chunk_file = sorted((Path(directory) / "chunks").glob("*.spcc"))[0]
            payload = bytearray(chunk_file.read_bytes())
            payload[len(payload) // 2] ^= 0x40
            chunk_file.write_bytes(bytes(payload))
            self.assertFalse(store.lookup(identity, digest).is_hit)

            self.assertTrue(store.invalidate(identity, digest))
            self.assertFalse(chunk_file.exists())

            # republication of the identical content must now succeed
            store.commit(identity=identity, context_digest=digest, chunks=(chunk,))
            restored = store.lookup(identity, digest)
            self.assertTrue(restored.is_hit, restored.reason)
            self.assertEqual(store.restore(restored), (chunk,))

    def test_invalidate_keeps_healthy_shared_chunks(self) -> None:
        chunk = _chunk()
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(directory)
            identity = _identity()
            store.commit(identity=identity, context_digest="a" * 64, chunks=(chunk,))
            chunk_file = sorted((Path(directory) / "chunks").glob("*.spcc"))[0]
            self.assertTrue(store.invalidate(identity, "a" * 64))
            self.assertTrue(chunk_file.exists())

    def test_context_chunk_is_an_immutable_snapshot(self) -> None:
        chunk = _chunk()

        with self.assertRaises(TypeError):
            chunk.records[StateRecord.BOUNDARY_HIDDEN] = b"mutated"  # type: ignore[index]

    def test_empty_required_record_is_not_treated_as_complete(self) -> None:
        records = dict(_chunk().records)
        records[StateRecord.MTP_DRAFT_KV] = b""

        with self.assertRaises(IncompleteEntry):
            ContextChunk(
                logical_start=0,
                logical_end=256,
                records=records,
            )

    def test_only_the_final_chunk_may_be_partial(self) -> None:
        context_digest = hashlib.sha256(b"two-short-chunks").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory))

            with self.assertRaises(ValueError):
                store.commit(
                    identity=_identity(),
                    context_digest=context_digest,
                    chunks=[_chunk(0, 128), _chunk(128, 256)],
                )

    def test_corrupted_chunk_becomes_a_clean_cache_miss(self) -> None:
        context_digest = hashlib.sha256(b"corruption-probe").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            chunk_path = next((root / "chunks").glob("*.spcc"))
            encoded = bytearray(chunk_path.read_bytes())
            encoded[-1] ^= 0x01
            chunk_path.write_bytes(encoded)

            lookup = store.lookup(_identity(), context_digest)
            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")
            with self.assertRaises(ValueError):
                store.restore(lookup)

    def test_corruption_after_lookup_cannot_be_restored(self) -> None:
        context_digest = hashlib.sha256(b"post-lookup-corruption").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            lookup = store.lookup(_identity(), context_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)

            chunk_path = next((root / "chunks").glob("*.spcc"))
            with chunk_path.open("ab") as stream:
                stream.write(b"CORRUPT_TRAILER")

            self.assertIsNone(store.restore(lookup))

    def test_restore_rejects_chunk_range_that_disagrees_with_manifest(
        self,
    ) -> None:
        context_digest = hashlib.sha256(b"range-mismatch").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            mismatched = cache_manifest._encode_chunk(_chunk(256, 512))
            mismatched_digest = hashlib.sha256(mismatched).hexdigest()
            (root / "chunks" / f"{mismatched_digest}.spcc").write_bytes(mismatched)
            manifest_path = next((root / "manifests").rglob("*.json"))
            manifest = json.loads(manifest_path.read_bytes())
            manifest["chunks"][0]["sha256"] = mismatched_digest
            manifest["chunks"][0]["bytes"] = len(mismatched)
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")),
                encoding="ascii",
            )

            lookup = store.lookup(
                _identity(),
                context_digest,
                verify_chunks=False,
                verify_chunk_metadata=True,
            )
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertIsNone(store.restore(lookup))

    def test_chunk_decoder_rejects_checksum_valid_trailing_bytes(self) -> None:
        context_digest = hashlib.sha256(b"claimed-trailer").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ManifestStore(root)
            store.commit(
                identity=_identity(),
                context_digest=context_digest,
                chunks=[_chunk()],
            )
            chunk_path = next((root / "chunks").glob("*.spcc"))
            encoded = chunk_path.read_bytes() + b"CLAIMED_BUT_UNPARSED"
            digest = hashlib.sha256(encoded).hexdigest()
            (root / "chunks" / f"{digest}.spcc").write_bytes(encoded)

            manifest_path = next((root / "manifests").rglob("*.json"))
            manifest = json.loads(manifest_path.read_bytes())
            manifest["chunks"][0]["sha256"] = digest
            manifest["chunks"][0]["bytes"] = len(encoded)
            manifest_path.write_text(
                json.dumps(
                    manifest,
                    sort_keys=True,
                    separators=(",", ":"),
                ),
                encoding="ascii",
            )

            lookup = store.lookup(_identity(), context_digest)
            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")


if __name__ == "__main__":
    unittest.main()
