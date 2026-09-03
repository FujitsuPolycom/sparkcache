"""Regression tests for the page-tail qualification harness.

The suite covers the observable qualification contract: byte-exact fixture
reuse, complete-snapshot comparison, bounded page-delta publication with
authenticated ancestry flattening, restart restoration from on-disk state
alone, corruption rejection, cache-identity namespace separation, and the
receipt schema's rejection of understated evidence.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sparkcache.qualification import (
    RECEIPT_SCHEMA,
    build_fixture,
    run_page_tail_qualification,
    validate_receipt,
)
from sparkcache.qualification.fixture import snapshot_digest
from sparkcache.qualification.harness import (
    PAGE_DELTA_SCHEMA,
    PAGE_SNAPSHOT_SCHEMA,
    _cache_identity,
)
from sparkcache.persistent_context_cache.cache_manifest import (
    CacheIdentity,
    ManifestStore,
)
from sparkcache.spark_context_cache_codec import context_prefix_digest


_FIXTURE = build_fixture()


def _manifest(root: Path, identity: CacheIdentity, digest: str) -> dict:
    path = root / "manifests" / identity.storage_key / f"{digest}.json"
    return json.loads(path.read_bytes())


class FixtureTests(unittest.TestCase):
    def test_reused_page_payloads_are_byte_identical_across_snapshots(
        self,
    ) -> None:
        base = _FIXTURE.snapshot_bytes(1)
        extended = _FIXTURE.snapshot_bytes(4)
        # Reused base pages must remain byte-identical inside the extended
        # snapshot whenever the extension reuses page 0.
        reuse_count = 1
        from sparkcache.spark_context_cache_hybrid import plan_page_snapshot

        base_plan = plan_page_snapshot(_FIXTURE.layout, base, (1, 1))
        extended_plan = plan_page_snapshot(_FIXTURE.layout, extended, (4, 1))
        for base_span, extended_span in zip(
            base_plan.spans, extended_plan.spans, strict=True
        ):
            base_view = base[base_span.source_start : base_span.source_end]
            extended_view = extended[
                extended_span.source_start : extended_span.source_end
            ]
            reused = base_view[: reuse_count * base_span.bytes_per_page]
            self.assertEqual(
                extended_view[: reuse_count * base_span.bytes_per_page],
                reused,
                extended_span.layer_name,
            )

    def test_fixture_namespace_digest_uses_the_production_topology_function(
        self,
    ) -> None:
        identity = _cache_identity(_FIXTURE, 0)
        self.assertEqual(identity.publication_schema, "page-tail-cow-v1")
        self.assertEqual(identity.tp_degree, 4)
        self.assertEqual(identity.dcp_degree, 4)
        self.assertTrue(
            identity.quantization_layout.startswith(
                "glm53-flash-hybrid-block-pages-v1:manager-pages-v2:"
            )
        )

    def test_shard_ranks_produce_distinct_storage_keys(self) -> None:
        keys = {_cache_identity(_FIXTURE, rank).storage_key for rank in range(4)}
        self.assertEqual(len(keys), 4)


class CohortTests(unittest.TestCase):
    def test_cohort_restores_byte_exactly_and_validates_clean(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            receipt = run_page_tail_qualification(Path(directory))
            self.assertEqual(validate_receipt(receipt), [])
            self.assertEqual(receipt["schema"], RECEIPT_SCHEMA)
            self.assertEqual(receipt["runner"], "local")
            self.assertTrue(receipt["validation"]["byte_exact_restores"])
            self.assertTrue(receipt["validation"]["complete_snapshot_equal"])
            self.assertTrue(receipt["corruption"]["rejected"])

    def test_delta_publication_carries_only_changed_pages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_page_tail_qualification(root, corrupt_result=False)
            identity = _cache_identity(_FIXTURE, 0)
            for step in receipt["deltas"]:
                self.assertEqual(step["manifest_schema"], PAGE_DELTA_SCHEMA)
                manifest = _manifest(root, identity, step["context_digest"])
                object_bytes = sum(
                    int(descriptor["bytes"]) for descriptor in manifest["delta_objects"]
                )
                self.assertEqual(step["delta_encoded_bytes"], object_bytes)
            base_bytes = receipt["base"]["snapshot_bytes"]
            delta_total = sum(
                step["delta_encoded_bytes"] for step in receipt["deltas"]
            )
            self.assertLess(delta_total, 2 * base_bytes)

    def test_depth_limit_flattens_ancestry_without_a_complete_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_page_tail_qualification(root, steps=(1, 1, 1))
            self.assertEqual(validate_receipt(receipt), [])
            schemas = [step["manifest_schema"] for step in receipt["deltas"]]
            self.assertEqual(
                schemas,
                [PAGE_DELTA_SCHEMA, PAGE_DELTA_SCHEMA, PAGE_DELTA_SCHEMA],
            )
            self.assertTrue(receipt["deltas"][2]["flattened"])
            self.assertFalse(receipt["deltas"][2]["compacted"])
            self.assertFalse(receipt["deltas"][0]["compacted"])

    def test_restart_restoration_succeeds_without_in_memory_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_page_tail_qualification(root, corrupt_result=False)
            digest = receipt["result"]["context_digest"]
            identity = _cache_identity(_FIXTURE, 0)
            manifest = _manifest(root, identity, digest)
            self.assertIn(manifest["schema"], (PAGE_DELTA_SCHEMA, PAGE_SNAPSHOT_SCHEMA))
            restarted = ManifestStore(root)
            page_count = manifest["committed_tokens"] // _FIXTURE.tokens_per_page
            lookup = restarted.lookup(identity, digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            restored = restarted.restore_page_snapshot(
                lookup,
                layout=_FIXTURE.layout,
                result_block_counts=(page_count, 1),
                result_boundary_tokens=manifest["committed_tokens"],
            )
            self.assertEqual(
                snapshot_digest(bytes(restored)),
                receipt["result"]["expected_snapshot_sha256"],
            )
            self.assertGreater(receipt["result"]["restart_restore_ms"], 0.0)

    def test_corrupted_result_object_is_rejected_and_the_root_survives(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            receipt = run_page_tail_qualification(root)
            identity = _cache_identity(_FIXTURE, 0)
            digest = receipt["result"]["context_digest"]
            manifest = _manifest(root, identity, digest)
            descriptor = (manifest.get("delta_objects") or manifest["snapshot_objects"])[-1]
            object_path = root / "chunks" / f"{descriptor['sha256']}.spcc"
            # The probe restored the original bytes after rejection.
            import hashlib

            self.assertEqual(
                hashlib.sha256(object_path.read_bytes()).hexdigest(),
                descriptor["sha256"],
            )
            lookup = ManifestStore(root).lookup(identity, digest)
            self.assertTrue(lookup.is_hit, lookup.reason)


class NamespaceTests(unittest.TestCase):
    def test_page_tail_identity_never_reads_a_snapshot_v1_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "namespaces"
            store = ManifestStore(root)
            page_identity = _cache_identity(_FIXTURE, 0)
            snap_identity = CacheIdentity(
                **{
                    **page_identity.to_wire(),
                    "record_schema": tuple(page_identity.record_schema),
                    "publication_schema": "",
                }
            )
            tokens = tuple(range(256))
            digest = context_prefix_digest(
                tokens, _FIXTURE.identity_salt, token_count=256
            )
            snapshot = _FIXTURE.snapshot_bytes(1)
            store.commit_page_snapshot(
                identity=snap_identity,
                context_digest=digest,
                span_tokens=256,
                snapshot=snapshot,
            )
            self.assertTrue(store.lookup(snap_identity, digest).is_hit)
            miss = store.lookup(page_identity, digest)
            self.assertFalse(miss.is_hit)
            self.assertEqual(miss.reason, "absent")

    def test_shard_one_never_reads_shard_zero_entry(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = ManifestStore(Path(directory) / "shards")
            identity0 = _cache_identity(_FIXTURE, 0)
            identity1 = _cache_identity(_FIXTURE, 1)
            tokens = tuple(range(256))
            digest = context_prefix_digest(
                tokens, _FIXTURE.identity_salt, token_count=256
            )
            store.commit_page_snapshot(
                identity=identity0,
                context_digest=digest,
                span_tokens=256,
                snapshot=_FIXTURE.snapshot_bytes(1),
            )
            self.assertNotEqual(identity0.storage_key, identity1.storage_key)
            self.assertFalse(store.lookup(identity1, digest).is_hit)


class ReceiptValidationTests(unittest.TestCase):
    def _receipt(self, **kwargs) -> dict:
        with tempfile.TemporaryDirectory() as directory:
            return run_page_tail_qualification(
                Path(directory), corrupt_result=False, **kwargs
            )

    def _live_receipt(self) -> dict:
        path = (
            Path(__file__).resolve().parents[1]
            / "evidence"
            / "glm53-flash-dcp4-page-tail"
            / "qualification.json"
        )
        return json.loads(path.read_text(encoding="utf-8"))

    def test_checked_in_live_receipt_is_valid(self) -> None:
        self.assertEqual(validate_receipt(self._live_receipt()), [])

    def test_live_receipt_requires_four_rank_measurements(self) -> None:
        broken = self._live_receipt()
        broken["live"]["restart_restore_ms_by_rank"].pop()
        errors = validate_receipt(broken)
        self.assertIn(
            "live restart_restore_ms_by_rank must contain four rank values",
            errors,
        )

    def test_live_receipt_binds_summary_to_rank_measurement(self) -> None:
        broken = self._live_receipt()
        broken["result"]["restart_restore_ms"] = 1.0
        errors = validate_receipt(broken)
        self.assertIn(
            "restart_restore_ms is not a recorded restart rank value",
            errors,
        )

    def test_schema_rejects_missing_restart_measurement(self) -> None:
        receipt = self._receipt()
        broken = json.loads(json.dumps(receipt))
        del broken["result"]["restart_restore_ms"]
        errors = validate_receipt(broken)
        self.assertIn("restart_restore_ms is invalid", errors)

    def test_schema_rejects_expected_snapshot_digest_mismatch(self) -> None:
        receipt = self._receipt()
        broken = json.loads(json.dumps(receipt))
        broken["result"]["expected_snapshot_sha256"] = "0" * 64
        errors = validate_receipt(broken)
        self.assertIn(
            "result expected snapshot digest differs from the final delta",
            errors,
        )

    def test_schema_rejects_unchained_base_digest(self) -> None:
        receipt = self._receipt(steps=(1, 1))
        broken = json.loads(json.dumps(receipt))
        broken["deltas"][1]["base_context_digest"] = "0" * 64
        errors = validate_receipt(broken)
        self.assertIn("delta 1 base digest does not chain", errors)

    def test_schema_rejects_false_corruption_rejection(self) -> None:
        receipt = self._receipt()
        broken = json.loads(json.dumps(receipt))
        broken["validation"]["corruption_rejected"] = False
        errors = validate_receipt(broken)
        self.assertIn(
            "corruption_rejected must be true for an accepted cohort", errors
        )

    def test_schema_rejects_weakened_publication_schema(self) -> None:
        receipt = self._receipt()
        broken = json.loads(json.dumps(receipt))
        broken["identity"]["publication_schema"] = "snapshot-v1"
        errors = validate_receipt(broken)
        self.assertIn("identity publication schema differs", errors)

    def test_schema_rejects_non_object_receipt(self) -> None:
        self.assertEqual(validate_receipt(None), ["receipt is not an object"])

    def test_schema_rejects_incomplete_base_without_raising(self) -> None:
        receipt = self._receipt()
        broken = json.loads(json.dumps(receipt))
        del broken["base"]["committed_tokens"]
        del broken["base"]["context_digest"]
        errors = validate_receipt(broken)
        self.assertIn("base committed_tokens is invalid", errors)
        self.assertIn("base digest is not a digest", errors)

    def test_schema_rejects_non_object_final_delta_without_raising(self) -> None:
        receipt = self._receipt()
        broken = json.loads(json.dumps(receipt))
        broken["deltas"][-1] = "not-an-object"
        errors = validate_receipt(broken)
        self.assertIn("delta 1 is not an object", errors)


if __name__ == "__main__":
    unittest.main()
