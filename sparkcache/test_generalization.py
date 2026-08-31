"""Tests for model profiles and TP/DCP-degree generalization.

Shares the vLLM stubs and pool helpers with
test_spark_context_cache_connector, which installs them at import.
"""

from __future__ import annotations

import dataclasses
import tempfile
import types
import unittest
from pathlib import Path

import torch

import sparkcache.spark_context_cache_codec as codec
from sparkcache.spark_context_cache_profiles import (
    PROFILES,
    ProfileError,
    resolve_profile,
)
from sparkcache.test_spark_context_cache_connector import (
    _LAYERS,
    _drain,
    _drain_store,
    _make_pools,
    KVConnectorRole,
    SparkCacheConnectorMetadata,
    SparkContextCacheConnector,
    _ReqPlan,
)


def _make_connector_pn(
    root: Path,
    rank: int,
    *,
    tp: int = 4,
    dcp: int = 4,
    pp: int = 1,
    block_size: int = 64,
    extra_config: dict[str, object] | None = None,
    role: KVConnectorRole = KVConnectorRole.WORKER,
    kv_load_failure_policy: str | None = None,
) -> SparkContextCacheConnector:
    values = {
        "spark_cache_root": str(root),
        "spark_cache_model_profile": "glm52-nvfp4",
        "spark_cache_min_span_tokens": "256",
        "spark_cache_target_checkpoint_sha256": "1" * 64,
        "spark_cache_draft_checkpoint_sha256": "2" * 64,
        "spark_cache_draft_policy": "separate",
    }
    values.update(extra_config or {})
    kv_transfer_kwargs: dict[str, object] = {
        "get_from_extra_config": lambda key, default=None: values.get(key, default)
    }
    if kv_load_failure_policy is not None:
        kv_transfer_kwargs["kv_load_failure_policy"] = kv_load_failure_policy
    vllm_config = types.SimpleNamespace(
        kv_transfer_config=types.SimpleNamespace(**kv_transfer_kwargs),
        cache_config=types.SimpleNamespace(block_size=block_size),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            pipeline_parallel_size=pp,
        ),
        model_config=types.SimpleNamespace(model="test-target"),
    )
    connector = SparkContextCacheConnector(
        vllm_config=vllm_config,
        role=role,
        kv_cache_config=None,
    )
    connector._worker_rank = lambda r=rank % max(1, dcp): r  # type: ignore[method-assign]
    connector._physical_rank = lambda r=rank: r  # type: ignore[method-assign]
    return connector


class ProfileRegistryTests(unittest.TestCase):
    def test_glm52_profile_reproduces_frozen_identity_values(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(Path(directory), 0)
        self.assertEqual(
            connector._identity_base,
            {
                "target_checkpoint": "1" * 64,
                "draft_checkpoint": "2" * 64,
                "quantization_layout": "nvfp4_ds_mla-per-token-v1",
                "rope_layout": "glm52-rope-v1",
                "tp_degree": 4,
                "dcp_degree": 4,
                "cp_kv_cache_interleave_size": 1,
                "chunk_tokens": 256,
                "boundary_hidden_policy": "live_forward",
                "draft_kv_policy": "separate",
            },
        )

    def test_unknown_profile_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "unknown model profile"):
                _make_connector_pn(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_model_profile": "no-such-profile"},
                )

    def test_registry_profiles_validate_their_own_reference_geometry(self) -> None:
        for profile in PROFILES.values():
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=64,
                min_span_tokens=profile.chunk_tokens,
                cuda_restore=profile.storage_mode == "per_token_rows",
            )

    def test_resolve_profile_error_lists_known_names(self) -> None:
        with self.assertRaisesRegex(ProfileError, "glm52-nvfp4"):
            resolve_profile("missing")

    def test_deepseek_v4_profile_uses_identity_forked_block_pages(self) -> None:
        profile = resolve_profile("deepseek-v4-fp8-hma")
        self.assertEqual(profile.storage_mode, "block_pages_v1")
        self.assertEqual(profile.required_families, frozenset({"target_ckv"}))
        self.assertEqual(profile.classification_rules, ())
        self.assertIn("block-pages-v1", profile.quantization_layout)
        with self.assertRaisesRegex(ProfileError, "SparkCache CUDA restore"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=64,
                min_span_tokens=256,
                cuda_restore=True,
            )

    def test_glm53_profile_is_distinct_hybrid_namespace(self) -> None:
        profile = resolve_profile("glm53-flash-hybrid")
        self.assertEqual(profile.storage_mode, "block_pages_v1")
        self.assertEqual(profile.default_draft_kv_policy, "separate")
        self.assertEqual(profile.required_families, frozenset({"target_ckv"}))
        self.assertNotEqual(
            profile.quantization_layout,
            resolve_profile("glm52-nvfp4").quantization_layout,
        )

    def test_glm53_profile_accepts_resolved_hybrid_scheduler_block_size(self) -> None:
        profile = resolve_profile("glm53-flash-hybrid")
        profile.validate_for_deployment(
            dcp_degree=1,
            block_size=2304,
            min_span_tokens=4096,
            cuda_restore=False,
        )

    def test_glm53_profile_rejects_incommensurate_scheduler_block_size(self) -> None:
        profile = resolve_profile("glm53-flash-hybrid")
        with self.assertRaisesRegex(ProfileError, "not mutually divisible"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=384,
                min_span_tokens=4096,
                cuda_restore=False,
            )


class DeploymentValidationTests(unittest.TestCase):
    def test_indivisible_dcp_degree_fails_startup(self) -> None:
        # 256 % 3 != 0: a DCP3 store would silently drop owned positions.
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "not divisible"):
                _make_connector_pn(Path(directory), 0, tp=3, dcp=3)

    def test_indivisible_block_size_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "block_size"):
                _make_connector_pn(Path(directory), 0, block_size=96)

    def test_min_span_below_chunk_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "min_span"):
                _make_connector_pn(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_min_span_tokens": "128"},
                )

    def test_pipeline_parallel_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "pipeline parallelism"):
                _make_connector_pn(Path(directory), 0, pp=2)

    def test_non_recompute_load_failure_policy_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "recompute"):
                _make_connector_pn(
                    Path(directory), 0, kv_load_failure_policy="fail"
                )

    def test_recompute_policy_attribute_is_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(
                Path(directory), 0, kv_load_failure_policy="recompute"
            )
        self.assertEqual(connector._chunk_tokens, 256)


class Tp2Dcp1RoundTripTests(unittest.TestCase):
    """TP2 without DCP: every rank stores and restores the full span."""

    SPAN = 1024
    BLOCK_SIZE = 64

    def _plan(self, is_store: bool) -> _ReqPlan:
        # dcp1: 1024 local ordinals -> 16 blocks of 64
        return _ReqPlan(
            request_id="req-0",
            digest="f" * 64,
            span_tokens=self.SPAN,
            block_ids=tuple(range(16)),
            is_store=is_store,
        )

    def test_two_rank_store_wipe_restore_is_byte_exact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            pools = []
            originals = []
            connectors = []
            for rank in range(2):
                root = Path(directory) / f"rank{rank}"
                connector = _make_connector_pn(
                    root, rank, tp=2, dcp=1, block_size=self.BLOCK_SIZE
                )
                pool = _make_pools(18, self.BLOCK_SIZE)
                connector.register_kv_caches(pool)
                connectors.append(connector)
                pools.append(pool)
                originals.append({k: v.clone() for k, v in pool.items()})

            store_meta = SparkCacheConnectorMetadata(plans=[self._plan(True)])
            for connector in connectors:
                connector.bind_connector_metadata(store_meta)
                connector.wait_for_save()
                _drain_store(connector)
                self.assertEqual(connector.counters["store_committed"], 1)

            load_meta = SparkCacheConnectorMetadata(plans=[self._plan(False)])
            for rank, connector in enumerate(connectors):
                for tensor in pools[rank].values():
                    tensor.zero_()
                connector.bind_connector_metadata(load_meta)
                connector.start_load_kv(None)
                self.assertEqual(_drain(connector), {"req-0"})
                self.assertEqual(connector.counters["load_verified"], 1)
                self.assertEqual(connector.get_block_ids_with_load_errors(), set())

            positions = codec.owned_positions(self.SPAN, 1, 0)
            self.assertEqual(len(positions), self.SPAN)
            slots = codec.local_slots_for_positions(
                positions, self._plan(True).block_ids, self.BLOCK_SIZE, 1
            )
            slot_tensor = torch.tensor(slots, dtype=torch.long)
            for rank in range(2):
                for name, width in _LAYERS.items():
                    restored = pools[rank][name].reshape(-1, width)
                    original = originals[rank][name].reshape(-1, width)
                    torch.testing.assert_close(
                        restored[slot_tensor],
                        original[slot_tensor],
                        rtol=0,
                        atol=0,
                    )

    def test_quorum_requires_both_tp_ranks_at_dcp1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(
                Path(directory), 0, tp=2, dcp=1, role=KVConnectorRole.SCHEDULER
            )
            digest = "a" * 64
            connector._quorum[digest] = {0}
            self.assertFalse(connector._has_full_quorum(digest))
            connector._quorum[digest] = {0, 1}
            self.assertTrue(connector._has_full_quorum(digest))


class Tp1RoundTripTests(unittest.TestCase):
    """Minimal single-worker deployment: TP1/DCP1."""

    def test_single_rank_store_wipe_restore(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(
                Path(directory), 0, tp=1, dcp=1, block_size=64
            )
            pool = _make_pools(18, 64)
            connector.register_kv_caches(pool)
            original = {k: v.clone() for k, v in pool.items()}
            plan = _ReqPlan("req-1", "e" * 64, 1024, tuple(range(16)), True)
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(plans=[plan])
            )
            connector.wait_for_save()
            _drain_store(connector)
            for tensor in pool.values():
                tensor.zero_()
            connector.bind_connector_metadata(
                SparkCacheConnectorMetadata(
                    plans=[dataclasses.replace(plan, is_store=False)]
                )
            )
            connector.start_load_kv(None)
            self.assertEqual(_drain(connector), {"req-1"})
            self.assertEqual(connector.counters["load_verified"], 1)
            for name, width in _LAYERS.items():
                restored = pool[name].reshape(-1, width)[: 1024]
                torch.testing.assert_close(
                    restored,
                    original[name].reshape(-1, width)[: 1024],
                    rtol=0,
                    atol=0,
                )
            connector._quorum["e" * 64] = {0}
            self.assertTrue(connector._has_full_quorum("e" * 64))


class SchedulerProbeNoneTests(unittest.TestCase):
    def test_quorum_only_admission_without_local_root(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(
                Path(directory) / "no-cache-here",
                0,
                role=KVConnectorRole.SCHEDULER,
                extra_config={"spark_cache_scheduler_probe": "none"},
            )
            tokens = list(range(1100))
            digest = connector._digest(tokens, 1024)
            connector._quorum[digest] = set(range(4))
            request = types.SimpleNamespace(
                request_id="probe-none", prompt_token_ids=tokens
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (1024, True)
            )

    def test_probe_none_retirement_drops_quorum_without_disk_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(
                Path(directory) / "no-cache-here",
                0,
                role=KVConnectorRole.SCHEDULER,
                extra_config={"spark_cache_scheduler_probe": "none"},
            )
            digest = "b" * 64
            connector._quorum[digest] = set(range(4))
            connector._admitted["req-r"] = (digest, frozenset({1, 2}))
            connector.update_connector_output(
                types.SimpleNamespace(invalid_block_ids={2})
            )
            self.assertNotIn(digest, connector._quorum)
            self.assertEqual(connector._admitted, {})

    def test_invalid_probe_mode_fails_startup(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "scheduler_probe"):
                _make_connector_pn(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_scheduler_probe": "tp1"},
                )


class ClassificationRuleTests(unittest.TestCase):
    def test_glm52_profile_rules_match_registered_layer_names(self) -> None:
        profile = resolve_profile("glm52-nvfp4")
        self.assertEqual(
            codec.classify_layer(
                "model.layers.0.self_attn.indexer_cache",
                profile.classification_rules,
            ),
            "sparse_indexer",
        )
        self.assertEqual(
            codec.classify_layer(
                "draft.layers.0.self_attn.attn",
                profile.classification_rules,
            ),
            "mtp_draft_kv",
        )
        self.assertEqual(
            codec.classify_layer("model.layers.0.self_attn.attn"),
            "target_ckv",
        )

    def test_generic_defaults_do_not_infer_a_model_layout(self) -> None:
        self.assertEqual(
            codec.classify_layer("model.layers.0.self_attn.indexer_cache"),
            "target_ckv",
        )
        plans = codec.build_layer_plans({"opaque.cache": 8})
        self.assertEqual(plans[0].record_kind, "target_ckv")

    def test_custom_rules_first_match_wins_and_default_applies(self) -> None:
        rules = (("indexer", "sparse_indexer"),)
        self.assertEqual(
            codec.classify_layer(
                "model.layers.7.self_attn.indexer.k_cache", rules
            ),
            "sparse_indexer",
        )
        # An unmarked drafter layer falls to the default family, which is
        # how colocated draft state is represented.
        self.assertEqual(
            codec.classify_layer("model.layers.43.self_attn.attn", rules),
            "target_ckv",
        )
        # A name that would trip the reference rules is inert without them.
        self.assertEqual(
            codec.classify_layer("model.layers.1.spectral_mix", rules),
            "target_ckv",
        )

    def test_build_layer_plans_with_reduced_family_requirements(self) -> None:
        plans = codec.build_layer_plans(
            {"model.layers.0.self_attn.attn": 584},
            required_families=frozenset({"target_ckv"}),
            classification_rules=(("indexer", "sparse_indexer"),),
        )
        self.assertEqual(
            [(p.name, p.record_kind) for p in plans],
            [("model.layers.0.self_attn.attn", "target_ckv")],
        )

    def test_chunk_count_parameterized(self) -> None:
        self.assertEqual(codec.chunk_count(1024, 256), 4)
        self.assertEqual(codec.chunk_count(1024, 512), 2)
        with self.assertRaises(codec.CodecError):
            codec.chunk_count(1024, 300)


if __name__ == "__main__":
    unittest.main()
