"""GPU-free tests for the configuration-parsing and identity-construction module.

Covers ``spark_context_cache_config`` in isolation from the connector class:
parses a ``VllmConfig``-like object into an immutable
:class:`ConnectorConfig`, exercises representative startup errors, preserves
identity wire bytes across environment/extra-config precedence, and validates
``ConnectorConfig.build_identity`` against :class:`CacheIdentity`.
"""

from __future__ import annotations

import hashlib
import json
import os
import types
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

import sparkcache.spark_context_cache_config as cfg  # noqa: E402
from sparkcache.spark_context_cache_store import CacheIdentity  # noqa: E402


_ABS_LIB = str((Path.cwd() / "lib.so").resolve())


def _make_vllm_config(
    extra_config: dict[str, object] | None = None,
    tp: int = 4,
    dcp: int = 4,
    pp: int = 1,
    block_size: int = 64,
    cp_kv_cache_interleave_size: int = 1,
    max_model_len: int = 0,
    kv_cache_config: object | None = None,
) -> tuple[types.SimpleNamespace, types.SimpleNamespace]:
    """Build a (vllm_config, kv_transfer_config) pair for parse_connector_config."""

    values = {
        "spark_cache_root": "/cache/test",
        "spark_cache_model_profile": "glm52-nvfp4",
        "spark_cache_target_checkpoint_sha256": "1" * 64,
        "spark_cache_draft_checkpoint_sha256": "2" * 64,
        "spark_cache_draft_policy": "separate",
    }
    values.update(extra_config or {})
    kv_transfer_config = types.SimpleNamespace(
        get_from_extra_config=lambda key, default=None: values.get(key, default),
        kv_load_failure_policy="recompute",
    )
    vllm_config = types.SimpleNamespace(
        kv_transfer_config=kv_transfer_config,
        cache_config=types.SimpleNamespace(block_size=block_size),
        parallel_config=types.SimpleNamespace(
            tensor_parallel_size=tp,
            decode_context_parallel_size=dcp,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            pipeline_parallel_size=pp,
        ),
        model_config=types.SimpleNamespace(max_model_len=max_model_len),
    )
    return vllm_config, kv_transfer_config


_SHA = "a" * 64


class ParseConnectorConfigTests(unittest.TestCase):
    """Focused tests for parse_connector_config field extraction and defaults."""

    def test_returns_frozen_connector_config(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        import dataclasses

        self.assertTrue(dataclasses.is_dataclass(config))
        # Frozen: mutation must raise.
        with self.assertRaises(dataclasses.FrozenInstanceError):
            config.store_enabled = False  # type: ignore[misc]
        with self.assertRaises(TypeError):
            config.identity_base["target_checkpoint"] = "2" * 64  # type: ignore[index]

    def test_defaults_store_and_restore_on(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertTrue(config.store_enabled)
        self.assertTrue(config.restore_enabled)
        self.assertEqual(config.access_mode, "read-write")
        self.assertEqual(config.clear_once_token, "")

    def test_restore_only_mode_reads_without_publishing(self) -> None:
        vllm, _ = _make_vllm_config(
            {"spark_cache_access_mode": "restore-only"}
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertFalse(config.store_enabled)
        self.assertTrue(config.restore_enabled)
        self.assertEqual(config.access_mode, "restore-only")

    def test_access_mode_does_not_change_cache_identity(self) -> None:
        read_write_vllm, _ = _make_vllm_config()
        restore_only_vllm, _ = _make_vllm_config(
            {"spark_cache_access_mode": "restore-only"}
        )

        read_write = cfg.parse_connector_config(
            read_write_vllm,
            read_write_vllm.kv_transfer_config,
            None,
        )
        restore_only = cfg.parse_connector_config(
            restore_only_vllm,
            restore_only_vllm.kv_transfer_config,
            None,
        )

        self.assertEqual(
            read_write.build_identity(0, 0).storage_key,
            restore_only.build_identity(0, 0).storage_key,
        )

    def test_independent_controls_override_access_mode(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_access_mode": "restore-only",
                "spark_cache_store": "1",
                "spark_cache_restore": "0",
            }
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertTrue(config.store_enabled)
        self.assertFalse(config.restore_enabled)
        self.assertEqual(config.access_mode, "store-only")

    def test_unknown_access_mode_is_rejected(self) -> None:
        vllm, _ = _make_vllm_config(
            {"spark_cache_access_mode": "sometimes"}
        )

        with self.assertRaisesRegex(RuntimeError, "spark_cache_access_mode"):
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

    def test_restore_only_mode_disables_publication_accelerators(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_access_mode": "restore-only",
                "spark_cache_streaming_snapshots": "1",
            }
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertFalse(config.streaming_snapshots_enabled)

    def test_store_only_mode_disables_restore_accelerators(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_access_mode": "store-only",
                "spark_cache_cuda_restore": "1",
            }
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertFalse(config.cuda_restore_enabled)

    def test_per_token_rows_reject_nontrivial_dcp_interleave(self) -> None:
        vllm, _ = _make_vllm_config(cp_kv_cache_interleave_size=4)
        with self.assertRaisesRegex(
            RuntimeError,
            "per-token row storage supports only cp_kv_cache_interleave_size=1",
        ):
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

    def test_clear_once_accepts_operator_token_and_absolute_rank_root(self) -> None:
        root = (Path.cwd() / "cache" / "rank-0").resolve()
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_root": str(root),
                "spark_cache_clear_once": "deployment-2026-08-29T20:15",
            }
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertEqual(config.root, str(root))
        self.assertEqual(
            config.clear_once_token,
            "deployment-2026-08-29T20:15",
        )
        self.assertNotIn("clear_once", config.identity_base)

    def test_extra_config_takes_precedence_over_env(self) -> None:
        """Extra-config value wins because env is only the default fallback."""
        vllm, _ = _make_vllm_config(
            {"spark_cache_store": "0", "spark_cache_restore": "false"}
        )
        with mock.patch.dict(os.environ, {"SPARK_CONTEXT_CACHE_STORE": "1"}):
            config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        # Extra config "0"/"false" wins over env "1".
        self.assertFalse(config.store_enabled)
        self.assertFalse(config.restore_enabled)

    def test_root_and_explicit_profile_from_environment(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_root": None})
        vllm.kv_transfer_config.get_from_extra_config = lambda key, default=None: (
            default
        )
        with mock.patch.dict(
            os.environ,
            {
                "SPARK_CONTEXT_CACHE_ROOT": "/env/root",
                "SPARK_CONTEXT_CACHE_MODEL_PROFILE": "glm52-nvfp4",
                "SPARK_CONTEXT_CACHE_TARGET_CHECKPOINT_SHA256": "1" * 64,
                "SPARK_CONTEXT_CACHE_DRAFT_CHECKPOINT_SHA256": "2" * 64,
                "SPARK_CONTEXT_CACHE_DRAFT_POLICY": "separate",
            },
        ):
            config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.root, "/env/root")

    def test_capacity_policy_from_env(self) -> None:
        vllm, _ = _make_vllm_config()
        vllm.kv_transfer_config.get_from_extra_config = lambda key, default=None: (
            default
        )
        with mock.patch.dict(
            os.environ,
            {
                "SPARK_CONTEXT_CACHE_MAX_BYTES": "1000000",
                "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES": "900000",
                "SPARK_CONTEXT_CACHE_TTL_SECONDS": "3600",
                "SPARK_CONTEXT_CACHE_MODEL_PROFILE": "glm52-nvfp4",
                "SPARK_CONTEXT_CACHE_TARGET_CHECKPOINT_SHA256": "1" * 64,
                "SPARK_CONTEXT_CACHE_DRAFT_CHECKPOINT_SHA256": "2" * 64,
                "SPARK_CONTEXT_CACHE_DRAFT_POLICY": "separate",
            },
        ):
            config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.capacity_policy.max_bytes, 1000000)
        self.assertEqual(config.capacity_policy.low_watermark_bytes, 900000)
        self.assertEqual(config.capacity_policy.ttl_seconds, 3600)

    def test_missing_model_profile_is_rejected_explicitly(self) -> None:
        vllm, _ = _make_vllm_config()
        vllm.kv_transfer_config.get_from_extra_config = lambda key, default=None: (
            default
        )
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(RuntimeError, "model_profile is required"),
        ):
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

    def test_load_thread_limit_accepts_8_for_page_restores(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_load_threads": "8"})
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.load_thread_limit, 8)

    def test_load_thread_limit_clamped_to_8(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_load_threads": "16"})
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.load_thread_limit, 8)

    def test_load_thread_limit_native_restore_forces_one(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_load_threads": "4",
                "spark_cache_cuda_restore": "1",
                "spark_cache_cuda_placement_library": _ABS_LIB,
                "spark_cache_cuda_placement_library_sha256": _SHA,
                "spark_cache_cuda_placement_arena_bytes": "67108864",
            }
        )
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertTrue(config.cuda_restore_enabled)
        self.assertEqual(config.load_thread_limit, 1)

    def test_legacy_cuda_restore_config_is_accepted_with_one_warning(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_native_restore": "1",
                "spark_cache_native_library": _ABS_LIB,
                "spark_cache_native_library_sha256": _SHA,
                "spark_cache_native_arena_bytes": "67108864",
                "spark_cache_native_io_workers": "2",
            }
        )
        with (
            mock.patch.object(cfg, "_LEGACY_CUDA_RESTORE_WARNING_EMITTED", False),
            self.assertWarnsRegex(
                FutureWarning, "legacy SparkCache CUDA configuration names"
            ),
        ):
            config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertTrue(config.cuda_restore_enabled)
        self.assertTrue(config.native_restore_enabled)
        self.assertEqual(config.cuda_placement_library_path, _ABS_LIB)
        self.assertEqual(config.cuda_restore_io_workers, 2)

    def test_conflicting_cuda_restore_aliases_are_rejected(self) -> None:
        cases = (
            ("spark_cache_cuda_restore", "1", "spark_cache_native_restore", "0"),
            (
                "spark_cache_cuda_placement_library",
                _ABS_LIB,
                "spark_cache_native_library",
                str((Path.cwd() / "other.so").resolve()),
            ),
            (
                "spark_cache_cuda_placement_arena_bytes",
                "67108864",
                "spark_cache_native_arena_bytes",
                "134217728",
            ),
        )
        for canonical, canonical_value, legacy, legacy_value in cases:
            with self.subTest(canonical=canonical):
                vllm, _ = _make_vllm_config(
                    {canonical: canonical_value, legacy: legacy_value}
                )
                with self.assertRaisesRegex(RuntimeError, "conflicting"):
                    cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

    def test_conflicting_cuda_restore_environment_aliases_are_rejected(self) -> None:
        vllm, _ = _make_vllm_config()
        with (
            mock.patch.dict(
                os.environ,
                {
                    "SPARK_CONTEXT_CACHE_CUDA_RESTORE": "1",
                    "SPARK_CONTEXT_CACHE_NATIVE_RESTORE": "0",
                },
            ),
            self.assertRaisesRegex(RuntimeError, "conflicting"),
        ):
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

    def test_legacy_cuda_restore_environment_is_accepted(self) -> None:
        vllm, _ = _make_vllm_config()
        environment = {
            "SPARK_CONTEXT_CACHE_NATIVE_RESTORE": "1",
            "SPARK_CONTEXT_CACHE_NATIVE_LIBRARY": _ABS_LIB,
            "SPARK_CONTEXT_CACHE_NATIVE_LIBRARY_SHA256": _SHA,
            "SPARK_CONTEXT_CACHE_NATIVE_ARENA_BYTES": "67108864",
            "SPARK_CONTEXT_CACHE_NATIVE_IO_WORKERS": "3",
        }
        with (
            mock.patch.object(cfg, "_LEGACY_CUDA_RESTORE_WARNING_EMITTED", False),
            mock.patch.dict(os.environ, environment),
            self.assertWarnsRegex(
                FutureWarning, "legacy SparkCache CUDA configuration names"
            ),
        ):
            config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertTrue(config.cuda_restore_enabled)
        self.assertEqual(config.cuda_restore_io_workers, 3)

    def test_max_pending_restores_default_64(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.max_pending_restores, 64)

    def test_shared_prefix_lease_ttl_defaults_to_15_seconds(self) -> None:
        vllm, _ = _make_vllm_config()

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertEqual(config.shared_prefix_lease_ttl_seconds, 15.0)

    def test_shared_prefix_lease_ttl_accepts_bounded_override(self) -> None:
        vllm, _ = _make_vllm_config(
            {"spark_cache_shared_prefix_lease_ttl_seconds": "300"}
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertEqual(config.shared_prefix_lease_ttl_seconds, 300.0)

    def test_shared_prefix_lease_ttl_accepts_one_second_minimum(self) -> None:
        vllm, _ = _make_vllm_config(
            {"spark_cache_shared_prefix_lease_ttl_seconds": "1"}
        )

        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)

        self.assertEqual(config.shared_prefix_lease_ttl_seconds, 1.0)

    def test_shared_prefix_lease_ttl_does_not_change_cache_identity(self) -> None:
        default_vllm, _ = _make_vllm_config()
        retained_vllm, _ = _make_vllm_config(
            {"spark_cache_shared_prefix_lease_ttl_seconds": "300"}
        )

        default = cfg.parse_connector_config(
            default_vllm, default_vllm.kv_transfer_config, None
        )
        retained = cfg.parse_connector_config(
            retained_vllm, retained_vllm.kv_transfer_config, None
        )

        self.assertEqual(
            default.build_identity(0, 0).storage_key,
            retained.build_identity(0, 0).storage_key,
        )

    def test_scheduler_probe_default_tp0(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.scheduler_probe, "tp0")

    def test_streaming_snapshots_default_off(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertFalse(config.streaming_snapshots_enabled)
        self.assertFalse(config.async_page_capture_enabled)

    def test_tail_publication_is_opt_in_and_namespace_bound(self) -> None:
        default_vllm, _ = _make_vllm_config()
        tail_vllm, _ = _make_vllm_config(
            {"spark_cache_publication_schema": "tail-cow-v1"}
        )
        default = cfg.parse_connector_config(
            default_vllm,
            default_vllm.kv_transfer_config,
            None,
        )
        tail = cfg.parse_connector_config(
            tail_vllm,
            tail_vllm.kv_transfer_config,
            None,
        )

        self.assertEqual(default.publication_schema, "")
        self.assertNotIn("publication_schema", default.identity_base)
        self.assertEqual(tail.publication_schema, "tail-cow-v1")
        self.assertEqual(
            tail.identity_base["publication_schema"],
            "tail-cow-v1",
        )
        self.assertNotEqual(
            default.build_identity(0, 0).storage_key,
            tail.build_identity(0, 0).storage_key,
        )


class IdentityBaseTests(unittest.TestCase):
    """Tests for identity_base construction and build_identity round-trip."""

    def test_block_page_tail_publication_uses_page_delta_namespace(self) -> None:
        class FullAttentionSpec:
            block_size = 512
            storage_block_size = 512
            page_size_bytes = 528

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
            )
        )
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_publication_schema": "tail-cow-v1",
            },
            tp=1,
            dcp=1,
        )

        config = cfg.parse_connector_config(
            vllm,
            vllm.kv_transfer_config,
            kv_cache_config,
        )

        self.assertEqual(config.publication_schema, "page-tail-cow-v1")
        self.assertEqual(
            config.build_identity(0, 0).publication_schema,
            "page-tail-cow-v1",
        )

    def test_identity_base_contains_required_fields(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        base = config.identity_base
        self.assertEqual(base["target_checkpoint"], "1" * 64)
        self.assertEqual(base["draft_checkpoint"], "2" * 64)
        self.assertEqual(base["tp_degree"], 4)
        self.assertEqual(base["dcp_degree"], 4)
        self.assertIn("quantization_layout", base)
        self.assertIn("rope_layout", base)
        self.assertIn("chunk_tokens", base)
        self.assertIn("boundary_hidden_policy", base)
        self.assertIn("draft_kv_policy", base)

    def test_colocated_draft_policy_inherits_target_id(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_draft_policy": "colocated_target",
                "spark_cache_draft_checkpoint_sha256": "",
            }
        )
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertEqual(config.identity_base["draft_checkpoint"], "1" * 64)
        self.assertEqual(config.identity_base["draft_kv_policy"], "colocated_target")

    def test_build_identity_round_trips_to_cache_identity(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        identity = config.build_identity(shard_rank=1, tp_shard_rank=2)
        self.assertIsInstance(identity, CacheIdentity)
        self.assertEqual(identity.dcp_shard_rank, 1)
        self.assertEqual(identity.tp_shard_rank, 2)
        self.assertEqual(identity.target_checkpoint, "1" * 64)

    def test_build_identity_requires_physical_tp_shard_rank(self) -> None:
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        with self.assertRaises(TypeError):
            config.build_identity(shard_rank=0)  # type: ignore[call-arg]

    def test_identity_wire_bytes_match_reference(self) -> None:
        """The storage_key must equal the SHA-256 of the canonical JSON wire."""
        vllm, _ = _make_vllm_config()
        config = cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        identity = config.build_identity(shard_rank=0, tp_shard_rank=0)
        wire = {
            "target_checkpoint": "1" * 64,
            "draft_checkpoint": "2" * 64,
            "quantization_layout": config.identity_base["quantization_layout"],
            "rope_layout": config.identity_base["rope_layout"],
            "tp_degree": 4,
            "dcp_degree": 4,
            "cp_kv_cache_interleave_size": 1,
            "chunk_tokens": config.identity_base["chunk_tokens"],
            "dcp_shard_rank": 0,
            "tp_shard_rank": 0,
            "boundary_hidden_policy": config.identity_base["boundary_hidden_policy"],
            "draft_kv_policy": "separate",
        }
        expected_key = hashlib.sha256(
            json.dumps(wire, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self.assertEqual(identity.storage_key, expected_key)

    def test_block_pages_v1_adds_topology_digest_and_record_schema(self) -> None:
        """Block-page storage mode appends the topology digest to the
        quantization_layout and sets record_schema in identity_base."""

        class FullAttentionSpec:
            block_size = 512
            storage_block_size = 512
            page_size_bytes = 528

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
            )
        )
        vllm, _ = _make_vllm_config(
            {"spark_cache_model_profile": "deepseek-v4-fp8-hma"},
            dcp=1,
            tp=1,
        )
        config = cfg.parse_connector_config(
            vllm, vllm.kv_transfer_config, kv_cache_config
        )
        self.assertEqual(config.storage_mode, "block_pages_v1")
        self.assertIn("record_schema", config.identity_base)
        self.assertEqual(
            config.identity_base["record_schema"],
            ("target_ckv", "logical_positions"),
        )
        # quantization_layout must end with the topology digest
        topology_digest = cfg.kv_group_topology_digest(kv_cache_config)
        self.assertTrue(
            config.identity_base["quantization_layout"].endswith(":" + topology_digest)
        )
        with self.assertRaises(TypeError):
            config.group_topology[0]["group"] = 9  # type: ignore[index]


class ErrorPathTests(unittest.TestCase):
    """Representative startup paths that reject unverified configuration."""

    def test_dcp_must_divide_tp(self) -> None:
        vllm, _ = _make_vllm_config(tp=4, dcp=3)
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("must divide", str(ctx.exception))
        self.assertIn("spark-context-cache:", str(ctx.exception))

    def test_pipeline_parallel_unsupported(self) -> None:
        vllm, _ = _make_vllm_config(pp=2)
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("pipeline parallelism is unsupported", str(ctx.exception))

    def test_clear_once_requires_a_string_token(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_clear_once": 17})
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("spark_cache_clear_once must be a string", str(ctx.exception))

    def test_clear_once_rejects_unsafe_token_syntax(self) -> None:
        root = str((Path.cwd() / "cache" / "rank-0").resolve())
        for token in (" whitespace", "path/segment", ".leading", "x" * 129):
            with self.subTest(token=token):
                vllm, _ = _make_vllm_config(
                    {
                        "spark_cache_root": root,
                        "spark_cache_clear_once": token,
                    }
                )
                with self.assertRaises(RuntimeError) as ctx:
                    cfg.parse_connector_config(
                        vllm,
                        vllm.kv_transfer_config,
                        None,
                    )
                self.assertIn("1-128 character string", str(ctx.exception))

    def test_clear_once_rejects_relative_and_broad_roots(self) -> None:
        roots = (
            "relative/cache",
            str(Path(Path.cwd().anchor)),
            str(Path(Path.cwd().anchor) / "cache"),
            str(Path.home()),
        )
        for root in roots:
            with self.subTest(root=root):
                vllm, _ = _make_vllm_config(
                    {
                        "spark_cache_root": root,
                        "spark_cache_clear_once": "clear-2026-08-29",
                    }
                )
                with self.assertRaises(RuntimeError):
                    cfg.parse_connector_config(
                        vllm,
                        vllm.kv_transfer_config,
                        None,
                    )

    def test_load_failure_policy_must_be_recompute(self) -> None:
        vllm, _ = _make_vllm_config()
        vllm.kv_transfer_config.kv_load_failure_policy = "raise"
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("kv_load_failure_policy must be 'recompute'", str(ctx.exception))

    def test_target_checkpoint_must_be_sha256(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_target_checkpoint_sha256": "short"})
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn(
            "target checkpoint identity must be a 64-character lowercase SHA-256",
            str(ctx.exception),
        )

    def test_separate_draft_needs_sha256(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_draft_checkpoint_sha256": "short"})
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn(
            "separate draft checkpoint identity must be a 64-character",
            str(ctx.exception),
        )

    def test_colocated_draft_must_match_target(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_draft_policy": "colocated_target"})
        # draft_id is "2"*64 but target is "1"*64 → mismatch
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn(
            "colocated_target draft state must use the target checkpoint identity",
            str(ctx.exception),
        )

    def test_scheduler_probe_must_be_valid(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_scheduler_probe": "invalid"})
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("must be 'tp0' or 'none'", str(ctx.exception))

    def test_max_pending_restores_must_be_positive(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_max_pending_restores": "0"})
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("must be at least 1", str(ctx.exception))

    def test_max_pending_restores_must_be_integer(self) -> None:
        vllm, _ = _make_vllm_config({"spark_cache_max_pending_restores": True})
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("must be an integer", str(ctx.exception))

    def test_shared_prefix_lease_ttl_rejects_invalid_values(self) -> None:
        invalid = {
            "malformed": "not-a-duration",
            "negative": "-1",
            "zero": "0",
            "below-minimum": "0.5",
            "excessive": "300.001",
            "infinite": "inf",
            "boolean": True,
        }
        for label, value in invalid.items():
            with self.subTest(label=label):
                vllm, _ = _make_vllm_config(
                    {"spark_cache_shared_prefix_lease_ttl_seconds": value}
                )
                with self.assertRaisesRegex(
                    RuntimeError,
                    "spark_cache_shared_prefix_lease_ttl_seconds",
                ):
                    cfg.parse_connector_config(
                        vllm, vllm.kv_transfer_config, None
                    )

    def test_nnegative_config_int_rejects_float(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            cfg._nonnegative_config_int(1.5, "test_label")
        self.assertIn("must be an integer", str(ctx.exception))

    def test_nnegative_config_int_rejects_negative(self) -> None:
        with self.assertRaises(RuntimeError) as ctx:
            cfg._nonnegative_config_int(-1, "test_label")
        self.assertIn("must be non-negative", str(ctx.exception))

    def test_native_restore_requires_absolute_path(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_cuda_restore": "1",
                "spark_cache_cuda_placement_library": "relative/path.so",
                "spark_cache_cuda_placement_library_sha256": _SHA,
                "spark_cache_cuda_placement_arena_bytes": "67108864",
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn(
            "absolute CUDA placement library path, a 64-character lowercase SHA-256",
            str(ctx.exception),
        )

    def test_native_restore_requires_valid_sha256(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_cuda_restore": "1",
                "spark_cache_cuda_placement_library": _ABS_LIB,
                "spark_cache_cuda_placement_library_sha256": "short",
                "spark_cache_cuda_placement_arena_bytes": "67108864",
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("64-character lowercase SHA-256", str(ctx.exception))

    def test_native_restore_requires_valid_arena(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_cuda_restore": "1",
                "spark_cache_cuda_placement_library": _ABS_LIB,
                "spark_cache_cuda_placement_library_sha256": _SHA,
                "spark_cache_cuda_placement_arena_bytes": "12345",
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("arena bytes equal to 64, 128, or 256 MiB", str(ctx.exception))

    def test_native_restore_requires_valid_io_workers(self) -> None:
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_cuda_restore": "1",
                "spark_cache_cuda_placement_library": _ABS_LIB,
                "spark_cache_cuda_placement_library_sha256": _SHA,
                "spark_cache_cuda_placement_arena_bytes": "67108864",
                "spark_cache_cuda_restore_io_workers": "50",
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, None)
        self.assertIn("IO workers must be in [1, 32]", str(ctx.exception))

    def test_streaming_snapshots_rejects_block_pages(self) -> None:
        class FullAttentionSpec:
            block_size = 512
            storage_block_size = 512
            page_size_bytes = 528

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
            )
        )
        vllm, _ = _make_vllm_config(
            {
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_streaming_snapshots": "1",
            }
        )
        with self.assertRaises(RuntimeError) as ctx:
            cfg.parse_connector_config(vllm, vllm.kv_transfer_config, kv_cache_config)
        self.assertIn(
            "block-page storage does not support streaming snapshots",
            str(ctx.exception),
        )

    def test_async_page_capture_is_explicit_and_block_page_only(self) -> None:
        class FullAttentionSpec:
            block_size = 512
            storage_block_size = 512
            page_size_bytes = 528

        kv_cache_config = types.SimpleNamespace(
            num_blocks=8,
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
            ),
        )
        block_pages, _ = _make_vllm_config(
            {
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_async_page_capture": "1",
            }
        )
        parsed = cfg.parse_connector_config(
            block_pages,
            block_pages.kv_transfer_config,
            kv_cache_config,
        )
        self.assertTrue(parsed.async_page_capture_enabled)

        rows, _ = _make_vllm_config(
            {"spark_cache_async_page_capture": "1"}
        )
        with self.assertRaisesRegex(RuntimeError, "requires block-page storage"):
            cfg.parse_connector_config(rows, rows.kv_transfer_config, None)

        no_store, _ = _make_vllm_config(
            {
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_async_page_capture": "1",
                "spark_cache_store": "0",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "requires cache publication"):
            cfg.parse_connector_config(
                no_store,
                no_store.kv_transfer_config,
                kv_cache_config,
            )

        tail_store, _ = _make_vllm_config(
            {
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_async_page_capture": "1",
                "spark_cache_publication_schema": "tail-cow-v1",
            }
        )
        with self.assertRaisesRegex(RuntimeError, "complete snapshot publication only"):
            cfg.parse_connector_config(
                tail_store,
                tail_store.kv_transfer_config,
                kv_cache_config,
            )


class KvGroupTopologyTests(unittest.TestCase):
    """Tests for the topology extraction and digest helpers."""

    def test_empty_groups_yield_empty_topology(self) -> None:
        kv_cache_config = types.SimpleNamespace(kv_cache_groups=())
        self.assertEqual(cfg.kv_group_topology(kv_cache_config), ())

    def test_full_attention_policy(self) -> None:
        class FullAttentionSpec:
            block_size = 256
            storage_block_size = 256
            page_size_bytes = 512

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("layer0", "layer1"),
                ),
            )
        )
        topology = cfg.kv_group_topology(kv_cache_config)
        self.assertEqual(len(topology), 1)
        self.assertEqual(topology[0]["reuse_policy"], "full")
        self.assertIsNone(topology[0]["reuse_window_tokens"])
        self.assertEqual(topology[0]["layers"], ("layer0", "layer1"))

    def test_sliding_window_policy(self) -> None:
        class SlidingWindowSpec:
            block_size = 64
            sliding_window = 512

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=SlidingWindowSpec(),
                    is_eagle_group=True,
                    layer_names=("draft",),
                ),
            )
        )
        topology = cfg.kv_group_topology(kv_cache_config)
        self.assertEqual(topology[0]["reuse_policy"], "sliding")
        self.assertEqual(topology[0]["reuse_window_tokens"], 512)
        self.assertTrue(topology[0]["eagle"])

    def test_mamba_align_policy_records_recurrent_identity(self) -> None:
        class MambaSpec:
            block_size = 512
            storage_block_size = 512
            page_size_bytes = 4096
            mamba_cache_mode = "align"
            tokens_per_state = 256
            num_speculative_blocks = 7
            num_prefill_checkpoint_blocks = 1

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=MambaSpec(),
                    is_eagle_group=False,
                    layer_names=("recurrent",),
                ),
            )
        )
        topology = cfg.kv_group_topology(kv_cache_config, dcp_degree=2)
        self.assertEqual(topology[0]["reuse_policy"], "recurrent_align")
        self.assertIsNone(topology[0]["reuse_window_tokens"])
        self.assertTrue(topology[0]["dcp_replicated"])
        self.assertEqual(topology[0]["dcp_shard_count"], 1)
        self.assertEqual(topology[0]["logical_tokens_per_block"], 512)
        self.assertEqual(
            topology[0]["recurrent_state"],
            {
                "mamba_cache_mode": "align",
                "tokens_per_state": 256,
                "num_speculative_blocks": 7,
                "num_prefill_checkpoint_blocks": 1,
            },
        )

        vllm, _ = _make_vllm_config(
            {"spark_cache_model_profile": "glm53-flash-hybrid"},
            dcp=1,
            tp=4,
            block_size=256,
        )
        connector_config = cfg.parse_connector_config(
            vllm, vllm.kv_transfer_config, kv_cache_config
        )
        with self.assertRaises(TypeError):
            connector_config.group_topology[0]["recurrent_state"][
                "tokens_per_state"
            ] = 512

    def test_block_page_identity_separates_complete_manager_pages(self) -> None:
        class FullAttentionSpec:
            block_size = 2304
            storage_block_size = 2304
            page_size_bytes = 4096

        kv_cache_config = types.SimpleNamespace(
            num_blocks=8,
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("attention",),
                ),
            ),
        )
        vllm, _ = _make_vllm_config(
            {"spark_cache_model_profile": "glm53-flash-hybrid"},
            dcp=1,
            tp=1,
            block_size=2304,
        )
        connector_config = cfg.parse_connector_config(
            vllm, vllm.kv_transfer_config, kv_cache_config
        )
        identity = connector_config.build_identity(0, 0)

        self.assertIn(":manager-pages-v2:", identity.quantization_layout)
        row_indexed_identity = replace(
            identity,
            quantization_layout=identity.quantization_layout.replace(
                ":manager-pages-v2:", ":", 1
            ),
        )
        self.assertNotEqual(identity.storage_key, row_indexed_identity.storage_key)

    def test_glm53_manager_pages_accept_tp4_dcp2_and_dcp4(self) -> None:
        """Opaque manager pages retain the configured DCP degree in identity."""

        class FullAttentionSpec:
            block_size = 2304
            storage_block_size = 2304
            page_size_bytes = 4096

        kv_cache_config = types.SimpleNamespace(
            num_blocks=8,
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("attention",),
                ),
            ),
        )
        for dcp_degree in (2, 4):
            with self.subTest(dcp_degree=dcp_degree):
                vllm, _ = _make_vllm_config(
                    {"spark_cache_model_profile": "glm53-flash-hybrid"},
                    dcp=dcp_degree,
                    tp=4,
                    block_size=2304,
                    cp_kv_cache_interleave_size=4,
                )
                connector_config = cfg.parse_connector_config(
                    vllm,
                    vllm.kv_transfer_config,
                    kv_cache_config,
                )
                self.assertEqual(connector_config.dcp_degree, dcp_degree)
                self.assertEqual(connector_config.cp_kv_cache_interleave_size, 4)
                self.assertFalse(
                    connector_config.group_topology[0]["dcp_replicated"]
                )
                self.assertEqual(
                    connector_config.group_topology[0]["dcp_shard_count"],
                    dcp_degree,
                )
                self.assertEqual(
                    connector_config.group_topology[0][
                        "logical_tokens_per_block"
                    ],
                    2304 * dcp_degree,
                )
                self.assertEqual(
                    connector_config.build_identity(0, 0).dcp_degree,
                    dcp_degree,
                )
                self.assertEqual(
                    connector_config.build_identity(
                        0,
                        0,
                    ).cp_kv_cache_interleave_size,
                    4,
                )

    def test_mamba_non_align_policy_fails_closed(self) -> None:
        class MambaSpec:
            block_size = 256
            mamba_cache_mode = "all"

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=MambaSpec(),
                    is_eagle_group=False,
                    layer_names=("recurrent",),
                ),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "requires mamba_cache_mode 'align'"):
            cfg.kv_group_topology(kv_cache_config)

    def test_digest_is_stable_sha256(self) -> None:
        class FullAttentionSpec:
            block_size = 256
            storage_block_size = 256
            page_size_bytes = 512

        kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("a", "b"),
                ),
            )
        )
        digest = cfg.kv_group_topology_digest(kv_cache_config)
        self.assertEqual(len(digest), 64)
        # Same input → same digest
        self.assertEqual(cfg.kv_group_topology_digest(kv_cache_config), digest)

    def test_digest_changes_with_topology(self) -> None:
        class FullAttentionSpec:
            block_size = 256
            storage_block_size = 256
            page_size_bytes = 512

        config_a = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("a",),
                ),
            )
        )
        config_b = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("b",),
                ),
            )
        )
        self.assertNotEqual(
            cfg.kv_group_topology_digest(config_a),
            cfg.kv_group_topology_digest(config_b),
        )


if __name__ == "__main__":
    unittest.main()
