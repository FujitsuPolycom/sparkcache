"""Regression tests for the defects recorded in DEFECTS.md.

Each test names the defect it pins. The vLLM stubs and connector factory are
shared with test_spark_context_cache_connector, which installs them at import.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import sparkcache.native.python as native_package
import sparkcache.spark_cache_cuda as canonical_cuda
import sparkcache.spark_cache_native as legacy_native
import sparkcache.spark_context_cache_store as package_store
import sparkcache.streaming as streaming
import sparkcache.spark_context_cache_store as flat_store
from sparkcache.spark_context_cache_native_placement import ParkedRestore

from sparkcache.spark_context_cache_hybrid import HybridCodecError
from sparkcache.spark_context_cache_profiles import resolve_profile
from sparkcache.test_spark_context_cache_connector import (
    _drain_store,
    _hybrid_kv_cache_config,
    _make_connector,
    _make_pools,
    KVConnectorRole,
    SparkCacheConnectorMetadata,
    connector_module,
    _ReqPlan,
)
from sparkcache.spark_context_cache_store import IncompleteEntry


class DefectD6QuorumDeltaReportingTests(unittest.TestCase):
    """D-6: quorum control traffic stays bounded as inventories grow."""

    @staticmethod
    def _output(report: dict[str, object]) -> types.SimpleNamespace:
        stats = connector_module.SparkCacheStats(data={"reports": [report]})
        return types.SimpleNamespace(kv_connector_stats=stats)

    @staticmethod
    def _report(
        *,
        generation: str,
        epoch: int,
        sequence: int,
        added: list[str],
        removed: list[str],
    ) -> dict[str, object]:
        return {
            "rank": 0,
            "protocol": connector_module._QUORUM_DELTA_PROTOCOL,
            "generation": generation,
            "generation_epoch": epoch,
            "held_count": len(added),
            "delta": {
                "sequence": sequence,
                "base_sequence": sequence - 1,
                "added": added,
                "removed": removed,
            },
        }

    def test_stable_inventory_reports_do_not_retransmit_complete_held_sets(
        self,
    ) -> None:
        payload_sizes = []
        compatibility_withdrawals = []
        for inventory_size in (1, 128, 1024):
            with tempfile.TemporaryDirectory() as directory:
                connector = _make_connector(Path(directory), 0, 64)
                connector._held = {
                    hashlib.sha256(str(index).encode()).hexdigest()
                    for index in range(inventory_size)
                }
                connector.get_kv_connector_stats()
                report = connector.get_kv_connector_stats().data["reports"][0]
                payload_sizes.append(len(json.dumps(report, sort_keys=True)))
                compatibility_withdrawals.append(report["held"])

        self.assertLessEqual(max(payload_sizes), 20_000)
        self.assertLessEqual(max(payload_sizes), min(payload_sizes) * 16)
        self.assertEqual(compatibility_withdrawals, [[], [], []])

    def test_missed_duplicate_and_reordered_checkpoint_chunks_converge(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = _make_connector(root / "worker", 0, 64)
            scheduler = _make_connector(
                root / "scheduler",
                0,
                64,
                role=KVConnectorRole.SCHEDULER,
            )
            initial = {
                hashlib.sha256(f"initial-{index}".encode()).hexdigest()
                for index in range(130)
            }
            worker._held = set(initial)
            first_cycle = [
                worker.get_kv_connector_stats().data["reports"][0] for _ in range(3)
            ]
            for report in (
                first_cycle[2],
                first_cycle[2],
                first_cycle[0],
                first_cycle[1],
            ):
                scheduler._absorb_quorum(self._output(report))
            self.assertEqual(scheduler._worker_held[0], initial)
            stable_report = worker.get_kv_connector_stats().data["reports"][0]
            with mock.patch.object(
                scheduler,
                "_replace_worker_held",
                wraps=scheduler._replace_worker_held,
            ) as replace_worker_held:
                scheduler._absorb_quorum(self._output(stable_report))
            replace_worker_held.assert_not_called()
            self.assertNotIn(0, scheduler._worker_checkpoints)

            removed = min(initial)
            added = hashlib.sha256(b"replacement").hexdigest()
            replacement = initial - {removed} | {added}
            worker._held = set(replacement)
            worker.get_kv_connector_stats()  # dropped checkpoint chunk zero
            delivered = [
                worker.get_kv_connector_stats().data["reports"][0] for _ in range(5)
            ]
            complete_cycle = delivered[2:]
            for report in (
                complete_cycle[2],
                complete_cycle[0],
                complete_cycle[2],
                complete_cycle[1],
            ):
                scheduler._absorb_quorum(self._output(report))

            self.assertEqual(scheduler._worker_held[0], replacement)
            self.assertNotIn(0, scheduler._worker_desynchronized)
            self.assertNotIn(removed, scheduler._quorum)
            self.assertEqual(scheduler._quorum[added], {0})

    def test_reordered_deltas_fail_closed_then_converge(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler = _make_connector(
                Path(directory),
                0,
                64,
                role=KVConnectorRole.SCHEDULER,
            )
            first = "a" * 64
            replacement = "b" * 64
            sequence_one = self._report(
                generation="generation-a",
                epoch=100,
                sequence=1,
                added=[first],
                removed=[],
            )
            sequence_two = self._report(
                generation="generation-a",
                epoch=100,
                sequence=2,
                added=[replacement],
                removed=[first],
            )

            scheduler._absorb_quorum(self._output(sequence_two))
            self.assertNotIn(first, scheduler._quorum)
            self.assertNotIn(replacement, scheduler._quorum)
            scheduler._absorb_quorum(self._output(sequence_two))
            scheduler._absorb_quorum(self._output(sequence_one))

            self.assertNotIn(first, scheduler._quorum)
            self.assertEqual(scheduler._quorum[replacement], {0})
            self.assertNotIn(0, scheduler._worker_desynchronized)

    def test_post_restart_report_rejects_delayed_process_generation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler = _make_connector(
                Path(directory),
                0,
                64,
                role=KVConnectorRole.SCHEDULER,
            )
            stale = "c" * 64
            serving = "d" * 64
            replacement = hashlib.sha256(b"second-restart").hexdigest()
            before_restart = self._report(
                generation="generation-before-restart",
                epoch=100,
                sequence=1,
                added=[stale],
                removed=[],
            )
            after_restart = self._report(
                generation="generation-after-restart",
                epoch=50,
                sequence=1,
                added=[serving],
                removed=[],
            )
            second_restart = self._report(
                generation="generation-after-second-restart",
                epoch=25,
                sequence=1,
                added=[replacement],
                removed=[],
            )
            scheduler._absorb_quorum(self._output(before_restart))
            scheduler._absorb_quorum(self._output(after_restart))
            scheduler._absorb_quorum(self._output(second_restart))
            scheduler._absorb_quorum(self._output(before_restart))
            scheduler._absorb_quorum(self._output(after_restart))

            self.assertNotIn(stale, scheduler._quorum)
            self.assertNotIn(serving, scheduler._quorum)
            self.assertEqual(scheduler._quorum[replacement], {0})
            self.assertEqual(
                scheduler._worker_generations[0],
                "generation-after-second-restart",
            )
            self.assertEqual(
                scheduler._worker_retired_generations[0],
                ["generation-before-restart", "generation-after-restart"],
            )

    def test_retired_generation_memory_is_bounded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler = _make_connector(
                Path(directory),
                0,
                64,
                role=KVConnectorRole.SCHEDULER,
            )
            report_count = connector_module._QUORUM_RETIRED_GENERATION_LIMIT + 5
            reports = []
            for index in range(report_count):
                digest = hashlib.sha256(f"restart-{index}".encode()).hexdigest()
                report = self._report(
                    generation=f"generation-{index}",
                    epoch=report_count - index,
                    sequence=1,
                    added=[digest],
                    removed=[],
                )
                reports.append(report)
                scheduler._absorb_quorum(self._output(report))

            retired = scheduler._worker_retired_generations[0]
            self.assertEqual(
                len(retired), connector_module._QUORUM_RETIRED_GENERATION_LIMIT
            )
            serving_generation = scheduler._worker_generations[0]
            scheduler._absorb_quorum(self._output(reports[-2]))
            self.assertEqual(scheduler._worker_generations[0], serving_generation)

    def test_same_rank_aggregation_loss_recovers_from_replayed_delta(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            scheduler = _make_connector(
                Path(directory),
                0,
                64,
                role=KVConnectorRole.SCHEDULER,
            )
            first = "e" * 64
            replacement = "f" * 64
            sequence_one = self._report(
                generation="generation-a",
                epoch=100,
                sequence=1,
                added=[first],
                removed=[],
            )
            sequence_two = self._report(
                generation="generation-a",
                epoch=100,
                sequence=2,
                added=[replacement],
                removed=[first],
            )
            merged = connector_module.SparkCacheStats(
                data={"reports": [sequence_one]}
            ).aggregate(
                connector_module.SparkCacheStats(data={"reports": [sequence_two]})
            )
            self.assertEqual(merged.data["reports"][0]["delta"]["sequence"], 2)

            scheduler._absorb_quorum(types.SimpleNamespace(kv_connector_stats=merged))
            self.assertNotIn(replacement, scheduler._quorum)
            scheduler._absorb_quorum(self._output(sequence_one))
            self.assertEqual(scheduler._quorum[replacement], {0})


class DeadInterfaceRemovalTests(unittest.TestCase):
    """D-10: native and streaming code expose one model-serving interface each."""

    def test_native_binding_and_runtime_interfaces_have_one_owner(self) -> None:
        self.assertIs(native_package.PlacementConfig, canonical_cuda.PlacementConfig)
        self.assertIs(canonical_cuda.PlacementConfig, legacy_native.PlacementConfig)
        self.assertIs(
            canonical_cuda.CudaPlacementError, legacy_native.NativePlacementError
        )
        self.assertIs(flat_store.ManifestStore, package_store.ManifestStore)
        self.assertFalse(hasattr(canonical_cuda, "PlacementHandle"))
        self.assertFalse(hasattr(ParkedRestore, "submit_transposed_slab"))
        self.assertFalse(hasattr(streaming, "PreemptionDrainAdapter"))


class RestoreBacklogAdmissionTests(unittest.TestCase):
    """D-1: the async-restore bound must refuse admission, never evict."""

    SPAN = 1024
    BLOCKS = (3, 0, 5, 1)

    def _connector_with_hit(self, root: Path):
        connector = _make_connector(root, 0, 64)
        connector.register_kv_caches(_make_pools(8, 64))
        tokens = list(range(1100))
        digest = connector._digest(tokens, self.SPAN)
        connector.bind_connector_metadata(
            SparkCacheConnectorMetadata(
                plans=[_ReqPlan("seed", digest, self.SPAN, self.BLOCKS, True)]
            )
        )
        connector.wait_for_save()
        _drain_store(connector)
        connector.clear_connector_metadata()
        connector._quorum[digest] = set(range(4))
        return connector, tokens

    def test_full_backlog_reports_a_miss_and_evicts_nothing(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector, tokens = self._connector_with_hit(Path(directory))
            backlog = {
                f"parked-{index}": (f"{index:064d}", self.SPAN)
                for index in range(connector._max_pending_restores)
            }
            connector._need_load.update(backlog)
            request = types.SimpleNamespace(
                request_id="overflow", prompt_token_ids=tokens
            )
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0), (0, False)
            )
            # every already-promised entry survives untouched
            self.assertEqual(connector._need_load, backlog)
            self.assertEqual(connector.counters["restore_skip_backlog"], 1)

    def test_below_bound_still_admits(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector, tokens = self._connector_with_hit(Path(directory))
            request = types.SimpleNamespace(request_id="fits", prompt_token_ids=tokens)
            self.assertEqual(
                connector.get_num_new_matched_tokens(request, 0),
                (self.SPAN, True),
            )
            self.assertIn("fits", connector._need_load)


class MaxPendingRestoreParsingTests(unittest.TestCase):
    """D-11: the restore admission bound accepts only integer values."""

    def test_malformed_string_has_a_labeled_startup_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "spark-context-cache: spark_cache_max_pending_restores must be an integer",
            ):
                _make_connector(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_max_pending_restores": "64x"},
                )

    def test_null_has_a_labeled_startup_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "spark-context-cache: spark_cache_max_pending_restores must be an integer",
            ):
                _make_connector(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_max_pending_restores": None},
                )

    def test_boolean_is_not_silently_coerced_to_one(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(
                RuntimeError,
                "spark-context-cache: spark_cache_max_pending_restores must be an integer",
            ):
                _make_connector(
                    Path(directory),
                    0,
                    extra_config={"spark_cache_max_pending_restores": True},
                )

    def test_decimal_string_sets_the_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={"spark_cache_max_pending_restores": "7"},
            )
        self.assertEqual(connector._max_pending_restores, 7)


class RequestFinishedCleanupTests(unittest.TestCase):
    """D-1: finished requests release their scheduler-side tracking state."""

    def test_finished_request_clears_all_tracking_maps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory), 0, 64, role=KVConnectorRole.SCHEDULER
            )
            digest = "a" * 64
            connector._need_load["req-f"] = (digest, 1024)
            connector._pending_async_loads["req-f"] = (digest, 1024, (1, 2))
            connector._admitted["req-f"] = (digest, frozenset({1, 2}))
            connector._store_progress["req-f"] = (digest, 1024, 256, [1, 2])
            request = types.SimpleNamespace(request_id="req-f")
            self.assertEqual(connector.request_finished(request, [1, 2]), (False, None))
            self.assertEqual(connector._need_load, {})
            self.assertEqual(connector._pending_async_loads, {})
            self.assertEqual(connector._admitted, {})
            self.assertEqual(connector._store_progress, {})


class RowsViewAliasingTests(unittest.TestCase):
    """D-3: the restore scatter target must alias KV storage or fail."""

    def test_contiguous_tensor_yields_an_alias(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            tensor = torch.zeros(4, 8, 16, dtype=torch.float32)
            rows = connector._rows_view(tensor)
            self.assertEqual(rows.data_ptr(), tensor.data_ptr())
            self.assertEqual(rows.shape, (32, 16))

    def test_non_viewable_tensor_raises_instead_of_copying(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            # transpose makes the row merge impossible without a copy;
            # reshape would silently return that copy and the restore
            # scatter would write into a discarded temporary
            tensor = torch.zeros(8, 64, 32).transpose(1, 2)
            with self.assertRaises(RuntimeError):
                connector._rows_view(tensor)


class HybridReuseWindowTests(unittest.TestCase):
    """D-12: hybrid snapshots exclude recycled recurrent-cache block IDs."""

    @staticmethod
    def _windowed_config(sliding_window: int) -> types.SimpleNamespace:
        class FullAttentionSpec:
            block_size = 256
            storage_block_size = 256
            page_size_bytes = 64

        class SlidingWindowSpec:
            def __init__(self, window: int) -> None:
                self.sliding_window = window

        class SlidingWindowMLASpec(SlidingWindowSpec):
            pass

        return types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(
                        block_size=64,
                        storage_block_size=64,
                        page_size_bytes=64,
                        kv_cache_specs={"swa": SlidingWindowMLASpec(sliding_window)},
                    ),
                    is_eagle_group=False,
                    layer_names=("swa",),
                ),
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(
                        block_size=4,
                        storage_block_size=4,
                        page_size_bytes=32,
                        kv_cache_specs={"state": SlidingWindowMLASpec(8)},
                    ),
                    is_eagle_group=False,
                    layer_names=("state",),
                ),
                types.SimpleNamespace(
                    kv_cache_spec=types.SimpleNamespace(
                        block_size=8,
                        storage_block_size=8,
                        page_size_bytes=32,
                        kv_cache_specs={"state128": SlidingWindowMLASpec(128)},
                    ),
                    is_eagle_group=False,
                    layer_names=("state128",),
                ),
            )
        )

    @staticmethod
    def _windowed_connector(root: Path, window: int):
        return _make_connector(
            root,
            0,
            extra_config={
                "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                "spark_cache_min_span_tokens": "256",
            },
            tp=2,
            dcp=1,
            kv_cache_config=HybridReuseWindowTests._windowed_config(window),
        )

    def test_only_the_reusable_tail_is_selected_at_the_persistent_boundary(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_model_profile": "deepseek-v4-fp8-hma",
                    "spark_cache_min_span_tokens": "256",
                },
                tp=2,
                dcp=1,
                kv_cache_config=_hybrid_kv_cache_config(),
            )

        self.assertEqual(
            connector._select_group_blocks_for_span(
                ((3, 5, 9), (4, 2, 1)),
                1024,
            ),
            ((3, 5), (2,)),
        )

    def test_non_aligned_sliding_window_excludes_the_current_token(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = self._windowed_connector(Path(directory), 65)

        self.assertEqual(
            connector._select_group_blocks_for_span(
                (
                    (1,),
                    (10, 11, 12, 13),
                    tuple(range(100, 164)),
                    tuple(range(200, 232)),
                ),
                256,
            )[1],
            (13,),
        )

    def test_null_or_duplicate_blocks_in_selected_window_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = self._windowed_connector(Path(directory), 128)

        for unsafe_swa in ((10, 11, 0, 13), (10, 11, 13, 13)):
            with self.subTest(unsafe_swa=unsafe_swa):
                with self.assertRaises(HybridCodecError):
                    connector._select_group_blocks_for_span(
                        (
                            (1,),
                            unsafe_swa,
                            tuple(range(100, 164)),
                            tuple(range(200, 232)),
                        ),
                        256,
                    )

    def test_unknown_and_non_align_mamba_specs_are_rejected_at_startup(self) -> None:
        class MambaSpec:
            block_size = 4
            mamba_cache_mode = "all"

        for bad_spec, message in (
            (types.SimpleNamespace(block_size=4), "unsupported block-page"),
            (MambaSpec(), "requires mamba_cache_mode 'align'"),
        ):
            config = self._windowed_config(128)
            config.kv_cache_groups[2].kv_cache_spec.kv_cache_specs["state"] = bad_spec
            with self.subTest(spec=type(bad_spec).__name__):
                with tempfile.TemporaryDirectory() as directory:
                    with self.assertRaisesRegex(RuntimeError, message):
                        _make_connector(
                            Path(directory),
                            0,
                            extra_config={
                                "spark_cache_model_profile": "deepseek-v4-fp8-hma"
                            },
                            tp=2,
                            dcp=1,
                            kv_cache_config=config,
                        )

    def test_mamba_align_selects_only_the_boundary_state(self) -> None:
        class MambaSpec:
            block_size = 4
            storage_block_size = 4
            page_size_bytes = 32
            mamba_cache_mode = "align"
            tokens_per_state = 4
            num_speculative_blocks = 0
            num_prefill_checkpoint_blocks = 0

        config = self._windowed_config(128)
        config.kv_cache_groups[2].kv_cache_spec.kv_cache_specs["state"] = MambaSpec()
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_model_profile": "glm53-flash-hybrid",
                    "spark_cache_draft_checkpoint_sha256": "2" * 64,
                },
                tp=2,
                dcp=1,
                kv_cache_config=config,
            )

        groups = (
            (1,),
            (10, 11, 12, 13),
            tuple([0] * 63 + [163]),
            tuple(range(200, 232)),
        )
        self.assertEqual(
            connector._select_group_blocks_for_span(groups, 256)[2],
            (163,),
        )

        unsafe = list(groups)
        unsafe[2] = tuple([0] * 64)
        with self.assertRaisesRegex(HybridCodecError, "null block"):
            connector._select_group_blocks_for_span(tuple(unsafe), 256)

    def test_mamba_align_checkpoint_round_trips_with_attention_pages(self) -> None:
        class MambaSpec:
            block_size = 4
            storage_block_size = 4
            page_size_bytes = 32
            mamba_cache_mode = "align"
            tokens_per_state = 4
            num_speculative_blocks = 0
            num_prefill_checkpoint_blocks = 0

        config = self._windowed_config(128)
        config.kv_cache_groups[2].kv_cache_spec.kv_cache_specs["state"] = MambaSpec()
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                extra_config={
                    "spark_cache_model_profile": "glm53-flash-hybrid",
                    "spark_cache_draft_checkpoint_sha256": "2" * 64,
                },
                tp=2,
                dcp=1,
                kv_cache_config=config,
            )
            pools = {
                "full": torch.arange(8 * 64, dtype=torch.uint8).reshape(8, 1, 64),
                "swa": torch.arange(16 * 64, dtype=torch.uint8).reshape(16, 1, 64),
                "state": torch.arange(100 * 8, dtype=torch.float32).reshape(100, 1, 8),
                "state128": torch.arange(80 * 8, dtype=torch.float32).reshape(80, 1, 8),
            }
            connector.register_kv_caches(pools)
            source_groups = (
                (3,),
                (5, 6, 7, 8),
                tuple([0] * 63 + [63]),
                tuple(range(1, 33)),
            )
            destination_groups = (
                (4,),
                (9, 10, 11, 12),
                tuple([0] * 63 + [79]),
                tuple(range(33, 65)),
            )
            source_selected = connector._select_group_blocks_for_span(
                source_groups, 256
            )
            expected = {}
            for group, selected in zip(config.kv_cache_groups, source_selected):
                for name in group.layer_names:
                    expected[name] = pools[name][list(selected)].clone()
            store = _ReqPlan(
                "store",
                "d" * 64,
                256,
                source_groups[0],
                True,
                block_ids_by_group=source_groups,
            )
            connector._store_one(store)
            for tensor in pools.values():
                tensor.zero_()
            load = _ReqPlan(
                "load",
                store.digest,
                store.span_tokens,
                destination_groups[0],
                False,
                block_ids_by_group=destination_groups,
            )
            self.assertTrue(connector._load_one(load))
            destination_selected = connector._select_group_blocks_for_span(
                destination_groups, 256
            )
            for group, selected in zip(config.kv_cache_groups, destination_selected):
                for name in group.layer_names:
                    self.assertTrue(
                        torch.equal(pools[name][list(selected)], expected[name]),
                        name,
                    )

    def test_sliding_and_recurrent_groups_select_their_declared_windows(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = self._windowed_connector(Path(directory), 128)

        self.assertEqual(
            connector._select_group_blocks_for_span(
                (
                    (1,),
                    (10, 11, 12, 13),
                    tuple(range(100, 164)),
                    tuple(range(200, 232)),
                ),
                256,
            ),
            (
                (1,),
                (12, 13),
                (162, 163),
                tuple(range(216, 232)),
            ),
        )

    def test_reuse_window_geometry_forks_identity_and_old_entry_misses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = self._windowed_connector(root, 128)
            pools = {
                "full": torch.arange(8 * 1 * 64, dtype=torch.uint8).reshape(8, 1, 64),
                "swa": torch.arange(8 * 1 * 64, dtype=torch.uint8).reshape(8, 1, 64),
                "state": torch.arange(80 * 1 * 8, dtype=torch.float32).reshape(
                    80, 1, 8
                ),
                "state128": torch.arange(40 * 1 * 8, dtype=torch.float32).reshape(
                    40, 1, 8
                ),
            }
            first.register_kv_caches(pools)
            plan = _ReqPlan(
                "window-128",
                "c" * 64,
                256,
                (3,),
                True,
                block_ids_by_group=(
                    (3,),
                    (1, 2, 3, 4),
                    tuple(range(64)),
                    tuple(range(1, 33)),
                ),
            )
            first._store_one(plan)
            replacement = self._windowed_connector(root, 256)

            self.assertNotEqual(
                first._identity(0).storage_key,
                replacement._identity(0).storage_key,
            )
            lookup = replacement._store.lookup(
                replacement._identity(0),
                plan.digest,
                verify_chunks=False,
            )
            self.assertFalse(lookup.is_hit)


class DefectD14HybridSchedulerGeometryTests(unittest.TestCase):
    """D-14: a scheduler block may contain multiple storage chunks."""

    def test_resolved_scheduler_block_size_accepts_exact_chunk_multiple(self) -> None:
        resolve_profile("glm53-flash-hybrid").validate_for_deployment(
            dcp_degree=1,
            block_size=2304,
            min_span_tokens=4096,
            cuda_restore=False,
        )


class DefectD15HybridBlockDeltaTests(unittest.TestCase):
    """D-15: a hybrid allocation delta may be empty for one cache group."""

    def test_empty_group_delta_preserves_the_complete_request_table(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, block_size=64)
            connector._store_progress["hybrid"] = (
                "a" * 64,
                1024,
                256,
                [[10], [20], [30]],
            )
            step = types.SimpleNamespace(
                scheduled_new_reqs=[],
                num_scheduled_tokens={"hybrid": 256},
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=["hybrid"],
                    resumed_req_ids=set(),
                    num_computed_tokens=[256],
                    new_block_ids=[([11], [], [31])],
                ),
            )

            metadata = connector.build_connector_meta(step)

            self.assertEqual(metadata.plans, [])
            self.assertEqual(
                connector._store_progress["hybrid"][3],
                [[10, 11], [20], [30, 31]],
            )


class DefectD16HybridPageBoundaryTests(unittest.TestCase):
    """D-16: page deltas replace a partial terminal HMA page."""

    def test_glm_page_delta_extends_a_chunk_boundary_inside_an_hma_page(
        self,
    ) -> None:
        class FullAttentionSpec:
            block_size = 256
            storage_block_size = 256
            page_size_bytes = 1024

        class MambaSpec:
            block_size = 2304
            storage_block_size = 2304
            page_size_bytes = 1024
            mamba_cache_mode = "align"
            tokens_per_state = 2304
            num_speculative_blocks = 0
            num_prefill_checkpoint_blocks = 1

        config = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
                types.SimpleNamespace(
                    kv_cache_spec=MambaSpec(),
                    is_eagle_group=False,
                    layer_names=("recurrent",),
                ),
            )
        )
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                extra_config={
                    "spark_cache_model_profile": "glm53-flash-hybrid",
                    "spark_cache_publication_schema": "tail-cow-v1",
                },
                tp=1,
                dcp=1,
                kv_cache_config=config,
            )
            pools = {
                name: (
                    torch.arange(256 * 1024, dtype=torch.int32)
                    .add(offset)
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape(256, 1, 1024)
                )
                for name, offset in (("full", 0), ("recurrent", 37))
            }
            connector.register_kv_caches(pools)
            tokens = tuple(range(12032))
            base_span = 7168
            result_span = 12032
            base_digest = connector._digest(list(tokens), base_span)
            result_digest = connector._digest(list(tokens), result_span)
            base_groups = (tuple(range(1, 29)), (70, 71, 72, 73))
            result_groups = (tuple(range(1, 48)), (70, 71, 72, 73, 74, 75))

            connector._store_one(
                _ReqPlan(
                    "glm-base-store",
                    base_digest,
                    base_span,
                    base_groups[0],
                    True,
                    block_ids_by_group=base_groups,
                    token_ids=tokens[:base_span],
                )
            )
            expected_base = {
                "full": pools["full"][list(base_groups[0])].clone(),
                "recurrent": pools["recurrent"][[base_groups[1][-1]]].clone(),
            }
            base_destination = (tuple(range(128, 156)), (160, 161, 162, 163))
            pools["full"][list(base_destination[0])].zero_()
            pools["recurrent"][base_destination[1][-1]].zero_()
            self.assertTrue(
                connector._load_one(
                    _ReqPlan(
                        "glm-base-restore",
                        base_digest,
                        base_span,
                        base_destination[0],
                        False,
                        block_ids_by_group=base_destination,
                    )
                )
            )
            self.assertTrue(
                torch.equal(
                    pools["full"][list(base_destination[0])],
                    expected_base["full"],
                )
            )
            self.assertTrue(
                torch.equal(
                    pools["recurrent"][[base_destination[1][-1]]],
                    expected_base["recurrent"],
                )
            )

            expected_result = {
                "full": pools["full"][list(result_groups[0])].clone(),
                "recurrent": pools["recurrent"][[result_groups[1][-1]]].clone(),
            }
            extension_plan = _ReqPlan(
                "glm-extension-store",
                result_digest,
                result_span,
                result_groups[0],
                True,
                block_ids_by_group=result_groups,
                token_ids=tokens,
                base_context_digest=base_digest,
                base_span_tokens=base_span,
            )
            result_snapshot = connector._snapshot_hybrid_store(extension_plan)
            connector._store_one(extension_plan)

            lookup = connector._store.lookup(connector._identity(0), result_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.root_kind, "page_delta")
            manifest = lookup._manifest
            self.assertIsNotNone(manifest)
            assert manifest is not None
            self.assertEqual(manifest["base_block_counts"], [28, 1])
            self.assertEqual(manifest["result_block_counts"], [47, 1])
            encoded_delta = connector._store._read_page_delta_objects(
                manifest["delta_objects"],
                encoded_bytes=manifest["delta_encoded_bytes"],
                encoded_sha256=manifest["delta_sha256"],
            )
            self.assertLess(
                len(encoded_delta),
                len(result_snapshot.encoded_pages),
            )
            result_destination = (
                tuple(range(176, 223)),
                (230, 231, 232, 233, 234, 235),
            )
            pools["full"][list(result_destination[0])].zero_()
            pools["recurrent"][result_destination[1][-1]].zero_()
            self.assertTrue(
                connector._load_one(
                    _ReqPlan(
                        "glm-extension-restore",
                        result_digest,
                        result_span,
                        result_destination[0],
                        False,
                        block_ids_by_group=result_destination,
                    )
                )
            )
            self.assertTrue(
                torch.equal(
                    pools["full"][list(result_destination[0])],
                    expected_result["full"],
                )
            )
            self.assertTrue(
                torch.equal(
                    pools["recurrent"][[result_destination[1][-1]]],
                    expected_result["recurrent"],
                )
            )


class DefectD17RecurrentBoundaryMetadataTests(unittest.TestCase):
    """D-17: vLLM identifies off-table recurrent replay boundaries."""

    BOUNDARY = 6912
    PROMPT_TOKENS = 6992
    NONALIGNED_BOUNDARY = 8192
    NONALIGNED_PROMPT_TOKENS = 8256
    BOUNDARY_BLOCK = 42
    COW_BLOCK = 142

    @staticmethod
    def _config() -> types.SimpleNamespace:
        class FullAttentionSpec:
            block_size = 2304
            storage_block_size = 2304
            page_size_bytes = 64

        class MambaSpec:
            block_size = 2304
            storage_block_size = 2304
            page_size_bytes = 64
            mamba_cache_mode = "align"
            tokens_per_state = 2304
            num_speculative_blocks = 7
            num_prefill_checkpoint_blocks = 0

        return types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=FullAttentionSpec(),
                    is_eagle_group=False,
                    layer_names=("full",),
                ),
                types.SimpleNamespace(
                    kv_cache_spec=MambaSpec(),
                    is_eagle_group=False,
                    layer_names=("recurrent",),
                ),
            )
        )

    @classmethod
    def _tables(cls) -> tuple[tuple[int, ...], ...]:
        # The recurrent replay-boundary slot is null after vLLM advances the
        # running state. Entries 71..78 are the later running state and seven
        # DFlash verification slots, so none can substitute for block 42.
        return ((11, 12, 13, 14), (0, 0, 0, 71, 72, 73, 74, 75, 76, 77, 78))

    @classmethod
    def _scheduler_output(
        cls,
        recurrent_boundary_blocks: object = None,
    ) -> types.SimpleNamespace:
        request_id = "dflash-recurrent-boundary"
        output = types.SimpleNamespace(
            scheduled_new_reqs=[
                types.SimpleNamespace(
                    req_id=request_id,
                    prompt_token_ids=list(range(cls.PROMPT_TOKENS)),
                    block_ids=cls._tables(),
                    num_computed_tokens=0,
                )
            ],
            scheduled_cached_reqs=types.SimpleNamespace(
                req_ids=[],
                resumed_req_ids=set(),
                num_computed_tokens=[],
                new_block_ids=[],
            ),
            num_scheduled_tokens={request_id: cls.PROMPT_TOKENS},
            preempted_req_ids=set(),
        )
        if recurrent_boundary_blocks is not None:
            output.recurrent_boundary_blocks = recurrent_boundary_blocks
        return output

    @classmethod
    def _cached_scheduler_output(
        cls,
        *,
        num_computed_tokens: int,
        recurrent_boundary_blocks: object = None,
        group_count: int = 2,
        num_scheduled_tokens: int = 1,
        resumed: bool = False,
        new_block_ids: object = None,
    ) -> types.SimpleNamespace:
        request_id = "dflash-recurrent-boundary"
        output = types.SimpleNamespace(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=types.SimpleNamespace(
                req_ids=[request_id],
                resumed_req_ids={request_id} if resumed else set(),
                num_computed_tokens=[num_computed_tokens],
                new_block_ids=[
                    new_block_ids
                    if new_block_ids is not None
                    else tuple(() for _ in range(group_count))
                ],
            ),
            num_scheduled_tokens={request_id: num_scheduled_tokens},
            preempted_req_ids=set(),
        )
        if recurrent_boundary_blocks is not None:
            output.recurrent_boundary_blocks = recurrent_boundary_blocks
        return output

    def test_explicit_boundary_block_round_trips_through_manifest_store(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self._config()
            scheduler = _make_connector(
                root,
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=config,
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )
            boundary_metadata = {
                "dflash-recurrent-boundary": [
                    (1, self.BOUNDARY_BLOCK, self.BOUNDARY)
                ]
            }
            first_metadata = scheduler.build_connector_meta(
                self._scheduler_output(boundary_metadata)
            )
            self.assertEqual(first_metadata.plans, [])
            self.assertIn("dflash-recurrent-boundary", scheduler._store_progress)
            self.assertEqual(
                scheduler._store_recurrent_boundaries[
                    "dflash-recurrent-boundary"
                ],
                ((1, self.BOUNDARY_BLOCK),),
            )
            output = self._cached_scheduler_output(
                num_computed_tokens=self.PROMPT_TOKENS,
            )

            self.assertTrue(scheduler.supports_recurrent_boundary_blocks)
            metadata = scheduler.build_connector_meta(output)

            self.assertEqual(len(metadata.plans), 1)
            plan = metadata.plans[0]
            self.assertEqual(plan.span_tokens, self.BOUNDARY)
            self.assertEqual(plan.recurrent_boundary_blocks, ((1, 42),))
            worker = _make_connector(
                root,
                0,
                block_size=256,
                tp=1,
                dcp=1,
                kv_cache_config=config,
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )
            pools = {
                name: (
                    torch.arange(128 * 64, dtype=torch.int32)
                    .add(offset)
                    .remainder(251)
                    .to(torch.uint8)
                    .reshape(128, 1, 64)
                )
                for name, offset in (("full", 0), ("recurrent", 29))
            }
            worker.register_kv_caches(pools)
            expected_full = pools["full"][[11, 12, 13]].clone()
            expected_recurrent = pools["recurrent"][[self.BOUNDARY_BLOCK]].clone()

            worker._store_one(plan)

            lookup = worker._store.lookup(worker._identity(0), plan.digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            destination = (
                (90, 91, 92),
                (0, 0, 93, 94, 95, 96, 97, 98, 99, 100, 101),
            )
            pools["full"][[90, 91, 92]].zero_()
            pools["recurrent"][[93]].zero_()
            self.assertTrue(
                worker._load_one(
                    _ReqPlan(
                        "dflash-recurrent-restore",
                        plan.digest,
                        self.BOUNDARY,
                        destination[0],
                        False,
                        block_ids_by_group=destination,
                    )
                )
            )
            self.assertTrue(torch.equal(pools["full"][[90, 91, 92]], expected_full))
            self.assertTrue(
                torch.equal(pools["recurrent"][[93]], expected_recurrent)
            )

    def test_nonaligned_boundary_waits_for_partial_tail_cow_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=self._config(),
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )
            output = self._scheduler_output()
            request = output.scheduled_new_reqs[0]
            request.prompt_token_ids = list(range(self.NONALIGNED_PROMPT_TOKENS))
            output.num_scheduled_tokens[request.req_id] = (
                self.NONALIGNED_PROMPT_TOKENS
            )

            first_metadata = connector.build_connector_meta(output)
            self.assertEqual(first_metadata.plans, [])
            self.assertIn(request.req_id, connector._store_progress)
            pending = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.NONALIGNED_PROMPT_TOKENS,
                )
            )
            self.assertEqual(pending.plans, [])
            self.assertIn(request.req_id, connector._store_progress)
            metadata = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.NONALIGNED_PROMPT_TOKENS + 1,
                    recurrent_boundary_blocks={
                        request.req_id: [
                            (1, self.COW_BLOCK, self.NONALIGNED_BOUNDARY)
                        ]
                    },
                )
            )

            self.assertEqual(len(metadata.plans), 1)
            plan = metadata.plans[0]
            self.assertEqual(plan.span_tokens, self.NONALIGNED_BOUNDARY)
            self.assertEqual(plan.recurrent_boundary_blocks, ((1, self.COW_BLOCK),))
            self.assertEqual(
                connector._select_group_blocks_for_span(
                    plan.block_ids_by_group,
                    plan.span_tokens,
                    recurrent_boundary_blocks=plan.recurrent_boundary_blocks,
                ),
                ((11, 12, 13, 14), (self.COW_BLOCK,)),
            )
            self.assertEqual(
                connector.counters["recurrent_boundary_metadata_rejected"],
                0,
            )

    def test_conflicting_mapping_poisons_latched_publication(self) -> None:
        output = self._scheduler_output(
            {
                "dflash-recurrent-boundary": [
                    (1, self.BOUNDARY_BLOCK, self.BOUNDARY)
                ]
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=self._config(),
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )

            first_metadata = connector.build_connector_meta(output)
            self.assertEqual(first_metadata.plans, [])
            metadata = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.NONALIGNED_PROMPT_TOKENS,
                    recurrent_boundary_blocks={
                        "dflash-recurrent-boundary": [
                            (1, self.BOUNDARY_BLOCK + 1, self.BOUNDARY)
                        ]
                    },
                )
            )

            self.assertEqual(metadata.plans, [])
            self.assertEqual(
                connector.counters["recurrent_boundary_metadata_rejected"],
                1,
            )

    def test_chunked_prefill_validates_only_at_publication_step(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=self._config(),
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )
            first = self._scheduler_output()
            first.num_scheduled_tokens["dflash-recurrent-boundary"] = 2304

            self.assertEqual(connector.build_connector_meta(first).plans, [])
            middle = self._cached_scheduler_output(
                num_computed_tokens=2304,
                num_scheduled_tokens=2304,
            )
            self.assertEqual(connector.build_connector_meta(middle).plans, [])
            self.assertIn("dflash-recurrent-boundary", connector._store_progress)
            self.assertEqual(
                connector.counters["recurrent_boundary_metadata_rejected"],
                0,
            )
            final = self._cached_scheduler_output(
                num_computed_tokens=4608,
                num_scheduled_tokens=2304,
                recurrent_boundary_blocks={
                    "dflash-recurrent-boundary": [
                        (1, self.BOUNDARY_BLOCK, self.BOUNDARY)
                    ]
                },
            )

            metadata = connector.build_connector_meta(final)

            self.assertEqual(len(metadata.plans), 1)
            self.assertEqual(
                metadata.plans[0].recurrent_boundary_blocks,
                ((1, self.BOUNDARY_BLOCK),),
            )
            self.assertNotIn("dflash-recurrent-boundary", connector._store_progress)

    def test_missing_or_wrong_request_metadata_keeps_publication_pending(self) -> None:
        for boundary_metadata in (
            None,
            {"another-request": [(1, self.BOUNDARY_BLOCK, self.BOUNDARY)]},
        ):
            with self.subTest(boundary_metadata=boundary_metadata):
                with tempfile.TemporaryDirectory() as directory:
                    connector = _make_connector(
                        Path(directory),
                        0,
                        block_size=256,
                        role=KVConnectorRole.SCHEDULER,
                        override_worker_rank=False,
                        tp=1,
                        dcp=1,
                        kv_cache_config=self._config(),
                        extra_config={
                            "spark_cache_model_profile": "glm53-flash-hybrid"
                        },
                    )
                    output = self._scheduler_output(boundary_metadata)
                    full, recurrent = output.scheduled_new_reqs[0].block_ids
                    recurrent = list(recurrent)
                    recurrent[2] = 69  # stale or recycled, not boundary-proven
                    output.scheduled_new_reqs[0].block_ids = (full, tuple(recurrent))
                    first_metadata = connector.build_connector_meta(output)
                    self.assertEqual(first_metadata.plans, [])
                    metadata = connector.build_connector_meta(
                        self._cached_scheduler_output(
                            num_computed_tokens=self.PROMPT_TOKENS,
                            recurrent_boundary_blocks=boundary_metadata,
                        )
                    )
                    self.assertEqual(metadata.plans, [])
                    self.assertEqual(
                        connector.counters[
                            "recurrent_boundary_metadata_rejected"
                        ],
                        0,
                    )
                    self.assertIn(
                        "dflash-recurrent-boundary", connector._store_progress
                    )
                    connector.request_finished(
                        types.SimpleNamespace(
                            request_id="dflash-recurrent-boundary"
                        ),
                        [],
                    )
                    self.assertNotIn(
                        "dflash-recurrent-boundary", connector._store_progress
                    )
                    self.assertNotIn(
                        "dflash-recurrent-boundary",
                        connector._store_recurrent_boundaries,
                    )

    def test_contradictory_boundary_metadata_skips_publication(self) -> None:
        invalid_entries = (
            [],
            [(1, self.BOUNDARY_BLOCK, self.BOUNDARY - 256)],
            [(2, self.BOUNDARY_BLOCK, self.BOUNDARY)],
            [(1, self.BOUNDARY_BLOCK, self.BOUNDARY), (1, 43, self.BOUNDARY)],
            [(1, 0, self.BOUNDARY)],
            [(0, self.BOUNDARY_BLOCK, self.BOUNDARY)],
        )
        for entries in invalid_entries:
            with self.subTest(entries=entries):
                with tempfile.TemporaryDirectory() as directory:
                    connector = _make_connector(
                        Path(directory),
                        0,
                        block_size=256,
                        role=KVConnectorRole.SCHEDULER,
                        override_worker_rank=False,
                        tp=1,
                        dcp=1,
                        kv_cache_config=self._config(),
                        extra_config={
                            "spark_cache_model_profile": "glm53-flash-hybrid"
                        },
                    )
                    first_metadata = connector.build_connector_meta(
                        self._scheduler_output()
                    )
                    self.assertEqual(first_metadata.plans, [])
                    metadata = connector.build_connector_meta(
                        self._cached_scheduler_output(
                            num_computed_tokens=self.PROMPT_TOKENS,
                            recurrent_boundary_blocks={
                                "dflash-recurrent-boundary": entries
                            },
                        )
                    )
                    self.assertEqual(metadata.plans, [])
                    self.assertEqual(
                        connector.counters[
                            "recurrent_boundary_metadata_rejected"
                        ],
                        1,
                    )

    def test_partial_recurrent_group_coverage_skips_publication(self) -> None:
        config = self._config()
        second_recurrent = types.SimpleNamespace(
            kv_cache_spec=config.kv_cache_groups[1].kv_cache_spec,
            is_eagle_group=False,
            layer_names=("recurrent-2",),
        )
        config.kv_cache_groups = (*config.kv_cache_groups, second_recurrent)
        output = self._scheduler_output()
        output.scheduled_new_reqs[0].block_ids = (
            *self._tables(),
            (0, 0, 0, 81, 82, 83, 84, 85, 86, 87, 88),
        )
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=config,
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )

            first_metadata = connector.build_connector_meta(output)
            self.assertEqual(first_metadata.plans, [])
            metadata = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.PROMPT_TOKENS,
                    recurrent_boundary_blocks={
                        "dflash-recurrent-boundary": [(1, 42, self.BOUNDARY)]
                    },
                    group_count=3,
                )
            )

            self.assertEqual(metadata.plans, [])
            self.assertEqual(
                connector.counters["recurrent_boundary_metadata_rejected"],
                1,
            )

    def test_preemption_discards_request_lifetime_boundary_block(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=self._config(),
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )
            request_id = "dflash-recurrent-boundary"
            connector.build_connector_meta(
                self._scheduler_output(
                    {
                        request_id: [
                            (1, self.BOUNDARY_BLOCK, self.BOUNDARY)
                        ]
                    }
                )
            )
            self.assertIn(request_id, connector._store_recurrent_boundaries)
            output = types.SimpleNamespace(
                scheduled_new_reqs=[],
                scheduled_cached_reqs=types.SimpleNamespace(
                    req_ids=[],
                    resumed_req_ids=set(),
                    num_computed_tokens=[],
                    new_block_ids=[],
                ),
                num_scheduled_tokens={},
                preempted_req_ids={request_id},
            )

            metadata = connector.build_connector_meta(output)

            self.assertEqual(metadata.preempted_request_ids, (request_id,))
            self.assertNotIn(request_id, connector._store_recurrent_boundaries)
            self.assertIn(request_id, connector._store_progress)

            resumed = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.PROMPT_TOKENS,
                    resumed=True,
                    new_block_ids=self._tables(),
                )
            )
            self.assertEqual(resumed.plans, [])
            self.assertNotIn(request_id, connector._store_recurrent_boundaries)
            published = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.PROMPT_TOKENS + 1,
                    recurrent_boundary_blocks={
                        request_id: [
                            (1, self.BOUNDARY_BLOCK + 10, self.BOUNDARY)
                        ]
                    },
                )
            )
            self.assertEqual(len(published.plans), 1)
            self.assertEqual(
                published.plans[0].recurrent_boundary_blocks,
                ((1, self.BOUNDARY_BLOCK + 10),),
            )

    def test_quorum_retires_pending_store_without_boundary_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(
                Path(directory),
                0,
                block_size=256,
                role=KVConnectorRole.SCHEDULER,
                override_worker_rank=False,
                tp=1,
                dcp=1,
                kv_cache_config=self._config(),
                extra_config={"spark_cache_model_profile": "glm53-flash-hybrid"},
            )
            request_id = "dflash-recurrent-boundary"
            connector.build_connector_meta(self._scheduler_output())
            digest = connector._store_progress[request_id][0]
            connector._quorum[digest] = {0}

            metadata = connector.build_connector_meta(
                self._cached_scheduler_output(
                    num_computed_tokens=self.PROMPT_TOKENS,
                    recurrent_boundary_blocks="malformed-but-unneeded",
                )
            )

            self.assertEqual(metadata.plans, [])
            self.assertNotIn(request_id, connector._store_progress)
            self.assertNotIn(request_id, connector._store_recurrent_boundaries)
            self.assertEqual(connector.counters["store_skipped_quorum"], 1)
            self.assertEqual(
                connector.counters["recurrent_boundary_metadata_rejected"],
                0,
            )


class DigestNamespaceTests(unittest.TestCase):
    """D-4: context digests are identical across roles and physical ranks."""

    def test_worker_rank_does_not_fork_the_digest(self) -> None:
        tokens = list(range(1100))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            scheduler = _make_connector(root, 0, 64, role=KVConnectorRole.SCHEDULER)
            workers = [
                _make_connector(root, rank, 64, role=KVConnectorRole.WORKER)
                for rank in range(4)
            ]
            digests = {c._digest(tokens, 1024) for c in [scheduler, *workers]}
            self.assertEqual(len(digests), 1)


class StoreSpanCompletenessTests(unittest.TestCase):
    """D-7: a truncated chunk sequence must fail the commit, not publish."""

    def test_truncated_store_commit_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("truncated", "e" * 64, 1024, (3, 0, 5, 1), True)
            snapshot = connector._snapshot_store(plan)
            from sparkcache.spark_context_cache_connector import _SnapshotChunks

            complete = list(_SnapshotChunks(snapshot, connector._dcp_degree))
            with self.assertRaises(IncompleteEntry):
                connector._store.commit(
                    identity=snapshot.identity,
                    context_digest=plan.digest,
                    chunks=complete[:-1],
                    span_tokens=plan.span_tokens,
                )
            # the failed commit must not publish a visible manifest
            self.assertFalse(
                connector._store.lookup(
                    snapshot.identity, plan.digest, verify_chunks=False
                ).is_hit
            )


class InvalidManifestMaintenanceTests(unittest.TestCase):
    """D-13: maintenance removes manifests that restore must reject."""

    def test_restore_invalid_manifest_and_its_unshared_chunk_are_removed(self) -> None:
        identity = package_store.CacheIdentity(
            target_checkpoint="1" * 64,
            draft_checkpoint="2" * 64,
            quantization_layout="nvfp4-ds-mla-v1",
            rope_layout="glm52-bf16-rope-v1",
            tp_degree=1,
            dcp_degree=1,
        )
        chunk = package_store.ContextChunk(
            logical_start=0,
            logical_end=256,
            records={
                record: record.value.encode() for record in package_store.StateRecord
            },
        )
        context_digest = hashlib.sha256(b"D-13-invalid-manifest").hexdigest()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = package_store.ManifestStore(root)
            store.commit(
                identity=identity,
                context_digest=context_digest,
                chunks=[chunk],
            )
            manifest_path = (
                root / "manifests" / identity.storage_key / f"{context_digest}.json"
            )
            manifest = json.loads(manifest_path.read_bytes())
            manifest["committed_tokens"] = 0
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
            chunk_path = next((root / "chunks").glob("*.spcc"))

            lookup = store.lookup(identity, context_digest, verify_chunks=False)
            self.assertFalse(lookup.is_hit)
            self.assertEqual(lookup.reason, "corrupt")

            report = store.maintain(package_store.CapacityPolicy(ttl_seconds=3600))

            self.assertEqual(report.manifests_evicted, 1)
            self.assertEqual(report.chunks_deleted, 1)
            self.assertFalse(manifest_path.exists())
            self.assertFalse(chunk_path.exists())


class FailureInvalidationTests(unittest.TestCase):
    """D-8: load failure cleanup stays off the request critical path."""

    def test_failure_invalidation_does_not_rehash_chunk_payloads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            digest = "8" * 64
            connector._held.add(digest)
            connector._store.invalidate = mock.Mock(return_value=True)

            connector._invalidate_after_failure(digest)

            connector._store.invalidate.assert_called_once_with(
                connector._identity(0),
                digest,
                verify_chunk_payloads=False,
            )
            self.assertNotIn(digest, connector._held)

    def test_republish_repairs_a_corrupt_content_addressed_chunk(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connector = _make_connector(root, 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            plan = _ReqPlan("repair", "9" * 64, 1024, (3, 0, 5, 1), True)
            connector._store_one(plan)
            connector._held.add(plan.digest)
            chunk_path = next((root / "chunks").glob("*.spcc"))
            damaged = bytearray(chunk_path.read_bytes())
            damaged[len(damaged) // 2] ^= 0x40
            chunk_path.write_bytes(damaged)

            connector._invalidate_after_failure(plan.digest)

            self.assertTrue(chunk_path.exists())
            connector._store_one(plan)
            lookup = connector._store.lookup(connector._identity(0), plan.digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertIsNotNone(connector._store.restore(lookup))


class IntegritySweepPublicationTests(unittest.TestCase):
    """D-8: integrity sweeps preserve entries published while scanning."""

    def test_sweep_merges_a_concurrent_publication_into_held_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector(Path(directory), 0, 64)
            connector.register_kv_caches(_make_pools(8, 64))
            existing = _ReqPlan("existing", "a" * 64, 1024, (3, 0, 5, 1), True)
            connector._store_one(existing)
            connector._held.add(existing.digest)
            published_during_sweep = "b" * 64
            restore = connector._store.restore

            def restore_after_publication(lookup):
                connector._finish_store(published_during_sweep, committed=True)
                return restore(lookup)

            connector._store.restore = restore_after_publication

            self.assertEqual(
                connector.sweep_integrity(),
                {"checked": 1, "invalidated": 0},
            )
            self.assertEqual(
                connector._held,
                {existing.digest, published_during_sweep},
            )


if __name__ == "__main__":
    unittest.main()
