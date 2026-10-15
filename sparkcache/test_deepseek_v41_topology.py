"""DeepSeek-V4.1-Flash cache-group conformance on the release 2026.10.4 vLLM line.

The fixture reproduces the V4.1 cache classes inferred from the model's spec
declarations: compressed-latent and indexer-key groups at compression ratios
2 and 1, prefix-cacheable encoder sliding-window groups, request-private
decoder and draft sliding-window groups (prefill replay reserve 128), and
per-request compressor rings. One group per cache class is a conformance
fixture built from vLLM's spec field names, not a hardware dump; replace its
grouping with the recorded topology inventory before hardware qualification.
"""

from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path

import torch

from sparkcache.request_cache_scope import UNSALTED_SCOPE
from sparkcache.spark_context_cache_profiles import (
    ProfileError,
    resolve_profile,
)
from sparkcache.test_spark_context_cache_config import _make_vllm_config
from sparkcache.test_spark_context_cache_connector import (
    _ReqPlan,
    _deepseek_tp4_hma_config,
    _make_connector,
    SparkContextCacheConnector,
)

REPLAY_RESERVE = 128
CHUNK_TOKENS = 256


class _EngineSpec:
    """Fake with vLLM's KVCacheSpec property and field names."""

    block_size = 256
    storage_block_size = 256
    page_size_bytes = 0
    tokens_per_state = 1

    def __init__(
        self,
        *,
        prefix_cacheable: bool = True,
        prefill_replay_tokens: int = 0,
        block_size: int = 256,
        page_size_bytes: int = 0,
        tokens_per_state: int = 1,
    ) -> None:
        self._prefix_cacheable = prefix_cacheable
        self._prefill_replay_tokens = prefill_replay_tokens
        self.block_size = block_size
        self.storage_block_size = block_size
        self.page_size_bytes = page_size_bytes
        self.tokens_per_state = tokens_per_state

    @property
    def prefix_cacheable(self) -> bool:
        return self._prefix_cacheable

    @property
    def prefill_replay_tokens(self) -> int:
        return self._prefill_replay_tokens


class FullAttentionSpec(_EngineSpec):
    """MRO-name stand-in for vLLM's attention spec base."""


class SlidingWindowSpec(_EngineSpec):
    """MRO-name stand-in for vLLM's sliding-window spec base."""

    sliding_window = 128


class LatentSpec(FullAttentionSpec):
    """Compressed-latent page: 288 bytes per state at the compression ratio."""


class IndexerSpec(FullAttentionSpec):
    """DSA indexer-key page: 68 bytes per state at the compression ratio."""


class EncoderSwaSpec(SlidingWindowSpec):
    """Prefix-cacheable encoder sliding-window page: 528 bytes per token."""

    def __init__(self, *, page_size_bytes: int) -> None:
        super().__init__(
            prefix_cacheable=True,
            block_size=128,
            page_size_bytes=page_size_bytes,
        )
        self.sliding_window = 128


class RequestPrivateSwaSpec(EncoderSwaSpec):
    """Decoder or draft sliding-window page that vLLM rebuilds from the
    replay reserve: request-private, never prefix-cacheable."""

    def __init__(self, *, page_size_bytes: int) -> None:
        super().__init__(page_size_bytes=page_size_bytes)
        self._prefix_cacheable = False
        self._prefill_replay_tokens = REPLAY_RESERVE


class CircularBufferSpec(FullAttentionSpec):
    """vLLM compressor ring: one in-flight block per request, never
    prefix-cacheable, no replay reserve."""

    def __init__(self, *, page_size_bytes: int) -> None:
        super().__init__(
            prefix_cacheable=False,
            block_size=8,
            page_size_bytes=page_size_bytes,
        )


def deepseek_v41_layout(*, small_pages: bool = False) -> types.SimpleNamespace:
    """One manager group per V4.1 cache class, in vLLM field names.

    ``small_pages`` swaps the declared page sizes for tiny ones so
    round-trip tests can allocate small pools; block sizes, policies, and
    state quanta keep the values the model declares.
    """

    def latent(ratio: int, layers: tuple[str, ...]) -> types.SimpleNamespace:
        page = 256 // ratio if small_pages else 288 * 128 // ratio
        return types.SimpleNamespace(
            kv_cache_spec=LatentSpec(tokens_per_state=ratio, page_size_bytes=page),
            is_eagle_group=False,
            layer_names=tuple(f"layers.{name}.latent" for name in layers),
        )

    def indexer(ratio: int, layers: tuple[str, ...]) -> types.SimpleNamespace:
        page = 128 // ratio if small_pages else 68 * 128 // ratio
        return types.SimpleNamespace(
            kv_cache_spec=IndexerSpec(tokens_per_state=ratio, page_size_bytes=page),
            is_eagle_group=False,
            layer_names=tuple(f"layers.{name}.indexer" for name in layers),
        )

    encoder_layers = tuple(str(layer) for layer in range(20))
    decoder_layers = tuple(str(layer) for layer in range(20, 40))
    draft_layers = tuple(str(layer) for layer in range(40, 43))
    ring_layers = ("2", "8", "14")
    swa_page = 64 if small_pages else 528 * 128
    ring_page = 32 if small_pages else 68 * 8
    groups = (
        latent(2, ring_layers),
        latent(1, ("20",)),
        indexer(2, ring_layers),
        indexer(1, ("20",)),
        types.SimpleNamespace(
            kv_cache_spec=EncoderSwaSpec(page_size_bytes=swa_page),
            is_eagle_group=False,
            layer_names=tuple(f"layers.{name}.swa" for name in encoder_layers),
        ),
        types.SimpleNamespace(
            kv_cache_spec=RequestPrivateSwaSpec(page_size_bytes=swa_page),
            is_eagle_group=False,
            layer_names=tuple(f"layers.{name}.swa" for name in decoder_layers),
        ),
        types.SimpleNamespace(
            kv_cache_spec=RequestPrivateSwaSpec(page_size_bytes=swa_page),
            is_eagle_group=True,
            layer_names=tuple(f"layers.{name}.swa" for name in draft_layers),
        ),
        types.SimpleNamespace(
            kv_cache_spec=CircularBufferSpec(page_size_bytes=ring_page),
            is_eagle_group=False,
            layer_names=tuple(f"layers.{name}.ring" for name in ring_layers),
        ),
    )
    return types.SimpleNamespace(kv_cache_groups=groups, num_blocks=512)


def span_tables(span: int) -> tuple[tuple[tuple[int, ...], ...], tuple[int, ...]]:
    """Request block tables covering ``span`` for the V4.1 fixture groups."""

    from sparkcache import spark_context_cache_config as cfg

    topology = cfg.kv_group_topology(deepseek_v41_layout(small_pages=True))
    tables = []
    for group in topology:
        width = int(group["logical_tokens_per_block"])
        required = (span + width - 1) // width
        # Non-persisted request tables stay short: a compressor ring owns one
        # block and a request-private window covers only computed rows.
        if group["reuse_policy"] in ("transient", "replay"):
            required = 1
        tables.append(tuple(range(1, required + 1)))
    return tuple(tables), tuple(int(group["logical_tokens_per_block"])
                                for group in topology)


def _v41_connector(
    root: Path, layout: types.SimpleNamespace
) -> SparkContextCacheConnector:
    return _make_connector(
        root,
        0,
        block_size=CHUNK_TOKENS,
        extra_config={
            "spark_cache_model_profile": "deepseek-v41-b12x-hma",
            "spark_cache_min_span_tokens": str(CHUNK_TOKENS),
            "spark_cache_draft_policy": "colocated_target",
            "spark_cache_draft_checkpoint_sha256": "",
            "spark_cache_publication_schema": "tail-cow-v2",
        },
        tp=1,
        dcp=1,
        kv_cache_config=layout,
    )


def _v41_identity(layout: types.SimpleNamespace, **extra: object) -> object:
    from sparkcache import spark_context_cache_config as cfg

    vllm, transfer = _make_vllm_config(
        {
            "spark_cache_model_profile": "deepseek-v41-b12x-hma",
            "spark_cache_draft_policy": "colocated_target",
            "spark_cache_draft_checkpoint_sha256": "",
            **extra,
        },
        tp=1,
        dcp=1,
        block_size=CHUNK_TOKENS,
    )
    return cfg.parse_connector_config(
        vllm, transfer, layout
    ).build_identity(0, 0)


class DeepSeekV41TopologyTests(unittest.TestCase):
    """Group classification and topology-digest behavior for V4.1."""

    def test_group_inventory_matches_v41_cache_classes(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        topology = cfg.kv_group_topology(deepseek_v41_layout())
        self.assertEqual(
            [group["reuse_policy"] for group in topology],
            ["full", "full", "full", "full", "sliding", "replay", "replay",
             "transient"],
        )
        self.assertEqual(
            [group["spec"] for group in topology],
            ["LatentSpec", "LatentSpec", "IndexerSpec", "IndexerSpec",
             "EncoderSwaSpec", "RequestPrivateSwaSpec",
             "RequestPrivateSwaSpec", "CircularBufferSpec"],
        )
        self.assertEqual(topology[-1]["block_size"], 8)
        self.assertEqual([group["eagle"] for group in topology][-2:], [True, False])

    def test_non_persisted_groups_record_replay_and_state_geometry(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        topology = cfg.kv_group_topology(deepseek_v41_layout())
        for group in topology:
            if group["reuse_policy"] in ("transient", "replay"):
                self.assertEqual(
                    group["replay_tokens"],
                    REPLAY_RESERVE if group["reuse_policy"] == "replay" else 0,
                )
                self.assertEqual(group["tokens_per_state"], 1)
            else:
                self.assertNotIn("replay_tokens", group)
                self.assertNotIn("tokens_per_state", group)

    def test_digest_forks_on_replay_tokens_state_geometry_and_policy(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        layout = deepseek_v41_layout()
        original = cfg.kv_group_topology_digest(layout)
        layout.kv_cache_groups[5].kv_cache_spec._prefill_replay_tokens = 64
        self.assertNotEqual(cfg.kv_group_topology_digest(layout), original)

        layout = deepseek_v41_layout()
        original = cfg.kv_group_topology_digest(layout)
        layout.kv_cache_groups[7].kv_cache_spec.tokens_per_state = 2
        layout.kv_cache_groups[7].kv_cache_spec.page_size_bytes = 68 * 4
        self.assertNotEqual(cfg.kv_group_topology_digest(layout), original)

        layout = deepseek_v41_layout()
        original = cfg.kv_group_topology_digest(layout)
        layout.kv_cache_groups[5].kv_cache_spec._prefix_cacheable = True
        layout.kv_cache_groups[5].kv_cache_spec._prefill_replay_tokens = 0
        topology = cfg.kv_group_topology(layout)
        self.assertEqual(topology[5]["reuse_policy"], "sliding")
        self.assertNotEqual(cfg.kv_group_topology_digest(layout), original)

    def test_unknown_non_prefix_cacheable_spec_is_refused(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        class KpoolTailSpec(FullAttentionSpec):
            def __init__(self) -> None:
                super().__init__(
                    prefix_cacheable=False, block_size=8, page_size_bytes=64
                )

        layout = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=KpoolTailSpec(),
                    is_eagle_group=False,
                    layer_names=("scratch",),
                ),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "unsupported block-page"):
            cfg.kv_group_topology(layout)

    def test_circular_buffer_requires_engine_private_properties(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        class CircularBufferSpec(FullAttentionSpec):
            def __init__(self) -> None:
                super().__init__(
                    prefix_cacheable=True, block_size=8, page_size_bytes=64
                )

        layout = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=CircularBufferSpec(),
                    is_eagle_group=False,
                    layer_names=("ring",),
                ),
            )
        )
        with self.assertRaisesRegex(RuntimeError, "circular-buffer"):
            cfg.kv_group_topology(layout)


class DeepSeekV41ProfileTests(unittest.TestCase):
    """The deepseek-v41-b12x-hma profile binds its cache contract."""

    def test_profile_fields_and_layout_names(self) -> None:
        profile = resolve_profile("deepseek-v41-b12x-hma")
        self.assertEqual(profile.storage_mode, "block_pages_v1")
        self.assertEqual(profile.chunk_tokens, CHUNK_TOKENS)
        self.assertEqual(profile.required_block_size, CHUNK_TOKENS)
        self.assertTrue(profile.requires_replay_group)
        self.assertFalse(profile.cuda_page_restore)
        self.assertTrue(profile.kv_replicated_across_tp)
        self.assertEqual(profile.default_draft_kv_policy, "colocated_target")
        self.assertEqual(
            profile.quantization_layout, "deepseek-v41-b12x-hma-block-pages-v1"
        )
        v4 = resolve_profile("deepseek-v4-fp8-hma")
        self.assertNotEqual(profile.quantization_layout, v4.quantization_layout)
        self.assertNotEqual(profile.rope_layout, v4.rope_layout)

    def test_profile_binds_block_size(self) -> None:
        profile = resolve_profile("deepseek-v41-b12x-hma")
        profile.validate_for_deployment(
            dcp_degree=1,
            block_size=CHUNK_TOKENS,
            min_span_tokens=CHUNK_TOKENS,
            cuda_restore=False,
        )
        with self.assertRaisesRegex(ProfileError, "block_size 512 must be 256"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=512,
                min_span_tokens=CHUNK_TOKENS,
                cuda_restore=False,
            )

    def test_other_profiles_refuse_non_persisted_groups(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        topology = cfg.kv_group_topology(deepseek_v41_layout())
        for name in ("qwen38-flash-next-hybrid", "deepseek-v4-fp8-hma"):
            profile = resolve_profile(name)
            with self.assertRaisesRegex(ProfileError, "non-persisted"):
                profile.validate_for_deployment(
                    dcp_degree=1,
                    block_size=CHUNK_TOKENS,
                    min_span_tokens=CHUNK_TOKENS,
                    cuda_restore=False,
                    group_topology=topology,
                )

    def test_profile_requires_a_replay_group(self) -> None:
        from sparkcache import spark_context_cache_config as cfg

        profile = resolve_profile("deepseek-v41-b12x-hma")
        profile.validate_for_deployment(
            dcp_degree=1,
            block_size=CHUNK_TOKENS,
            min_span_tokens=CHUNK_TOKENS,
            cuda_restore=False,
            group_topology=cfg.kv_group_topology(deepseek_v41_layout()),
        )
        class MambaSpec:
            block_size = CHUNK_TOKENS
            storage_block_size = CHUNK_TOKENS
            page_size_bytes = 8
            mamba_cache_mode = "align"
            tokens_per_state = 1
            num_speculative_blocks = 0
            num_prefill_checkpoint_blocks = 1

        recurrent_layout = types.SimpleNamespace(
            kv_cache_groups=(
                types.SimpleNamespace(
                    kv_cache_spec=MambaSpec(),
                    is_eagle_group=False,
                    layer_names=("recurrent",),
                ),
            )
        )
        with self.assertRaisesRegex(ProfileError, "no 'replay'"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=CHUNK_TOKENS,
                min_span_tokens=CHUNK_TOKENS,
                cuda_restore=False,
                group_topology=cfg.kv_group_topology(recurrent_layout),
            )

    def test_cuda_restore_refused_and_connector_binds_profile(self) -> None:
        profile = resolve_profile("deepseek-v41-b12x-hma")
        with self.assertRaisesRegex(ProfileError, "SparkCache CUDA restore"):
            profile.validate_for_deployment(
                dcp_degree=1,
                block_size=CHUNK_TOKENS,
                min_span_tokens=CHUNK_TOKENS,
                cuda_restore=True,
            )
        with tempfile.TemporaryDirectory() as directory:
            connector = _v41_connector(Path(directory), deepseek_v41_layout())
            self.assertEqual(connector._profile.name, "deepseek-v41-b12x-hma")
            self.assertEqual(connector._prefill_replay_tokens, REPLAY_RESERVE)
            self.assertEqual(connector._prefix_state_alignment, 2)
            self.assertEqual(connector._block_size, CHUNK_TOKENS)

    def test_v41_entry_never_matches_deepseek_v4_entry(self) -> None:
        v41 = _v41_identity(deepseek_v41_layout())
        v4 = _v41_identity(
            _deepseek_tp4_hma_config(),
            **{"spark_cache_model_profile": "deepseek-v4-fp8-hma"},
        )
        self.assertNotEqual(v41.storage_key, v4.storage_key)
        self.assertNotEqual(
            v41.to_wire()["quantization_layout"],
            v4.to_wire()["quantization_layout"],
        )

    def test_replay_reserve_state_geometry_and_pages_fork_the_entry(self) -> None:
        original = _v41_identity(deepseek_v41_layout())
        narrower = deepseek_v41_layout()
        narrower.kv_cache_groups[5].kv_cache_spec._prefill_replay_tokens = 64
        self.assertNotEqual(original.storage_key, _v41_identity(narrower).storage_key)
        coarser = deepseek_v41_layout()
        coarser.kv_cache_groups[7].kv_cache_spec.tokens_per_state = 2
        coarser.kv_cache_groups[7].kv_cache_spec.page_size_bytes = 68 * 4
        self.assertNotEqual(original.storage_key, _v41_identity(coarser).storage_key)
        larger_pages = deepseek_v41_layout()
        larger_pages.kv_cache_groups[0].kv_cache_spec.page_size_bytes = 288 * 256
        self.assertNotEqual(
            original.storage_key, _v41_identity(larger_pages).storage_key
        )

    def test_crc32_fast_entry_never_matches_sha256_entry(self) -> None:
        sha256_entry = _v41_identity(
            deepseek_v41_layout(),
            spark_cache_publication_schema="tail-cow-v2",
            spark_cache_payload_integrity="sha256",
        )
        fast_entry = _v41_identity(
            deepseek_v41_layout(),
            spark_cache_publication_schema="tail-cow-v2",
        )
        self.assertEqual(
            sha256_entry.to_wire()["publication_schema"], "page-tail-cow-v2"
        )
        self.assertEqual(
            fast_entry.to_wire()["publication_schema"], "page-tail-cow-v3-fast"
        )
        self.assertNotEqual(sha256_entry.storage_key, fast_entry.storage_key)



class DeepSeekV41SpanTests(unittest.TestCase):
    """Span selection leaves the replay reserve and stays state-aligned."""

    def _span_connector(
        self,
    ) -> tuple[SparkContextCacheConnector, tempfile.TemporaryDirectory]:
        lifetime = tempfile.TemporaryDirectory()
        connector = _v41_connector(Path(lifetime.name), deepseek_v41_layout())
        return connector, lifetime

    def test_aligned_span_property_over_prompt_lengths(self) -> None:
        connector, lifetime = self._span_connector()
        try:
            aligned = SparkContextCacheConnector._aligned_span
            for prompt_len in range(1, 2_000_001):
                span = aligned(connector, prompt_len)
                self.assertLessEqual(span, max(0, prompt_len - REPLAY_RESERVE))
                self.assertEqual(span % CHUNK_TOKENS, 0)
                self.assertEqual(span % 2, 0)
                if prompt_len < CHUNK_TOKENS + REPLAY_RESERVE:
                    self.assertEqual(span, 0)
                else:
                    self.assertGreater(span, 0)
        finally:
            lifetime.cleanup()

    def test_aligned_span_edges_around_the_reserve(self) -> None:
        connector, lifetime = self._span_connector()
        try:
            aligned = SparkContextCacheConnector._aligned_span
            self.assertEqual(aligned(connector, 1), 0)
            self.assertEqual(aligned(connector, REPLAY_RESERVE), 0)
            self.assertEqual(aligned(connector, REPLAY_RESERVE + 1), 0)
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS + REPLAY_RESERVE), CHUNK_TOKENS
            )
            # A final chunk shorter than the replay reserve still recomputes
            # whole chunks: the span stops at the last chunk boundary that
            # leaves the full reserve.
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS * 10 + 127), CHUNK_TOKENS * 9
            )
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS * 10 + 128), CHUNK_TOKENS * 10
            )
            self.assertEqual(
                aligned(connector, CHUNK_TOKENS * 10 + 129), CHUNK_TOKENS * 10
            )
            self.assertEqual(aligned(connector, CHUNK_TOKENS * 11), CHUNK_TOKENS * 10)
        finally:
            lifetime.cleanup()


class DeepSeekV41StoreRestoreTests(unittest.TestCase):
    """Stores and restores publish only persisted groups' pages."""

    BASE_SPAN = CHUNK_TOKENS * 4
    RESULT_SPAN = CHUNK_TOKENS * 9

    def test_selection_returns_empty_non_persisted_selections(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            connector = _v41_connector(
                Path(directory), deepseek_v41_layout(small_pages=True)
            )
            tables, _ = span_tables(self.RESULT_SPAN)
            selected = connector._select_group_blocks_for_span(
                tables, self.RESULT_SPAN
            )
            self.assertEqual(
                connector._group_block_counts_for_span(self.RESULT_SPAN),
                (9, 9, 9, 9, 1, 0, 0, 0),
            )
            self.assertEqual(selected[4], (tables[4][17],))
            self.assertEqual(selected[5], ())
            self.assertEqual(selected[6], ())
            self.assertEqual(selected[7], ())

    def test_store_and_restore_skip_non_persisted_groups(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            layout = deepseek_v41_layout(small_pages=True)
            connector = _v41_connector(root, layout)
            pools: dict[str, torch.Tensor] = {}
            generator = torch.Generator().manual_seed(20261015)
            for group in layout.kv_cache_groups:
                width = int(group.kv_cache_spec.page_size_bytes)
                for name in group.layer_names:
                    pools[name] = torch.randint(
                        0, 256, (512, 1, width), dtype=torch.uint8,
                        generator=generator,
                    )
            connector.register_kv_caches(pools)

            tokens = tuple(range(self.RESULT_SPAN + REPLAY_RESERVE))
            base_digest = connector._digest(list(tokens), self.BASE_SPAN)
            result_digest = connector._digest(list(tokens), self.RESULT_SPAN)
            base_tables, _ = span_tables(self.BASE_SPAN)
            result_tables, _ = span_tables(self.RESULT_SPAN)

            connector._store_one(
                _ReqPlan(
                    "v41-base-store",
                    base_digest,
                    self.BASE_SPAN,
                    base_tables[0],
                    True,
                    block_ids_by_group=base_tables,
                    token_ids=tokens[: self.BASE_SPAN],
                    request_scope=UNSALTED_SCOPE,
                )
            )
            extension_plan = _ReqPlan(
                "v41-extension-store",
                result_digest,
                self.RESULT_SPAN,
                result_tables[0],
                True,
                block_ids_by_group=result_tables,
                token_ids=tokens,
                base_context_digest=base_digest,
                base_span_tokens=self.BASE_SPAN,
                request_scope=UNSALTED_SCOPE,
            )
            connector._store_one(extension_plan)

            lookup = connector._store.lookup(connector._identity(0), result_digest)
            self.assertTrue(lookup.is_hit, lookup.reason)
            self.assertEqual(lookup.root_kind, "page_delta")
            manifest = lookup._manifest
            assert manifest is not None
            self.assertEqual(
                manifest["base_block_counts"], [4, 4, 4, 4, 1, 0, 0, 0]
            )
            self.assertEqual(
                manifest["result_block_counts"], [9, 9, 9, 9, 1, 0, 0, 0]
            )

            destination = tuple(
                tuple(range(100 + 20 * index, 100 + 20 * index + len(table)))
                for index, table in enumerate(result_tables)
            )
            for index, table in enumerate(destination):
                for block in table:
                    for name in layout.kv_cache_groups[index].layer_names:
                        pools[name][block].zero_()
            self.assertTrue(
                connector._load_one(
                    _ReqPlan(
                        "v41-restore",
                        result_digest,
                        self.RESULT_SPAN,
                        destination[0],
                        False,
                        block_ids_by_group=destination,
                        request_scope=UNSALTED_SCOPE,
                    )
                )
            )
            expected = connector._select_group_blocks_for_span(
                result_tables, self.RESULT_SPAN
            )
            destination_selected = connector._select_group_blocks_for_span(
                destination, self.RESULT_SPAN
            )
            for index, group in enumerate(layout.kv_cache_groups):
                for source_block, destination_block in zip(
                    expected[index], destination_selected[index], strict=True
                ):
                    for name in group.layer_names:
                        self.assertTrue(
                            torch.equal(
                                pools[name][destination_block],
                                pools[name][source_block],
                            ),
                            f"group {index} layer {name} restored bytes differ",
                        )
            # Non-persisted groups contributed zero pages: their destination
            # blocks stay zeroed and their source blocks stay untouched.
            for index in (5, 6, 7):
                for block in destination[index]:
                    for name in layout.kv_cache_groups[index].layer_names:
                        self.assertTrue(
                            bool((pools[name][block] == 0).all()),
                            f"group {index} layer {name} was written on restore",
                        )


class DeepSeekV41LoadFailureTests(unittest.TestCase):
    """A failed full-group load recomputes within the replay cap."""

    SPAN = CHUNK_TOKENS * 9

    def _stored_connector(
        self, root: Path
    ) -> tuple[SparkContextCacheConnector, dict[str, torch.Tensor],
               tuple[tuple[int, ...], ...], str]:
        layout = deepseek_v41_layout(small_pages=True)
        connector = _v41_connector(root, layout)
        pools: dict[str, torch.Tensor] = {}
        generator = torch.Generator().manual_seed(20261016)
        for group in layout.kv_cache_groups:
            width = int(group.kv_cache_spec.page_size_bytes)
            for name in group.layer_names:
                pools[name] = torch.randint(
                    0, 256, (512, 1, width), dtype=torch.uint8,
                    generator=generator,
                )
        connector.register_kv_caches(pools)
        tables, _ = span_tables(self.SPAN)
        tokens = tuple(range(self.SPAN + REPLAY_RESERVE))
        digest = connector._digest(list(tokens), self.SPAN)
        connector._store_one(
            _ReqPlan(
                "v41-failure-store",
                digest,
                self.SPAN,
                tables[0],
                True,
                block_ids_by_group=tables,
                token_ids=tokens,
                request_scope=UNSALTED_SCOPE,
            )
        )
        return connector, pools, tables, digest

    def test_invalid_full_block_failure_recomputes_within_replay_cap(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            connector, _pools, tables, digest = self._stored_connector(root)
            lookup = connector._store.lookup(connector._identity(0), digest)
            self.assertTrue(lookup.is_hit, lookup.reason)

            chunks = sorted(root.rglob("*.spcc"))
            self.assertTrue(chunks)
            payload = bytearray(chunks[0].read_bytes())
            payload[len(payload) // 2] ^= 0x40
            chunks[0].write_bytes(payload)

            self.assertFalse(
                connector._load_one(
                    _ReqPlan(
                        "v41-failure-restore",
                        digest,
                        self.SPAN,
                        tables[0],
                        False,
                        block_ids_by_group=tables,
                        request_scope=UNSALTED_SCOPE,
                    )
                )
            )
            self.assertFalse(
                connector._store.lookup(connector._identity(0), digest).is_hit,
                "a failed load must invalidate its entry",
            )
            # The retry boundary is the connector's own offer for the same
            # prompt. It must stay below the replay reserve, so the reset
            # request recomputes the request-private suffix itself.
            aligned = SparkContextCacheConnector._aligned_span
            prompt_len = self.SPAN + REPLAY_RESERVE
            for length in (prompt_len, self.SPAN, self.SPAN - 1):
                self.assertLessEqual(
                    aligned(connector, length), length - REPLAY_RESERVE
                )
            retry_span = aligned(connector, prompt_len)
            retry_tables, _ = span_tables(retry_span)
            retry_selection = connector._select_group_blocks_for_span(
                retry_tables, retry_span
            )
            self.assertEqual(retry_selection[5], ())
            self.assertEqual(retry_selection[6], ())
            self.assertEqual(retry_selection[7], ())



if __name__ == "__main__":
    unittest.main()
