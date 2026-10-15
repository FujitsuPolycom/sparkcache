"""Inferred B12X Qwen layout on the release 2026.10.4 vLLM source line.

One manager group per layer is a conformance fixture, not a hardware dump.
Replace its grouping with the recorded Q2 inventory before qualification.
"""

import copy
from enum import Enum
from types import SimpleNamespace as NS

import pytest

from sparkcache import spark_context_cache_config as cfg
from sparkcache.test_spark_context_cache_config import _make_vllm_config


class FullAttentionSpec(NS):
    pass


class MambaSpec(NS):
    pass


class CircularBufferSpec(NS):
    pass


class MLAAttentionSpec(FullAttentionSpec):
    pass


def qwen_layout(tp):
    page = {2: 2848, 4: 1440}[tp]
    groups = []
    for layer in range(48):
        values = dict(block_size=page, storage_block_size=page,
                      page_size_padded=page * 576, page_size_bytes=page * 576)
        if layer % 4 == 3:
            spec = FullAttentionSpec(**values, dtype="float8_e4m3fn")
        else:
            spec = MambaSpec(**values, shapes=((10240 // tp, 6), (48 // tp, 128, 128)),
                             dtypes=("bfloat16", "float32"), mamba_cache_mode="align",
                             tokens_per_state=1, num_speculative_blocks=3,
                             num_prefill_checkpoint_blocks=1, prefill_checkpoint_alignment=16,
                             tp_replicated=False, boundary_capture=False)
        groups.append(NS(kv_cache_spec=spec, layer_names=[f"model.layers.{layer}"],
                         enable_kv_transfer=True, role="default", host_resident=False))
    ple = copy.deepcopy(groups[0])
    ple.layer_names = ["model.ple"]
    ple.kv_cache_spec.tp_replicated = True
    # PLE shape is not asserted until the hardware inventory is available.
    del ple.kv_cache_spec.shapes
    groups.append(ple)
    groups.append(NS(kv_cache_spec=FullAttentionSpec(
        block_size=page, storage_block_size=page, page_size_padded=page * 1088,
        page_size_bytes=page * 1088, dtype="bfloat16"),
        layer_names=["model.mtp"], is_eagle_group=True))
    return NS(kv_cache_groups=groups, num_blocks=128)


@pytest.mark.parametrize("tp,limit,effective", [(2, 49152, 48416), (4, 61920, 61920)])
def test_b12x_groups_geometry_and_page_aligned_restore(tp, limit, effective):
    layout = qwen_layout(tp)
    topology = cfg.kv_group_topology(layout)
    page = {2: 2848, 4: 1440}[tp]
    assert len(topology) == 50 <= 64
    assert sum(g["reuse_policy"] == "full" for g in topology) == 13
    assert sum(g["reuse_policy"] == "recurrent_align" for g in topology) == 37
    assert topology[-2]["recurrent_state"]["tp_replicated"] is True
    assert topology[-1]["eagle"] and topology[-1]["page_size_bytes"] == page * 1088
    vllm, transfer = _make_vllm_config({"spark_cache_model_profile": "qwen38-flash-next-hybrid",
        "spark_cache_max_span_tokens": limit}, tp=tp, dcp=1, block_size=page)
    parsed = cfg.parse_connector_config(vllm, transfer, layout)
    assert parsed.block_size == page and parsed.max_span == limit
    # Use the connector's real alignment implementation with the CPU suite's vLLM stubs.
    from sparkcache.test_spark_context_cache_connector import SparkContextCacheConnector
    connector = NS(_block_size=page, _storage_mode=parsed.storage_mode,
                   _dcp_degree=1, _chunk_tokens=parsed.chunk_tokens)
    aligned = SparkContextCacheConnector._aligned_span
    assert aligned(connector, limit + 1) == effective
    for length in (4096, 12000, 32000, limit, limit + 1):
        span = aligned(connector, length)
        assert span % page == span % 4 == span % 8 == 0
    assert effective * 8000 + 37 * page * 576 < 512 * 1024**2


def test_non_b12x_qsa_ring_is_refused():
    groups = [NS(kv_cache_spec=spec(block_size=32, tokens_per_state=4), layer_names=[name])
              for spec, name in [(CircularBufferSpec, "ring"), (MLAAttentionSpec, "compressed")]]
    with pytest.raises(RuntimeError, match="unsupported.*CircularBufferSpec"):
        cfg.kv_group_topology(NS(kv_cache_groups=groups))


def test_startup_reports_profile_group_count_and_full_topology(tmp_path):
    import json
    from unittest import mock
    from sparkcache.test_spark_context_cache_connector import _make_connector, connector_module

    with mock.patch.object(connector_module, "logger") as log:
        _make_connector(tmp_path, 0, tp=2, dcp=1, kv_cache_config=qwen_layout(2),
                        extra_config={"spark_cache_model_profile": "qwen38-flash-next-hybrid"})
    lines = [call.args[0] % call.args[1:] for call in log.info.call_args_list]
    assert "sparkcache: topology profile=qwen38-flash-next-hybrid groups=50 policies=full,recurrent_align" in lines
    dumps = [call.args[1] for call in log.debug.call_args_list
             if call.args[0] == "sparkcache: group_topology %s"]
    assert json.loads(dumps[0])[-2]["recurrent_state"]["tp_replicated"] is True


@pytest.mark.parametrize("field,value", [("tp_replicated", True),
    ("prefill_checkpoint_alignment", 32), ("boundary_capture", True)])
def test_nondefault_recurrent_fields_fork_namespace_and_mixed_groups_fail(field, value):
    layout = qwen_layout(2)
    original = cfg.kv_group_topology_digest(layout)
    spec = layout.kv_cache_groups[0].kv_cache_spec
    setattr(spec, field, value)
    assert cfg.kv_group_topology_digest(layout) != original
    other = copy.deepcopy(spec)
    setattr(other, field, {"tp_replicated": False, "prefill_checkpoint_alignment": None,
                           "boundary_capture": False}[field])
    layout.kv_cache_groups[0].layer_names = ["a", "b"]
    layout.kv_cache_groups[0].kv_cache_spec = NS(block_size=spec.block_size, kv_cache_specs={"a": spec, "b": other})
    with pytest.raises(RuntimeError, match="incompatible state geometry"):
        cfg.kv_group_topology(layout)


@pytest.mark.parametrize("field,value", [("enable_kv_transfer", False),
    ("role", "hisparse_source"), ("role", "hisparse_indexer"), ("host_resident", True)])
def test_rebuilt_and_host_groups_fail_closed(field, value):
    layout = qwen_layout(4)
    setattr(layout.kv_cache_groups[0], field, value)
    with pytest.raises(RuntimeError, match="non-transferable"):
        cfg.kv_group_topology(layout)


@pytest.mark.parametrize("profile", ["glm53-flash-hybrid", "deepseek-v4-fp8-hma"])
def test_default_fields_preserve_legacy_topology_and_cache_identity(profile):
    class Role(str, Enum):
        DEFAULT = "default"

    spec = MambaSpec(block_size=512, mamba_cache_mode="align", tokens_per_state=1,
                     num_speculative_blocks=3, num_prefill_checkpoint_blocks=1)
    group = NS(kv_cache_spec=spec, layer_names=["recurrent"])
    layout = NS(kv_cache_groups=[group])
    vllm, transfer = _make_vllm_config({"spark_cache_model_profile": profile},
                                      dcp=1, block_size=512)
    digest = cfg.kv_group_topology_digest(layout)
    identity = cfg.parse_connector_config(vllm, transfer, layout).build_identity(0, 0)
    spec.tp_replicated = spec.boundary_capture = False
    spec.prefill_checkpoint_alignment = None
    group.enable_kv_transfer, group.role, group.host_resident = True, Role.DEFAULT, False
    assert cfg.kv_group_topology_digest(layout) == digest
    assert cfg.parse_connector_config(vllm, transfer, layout).build_identity(0, 0) == identity
    assert set(cfg.kv_group_topology(layout)[0]["recurrent_state"]) == {
        "mamba_cache_mode", "tokens_per_state", "num_speculative_blocks", "num_prefill_checkpoint_blocks"}
