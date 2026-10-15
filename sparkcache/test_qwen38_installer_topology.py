"""Qwen TP2/TP4 manager groups observed on release 2026.10.4.

The 2026-10-15 hardware inventory has three 12-layer GDN groups, one PLE
state group, and one attention group containing 12 QSA layers and BF16 MTP.
Scheduler specs retain the first member of each worker uniform-type group.
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


class MLAAttentionSpec(FullAttentionSpec):
    pass


class UniformTypeKVCacheSpecs(NS):
    """Worker wrapper; its page bytes sum members and DCP defaults to false."""


def qwen_layout(tp, *, worker=False):
    page = {2: 2848, 4: 1440}[tp]
    groups = []
    for offset in range(3):
        names = [f"language_model.model.layers.{n}.linear_attn"
                 for n in range(offset, 48, 4)]
        spec = MambaSpec(block_size=page, page_size_bytes=page * 576,
                         mamba_cache_mode="align", tokens_per_state=-1,
                         num_speculative_blocks=3, num_prefill_checkpoint_blocks=1,
                         prefill_checkpoint_alignment=16, dcp_replicated=True)
        groups.append(NS(kv_cache_spec=spec, layer_names=names))
    ple = copy.deepcopy(groups[0])
    ple.layer_names = ["language_model.model.layers.1.ple"]
    ple.kv_cache_spec.tp_replicated = True
    groups.append(ple)
    groups.append(NS(kv_cache_spec=FullAttentionSpec(
        block_size=page, page_size_bytes=page * 576, dtype="float8_e4m3fn",
        dcp_replicated=False), layer_names=[
            *(f"language_model.model.layers.{n}.self_attn.attn" for n in range(3, 48, 4)),
            "mtp.layers.48.self_attn.attn"], is_eagle_group=False))
    if worker:
        for group in groups:
            members = {name: copy.deepcopy(group.kv_cache_spec) for name in group.layer_names}
            if "mtp.layers.48.self_attn.attn" in members:
                mtp = members["mtp.layers.48.self_attn.attn"]
                mtp.dtype, mtp.page_size_bytes = "bfloat16", page * 1088
            group.kv_cache_spec = UniformTypeKVCacheSpecs(
                block_size=page, page_size_bytes=sum(s.page_size_bytes for s in members.values()),
                dcp_replicated=False, kv_cache_specs=members)
    return NS(kv_cache_groups=groups, num_blocks=3)


@pytest.mark.parametrize("tp,dcp", [(2, 1), (4, 1), (4, 2), (4, 4)])
def test_qwen_hw_1_scheduler_worker_identity_and_context_digest(tmp_path, tp, dcp):
    from sparkcache.test_spark_context_cache_connector import _make_connector, KVConnectorRole

    scheduler_layout, worker_layout = qwen_layout(tp), qwen_layout(tp, worker=True)
    assert cfg.kv_group_topology(scheduler_layout, dcp_degree=dcp) == cfg.kv_group_topology(worker_layout, dcp_degree=dcp)
    assert cfg.kv_group_topology_digest(scheduler_layout, dcp_degree=dcp) == cfg.kv_group_topology_digest(worker_layout, dcp_degree=dcp)
    extra = {"spark_cache_model_profile": "qwen38-flash-next-hybrid",
             "spark_cache_publication_schema": "tail-cow-v2"}
    scheduler = _make_connector(tmp_path, 0, tp=tp, dcp=dcp, kv_cache_config=scheduler_layout,
                                role=KVConnectorRole.SCHEDULER, extra_config=extra)
    worker = _make_connector(tmp_path, 0, tp=tp, dcp=dcp, kv_cache_config=worker_layout, extra_config=extra)
    try:
        assert scheduler._context_digest_salt == worker._context_digest_salt
        assert scheduler._digest(list(range(6000)), 5760) == worker._digest(list(range(6000)), 5760)
        assert all(g["dcp_replicated"] for g in worker._group_topology[:4])
        assert worker._group_topology[4]["dcp_shard_count"] == dcp
    finally:
        scheduler.shutdown()
        worker.shutdown()


def test_qwen_hw_1_representative_follows_member_insertion_order():
    worker = qwen_layout(2, worker=True)
    spec = worker.kv_cache_groups[-1].kv_cache_spec
    spec.kv_cache_specs = dict(reversed(list(spec.kv_cache_specs.items())))
    scheduler = qwen_layout(2)
    scheduler.kv_cache_groups[-1].kv_cache_spec = next(iter(spec.kv_cache_specs.values()))
    assert cfg.kv_group_topology(worker) == cfg.kv_group_topology(scheduler)
    assert cfg.kv_group_topology(worker)[-1]["page_size_bytes"] == 2848 * 1088


@pytest.mark.parametrize("field,value", [("block_size", 32), ("storage_block_size", 32),
                                         ("dcp_replicated", False)])
def test_qwen_hw_1_inconsistent_member_geometry_is_refused(field, value):
    layout = qwen_layout(2, worker=True)
    members = list(layout.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.values())
    setattr(members[-1], field, value)
    with pytest.raises(RuntimeError, match="incompatible manager geometry"):
        cfg.kv_group_topology(layout)


def test_qwen_hw_1_missing_member_spec_is_refused():
    layout = qwen_layout(2, worker=True)
    layout.kv_cache_groups[0].kv_cache_spec.kv_cache_specs.popitem()
    with pytest.raises(RuntimeError, match="every layer spec"):
        cfg.kv_group_topology(layout)


@pytest.mark.parametrize("tp", [2, 4])
def test_qwen_hw_1_tail_cow_v2_worker_commits_scheduler_extension(tmp_path, tp):
    import torch
    from sparkcache.request_cache_scope import UNSALTED_SCOPE
    from sparkcache.test_spark_context_cache_connector import (
        _make_connector, _ReqPlan, KVConnectorRole, SparkCacheConnectorMetadata)

    page = {2: 2848, 4: 1440}[tp]
    layout = qwen_layout(tp, worker=True)
    extra = {"spark_cache_model_profile": "qwen38-flash-next-hybrid",
             "spark_cache_publication_schema": "tail-cow-v2"}
    scheduler = _make_connector(tmp_path, 0, tp=tp, dcp=1, kv_cache_config=qwen_layout(tp),
                                role=KVConnectorRole.SCHEDULER, extra_config=extra)
    worker = _make_connector(tmp_path, 0, tp=tp, dcp=1, kv_cache_config=layout, extra_config=extra)
    try:
        worker.register_kv_caches({name: torch.full((3, 1, spec.page_size_bytes), i + 1, dtype=torch.uint8)
            for i, group in enumerate(layout.kv_cache_groups)
            for name, spec in group.kv_cache_spec.kv_cache_specs.items()})
        tokens = tuple(range(page * 2))
        base = scheduler._digest(list(tokens), page)
        result = scheduler._digest(list(tokens), page * 2)
        plans = [
            _ReqPlan("qwen-base", base, page, (1,), True,
                     block_ids_by_group=((1,),) * 5, token_ids=tokens[:page], request_scope=UNSALTED_SCOPE),
            _ReqPlan("qwen-tail", result, page * 2, (1, 2), True,
                     block_ids_by_group=((1, 2),) * 5, token_ids=tokens,
                     base_context_digest=base, base_span_tokens=page, request_scope=UNSALTED_SCOPE),
        ]
        expected = worker._snapshot_hybrid_store(plans[1])
        for plan in plans:
            worker.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
            worker.wait_for_save()
            assert worker.wait_for_pending_stores(timeout=30)
        assert worker.counters["store_committed"] == 2
        lookup = worker._store.lookup(worker._identity(0), result)
        assert lookup.is_hit, lookup.reason
        assert lookup.root_kind == "page_delta"
        assert worker._store.restore_page_snapshot(lookup, layout=worker._page_layout,
            result_block_counts=expected.block_counts, result_boundary_tokens=page * 2) == expected.encoded_pages
    finally:
        scheduler.shutdown()
        worker.shutdown()


@pytest.mark.parametrize("tp,limit,effective", [(2, 49152, 48416), (4, 61920, 61920)])
def test_b12x_groups_geometry_and_page_aligned_restore(tp, limit, effective):
    layout = qwen_layout(tp)
    topology = cfg.kv_group_topology(layout)
    page = {2: 2848, 4: 1440}[tp]
    assert len(topology) == 5 <= 64
    assert sum(g["reuse_policy"] == "full" for g in topology) == 1
    assert sum(g["reuse_policy"] == "recurrent_align" for g in topology) == 4
    assert topology[-2]["recurrent_state"]["tp_replicated"] is True
    assert not topology[-1]["eagle"] and len(topology[-1]["layers"]) == 13
    members = qwen_layout(tp, worker=True).kv_cache_groups[-1].kv_cache_spec.kv_cache_specs
    assert sum(s.page_size_bytes for s in members.values()) == page * 8000
    vllm, transfer = _make_vllm_config({"spark_cache_model_profile": "qwen38-flash-next-hybrid",
        "spark_cache_max_span_tokens": limit}, tp=tp, dcp=1, block_size=page)
    parsed = cfg.parse_connector_config(vllm, transfer, layout)
    assert parsed.block_size == page and parsed.max_span == limit
    # Use the connector's real alignment implementation with the CPU suite's vLLM stubs.
    from sparkcache.test_spark_context_cache_connector import SparkContextCacheConnector
    connector = NS(_block_size=page, _storage_mode=parsed.storage_mode,
                   _dcp_degree=1, _chunk_tokens=parsed.chunk_tokens,
                   _prefix_state_alignment=parsed.prefix_state_alignment,
                   _prefill_replay_tokens=parsed.prefill_replay_tokens)
    aligned = SparkContextCacheConnector._aligned_span
    assert aligned(connector, limit + 1) == effective
    for length in (4096, 12000, 32000, limit, limit + 1):
        span = aligned(connector, length)
        assert span % page == span % 4 == span % 8 == 0
    assert effective * 8000 + 37 * page * 576 < 512 * 1024**2


def test_non_b12x_qsa_ring_is_refused():
    class QsaRingSpec(NS):
        pass

    groups = [NS(kv_cache_spec=spec(block_size=32, tokens_per_state=4), layer_names=[name])
              for spec, name in [(QsaRingSpec, "ring"), (MLAAttentionSpec, "compressed")]]
    with pytest.raises(RuntimeError, match="unsupported.*QsaRingSpec"):
        cfg.kv_group_topology(NS(kv_cache_groups=groups))


def test_circular_buffer_group_requires_engine_private_properties():
    class CircularBufferSpec(NS):
        pass

    groups = [NS(kv_cache_spec=CircularBufferSpec(block_size=8, tokens_per_state=1),
                 layer_names=["ring"])]
    with pytest.raises(RuntimeError, match="circular-buffer"):
        cfg.kv_group_topology(NS(kv_cache_groups=groups))


def test_startup_reports_profile_group_count_and_full_topology(tmp_path):
    import json
    from unittest import mock
    from sparkcache.test_spark_context_cache_connector import _make_connector, connector_module

    with mock.patch.object(connector_module, "logger") as log:
        _make_connector(tmp_path, 0, tp=2, dcp=1, kv_cache_config=qwen_layout(2),
                        extra_config={"spark_cache_model_profile": "qwen38-flash-next-hybrid"})
    lines = [call.args[0] % call.args[1:] for call in log.info.call_args_list]
    assert "sparkcache: topology profile=qwen38-flash-next-hybrid groups=5 policies=full,recurrent_align" in lines
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
