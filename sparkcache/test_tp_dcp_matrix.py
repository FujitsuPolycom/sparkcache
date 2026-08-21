"""GPU-free topology matrix for the per-token SparkCache storage path."""

from __future__ import annotations

import dataclasses
import tempfile
from pathlib import Path

import pytest
import torch

import sparkcache.spark_context_cache_codec as codec
from sparkcache.test_generalization import _make_connector_pn
from sparkcache.test_spark_context_cache_connector import (
    KVConnectorRole,
    SparkCacheConnectorMetadata,
    _ReqPlan,
    _drain,
    _drain_store,
)


_VALID_GEOMETRIES = (
    (1, 1),
    (2, 1),
    (2, 2),
    (4, 1),
    (4, 2),
    (4, 4),
)
_LAYERS = {
    "model.layers.0.self_attn.attn": 8,
    "model.layers.0.self_attn.indexer_cache": 4,
    # The fixed-MTP4 GLM serving recipe (SparkRing identifier R7) registers MTP
    # state without a draft marker,
    # so colocated state classifies into the target record family.
    "model.layers.1.self_attn.attn": 8,
}
_BLOCK_IDS = (3, 0, 5, 1)
_BLOCK_SIZE = 64
_SPAN = 256


def _make_pool(seed: int) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(20260821 + seed)
    return {
        name: torch.randint(
            0,
            256,
            (6, _BLOCK_SIZE, width),
            dtype=torch.uint8,
            generator=generator,
        )
        for name, width in _LAYERS.items()
    }


def _connector(root: Path, physical_rank: int, tp: int, dcp: int):
    return _make_connector_pn(
        root,
        physical_rank,
        tp=tp,
        dcp=dcp,
        block_size=_BLOCK_SIZE,
        extra_config={
            "spark_cache_draft_policy": "colocated_target",
            "spark_cache_draft_checkpoint_sha256": "",
        },
    )


def test_target_tp_dcp_matrix_has_complete_dcp_position_coverage() -> None:
    for tp, dcp in _VALID_GEOMETRIES:
        owned = {
            position
            for dcp_rank in range(dcp)
            for position in codec.owned_positions(_SPAN, dcp, dcp_rank)
        }
        assert owned == set(range(_SPAN)), (tp, dcp)
        assert tp % dcp == 0
        assert 256 % dcp == 0


@pytest.mark.parametrize("tp,dcp", ((1, 2), (2, 4), (3, 2)))
def test_dcp_degree_must_divide_tensor_parallel_degree(tp: int, dcp: int) -> None:
    with tempfile.TemporaryDirectory() as directory:
        with pytest.raises(RuntimeError, match="must divide tensor parallel size"):
            _make_connector_pn(Path(directory), 0, tp=tp, dcp=dcp)


def test_target_tp_dcp_matrix_round_trips_each_physical_rank() -> None:
    for tp, dcp in _VALID_GEOMETRIES:
        with tempfile.TemporaryDirectory() as directory:
            connectors = []
            pools = []
            originals = []
            block_ids = _BLOCK_IDS[: _SPAN // dcp // _BLOCK_SIZE]
            store_plan = _ReqPlan(
                request_id=f"matrix-{tp}-{dcp}",
                digest="a" * 64,
                span_tokens=_SPAN,
                block_ids=block_ids,
                is_store=True,
            )
            for physical_rank in range(tp):
                connector = _connector(
                    Path(directory) / f"rank-{physical_rank}",
                    physical_rank,
                    tp,
                    dcp,
                )
                pool = _make_pool(physical_rank)
                connector.register_kv_caches(pool)
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(plans=[store_plan])
                )
                connector.wait_for_save()
                _drain_store(connector)
                assert connector.counters["store_committed"] == 1
                connectors.append(connector)
                pools.append(pool)
                originals.append({name: value.clone() for name, value in pool.items()})

            identities = {
                connector._identity(physical_rank % dcp).storage_key
                for physical_rank, connector in enumerate(connectors)
            }
            assert len(identities) == tp, (tp, dcp)

            load_plan = dataclasses.replace(store_plan, is_store=False)
            for physical_rank, connector in enumerate(connectors):
                pool = pools[physical_rank]
                for tensor in pool.values():
                    tensor.zero_()
                connector.bind_connector_metadata(
                    SparkCacheConnectorMetadata(plans=[load_plan])
                )
                connector.start_load_kv(None)
                assert _drain(connector) == {store_plan.request_id}
                assert connector.counters["load_verified"] == 1

                dcp_rank = physical_rank % dcp
                positions = codec.owned_positions(_SPAN, dcp, dcp_rank)
                slots = codec.local_slots_for_positions(
                    positions,
                    block_ids,
                    _BLOCK_SIZE,
                    dcp,
                )
                slot_tensor = torch.tensor(slots, dtype=torch.long)
                for name, width in _LAYERS.items():
                    actual = pool[name].view(-1, width)
                    expected = originals[physical_rank][name].view(-1, width)
                    torch.testing.assert_close(
                        actual[slot_tensor],
                        expected[slot_tensor],
                        rtol=0,
                        atol=0,
                    )
                    untouched = torch.ones(actual.shape[0], dtype=torch.bool)
                    untouched[slot_tensor] = False
                    assert (actual[untouched] == 0).all()

            for connector in connectors:
                connector.shutdown()


def test_target_tp_dcp_matrix_requires_every_physical_tp_rank_for_quorum() -> None:
    for tp, dcp in _VALID_GEOMETRIES:
        with tempfile.TemporaryDirectory() as directory:
            connector = _make_connector_pn(
                Path(directory),
                0,
                tp=tp,
                dcp=dcp,
                role=KVConnectorRole.SCHEDULER,
                extra_config={
                    "spark_cache_draft_policy": "colocated_target",
                    "spark_cache_draft_checkpoint_sha256": "",
                },
            )
            digest = "b" * 64
            connector._quorum[digest] = set(range(tp - 1))
            assert not connector._has_full_quorum(digest)
            connector._quorum[digest] = set(range(tp))
            assert connector._has_full_quorum(digest)
