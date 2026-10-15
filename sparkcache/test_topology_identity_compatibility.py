"""Pin topology and complete rank identities computed at a136fc55db39."""

import json
from pathlib import Path

import pytest

from sparkcache import spark_context_cache_config as cfg
from sparkcache.qualification.fixture import KvCacheConfig
from sparkcache.test_qwen38_installer_topology import qwen_layout
from sparkcache.test_spark_context_cache_config import _make_vllm_config
from sparkcache.test_spark_context_cache_connector import (
    _deepseek_tp4_hma_config, _hybrid_kv_cache_config,
)


RECEIPT = json.loads(Path(__file__).with_name("topology_identity_compatibility.json").read_text())
FACTORIES = {"glm52": lambda: None, "glm53": KvCacheConfig,
             "hybrid": _hybrid_kv_cache_config, "deepseek_v4": _deepseek_tp4_hma_config,
             "qwen_tp2": lambda: qwen_layout(2), "qwen_tp4": lambda: qwen_layout(4)}


@pytest.mark.parametrize("row", RECEIPT["fixtures"], ids=lambda row: f'{row["fixture"]}-dcp{row["dcp"]}')
def test_qwen_hw_1_preserves_qualified_topology_and_rank_identities(row):
    layout = FACTORIES[row["fixture"]]()
    vllm, transfer = _make_vllm_config({"spark_cache_model_profile": row["profile"]}, dcp=row["dcp"], tp=row.get("tp", 4))
    parsed = cfg.parse_connector_config(vllm, transfer, layout)
    assert cfg.kv_group_topology_digest(layout, dcp_degree=row["dcp"]) == row["topology_digest"]
    assert [parsed.build_identity(rank % row["dcp"], rank).storage_key
            for rank in range(row.get("tp", 4))] == row["storage_keys"]
