"""GPU-free contract tests for DeepSeek-V4-Flash-0731 TP4/DCP1."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from deploy.deployment_contract import source_tree_sha256
from deploy.deepseek_v4.tp4_cluster_preflight import validate_cluster
from deploy.deepseek_v4.tp4_profile import (
    LOW_WATERMARK_BYTES,
    MAX_BYTES,
    PROFILE,
    ProfileTransformError,
    build_kv_transfer_config,
    transform_inspection,
)


IMAGE = "sha256:" + "a" * 64
CHECKPOINT = "b" * 64


def test_profile_matches_the_repository_sparkcache_source() -> None:
    repository = Path(__file__).resolve().parents[2]
    assert PROFILE["sparkcache"]["source_sha256"] == source_tree_sha256(
        repository / "sparkcache"
    )


def _source(rank: int = 0) -> dict:
    command = [
        "serve",
        "/models/deepseek-v4-flash-0731",
        "--tensor-parallel-size",
        "4",
        "--nnodes",
        "4",
        "--node-rank",
        str(rank),
        "--master-addr",
        "192.168.0.10",
        "--master-port",
        "29500",
        "--distributed-executor-backend",
        "mp",
        "--dtype",
        "bfloat16",
        "--max-model-len",
        "524288",
        "--max-num-seqs",
        "32",
        "--gpu-memory-utilization",
        "0.70",
        "--kv-cache-memory-bytes",
        "34359738368",
        "--kv-cache-dtype",
        "fp8_ds_mla",
        "--tokenizer-mode",
        "deepseek_v4",
        "--kernel-config",
        '{"enable_cutedsl_warmup":false}',
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "deepseek_v4",
        "--speculative-config",
        '{"method":"dspark","num_speculative_tokens":5,'
        '"moe_backend":"b12x","draft_sample_method":"greedy"}',
        "--served-model-name",
        "deepseek-v4-flash-0731",
    ]
    if rank == 0:
        command.extend(("--host", "0.0.0.0", "--port", "8000"))
    else:
        command.append("--headless")
    return {
        "Image": IMAGE,
        "Config": {
            "Cmd": command,
            "Env": [
                "LD_PRELOAD=/usr/local/cuda/compat/libcuda.so.1:"
                "/opt/sparkring/nccl/libnccl.so.2",
                "VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2",
                "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
                "VLLM_USE_B12X_MOE=1",
                "VLLM_USE_B12X_SPARSE_INDEXER=1",
                "VLLM_DSPARK_IMPL=upstream",
                "VLLM_DSPARK_REPLICATE_MARKOV_W1=1",
                "VLLM_DSPARK_REPLICATE_MARKOV_W2=1",
                "HF_HUB_OFFLINE=1",
                "PYTHONPATH=/opt/spark-vllm",
                "SPARK_CONTEXT_CACHE_ENABLE=0",
                "MASTER_PORT=29500",
            ],
            "Labels": {"org.sparkring.profile": "deepseek-source"},
        },
        "HostConfig": {
            "NetworkMode": "host",
            "IpcMode": "host",
            "ShmSize": 17179869184,
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/srv/models/deepseek-v4-flash-0731",
                "Destination": "/models/deepseek-v4-flash-0731",
                "RW": False,
            }
        ],
    }


def _option(arguments: list[str], name: str) -> str:
    return arguments[arguments.index(name) + 1]


def _environment(inspection: dict) -> dict[str, str]:
    return dict(item.split("=", 1) for item in inspection["Config"]["Env"])


def test_profile_pins_the_tp4_dcp1_hma_contract() -> None:
    assert PROFILE["profile_id"] == "deepseek-v4-flash-0731-tp4-dcp1"
    assert PROFILE["serving"]["tensor_parallel_size"] == 4
    assert PROFILE["serving"]["decode_context_parallel_size"] == 1
    assert PROFILE["serving"]["kv_cache_dtype"] == "fp8_ds_mla"
    assert PROFILE["serving"]["block_size"] == 256
    assert PROFILE["serving"]["speculation_method"] == "dspark"
    assert PROFILE["serving"]["speculation_tokens"] == 5
    assert PROFILE["vllm"]["scheduler_hma_postimage_sha256"] == (
        "2f34aa9d65a495a86d814c90f654fbe1ff754cfdbecd204b98d513652ca3e06d"
    )


def test_transfer_config_is_bounded_python_hma_restore() -> None:
    config = build_kv_transfer_config(CHECKPOINT)
    assert config["kv_connector"] == "SparkContextCacheConnector"
    assert config["kv_connector_module_path"] == (
        "sparkcache.spark_context_cache_connector"
    )
    assert config["kv_load_failure_policy"] == "recompute"
    extra = config["kv_connector_extra_config"]
    assert extra["spark_cache_model_profile"] == "deepseek-v4-fp8-hma"
    assert extra["spark_cache_draft_policy"] == "colocated_target"
    assert extra["spark_cache_streaming_snapshots"] is False
    assert extra["spark_cache_native_restore"] is False
    assert extra["spark_cache_max_bytes"] == MAX_BYTES == 200 * 1024**3
    assert extra["spark_cache_low_watermark_bytes"] == LOW_WATERMARK_BYTES
    assert LOW_WATERMARK_BYTES == 180 * 1024**3


def test_transform_accepts_all_four_physical_ranks() -> None:
    for rank in range(4):
        transformed = transform_inspection(
            _source(rank),
            checkpoint_sha256=CHECKPOINT,
            api_port=8100,
            master_port=29600,
        )
        command = transformed["Config"]["Cmd"]
        assert _option(command, "--node-rank") == str(rank)
        assert _option(command, "--master-port") == "29600"
        assert _option(command, "--block-size") == "256"
        assert "--decode-context-parallel-size" not in command
        assert ("--headless" in command) is (rank != 0)
        if rank == 0:
            assert _option(command, "--port") == "8100"
        else:
            assert "--port" not in command
        transfer = json.loads(_option(command, "--kv-transfer-config"))
        assert transfer == build_kv_transfer_config(CHECKPOINT)
        environment = _environment(transformed)
        assert environment["MASTER_PORT"] == "29600"
        assert "SPARK_CONTEXT_CACHE_ENABLE" not in environment
        assert "SPARK_CONTEXT_CACHE_ENABLE" in environment[
            "SPARKRING_EXPLICITLY_UNSET"
        ]
        assert environment["PYTHONPATH"].startswith("/opt/sparkcache-src:")
        assert "/opt/sparkcache-src/sparkcache" not in environment["PYTHONPATH"]
        assert transformed["Config"]["Labels"][
            "org.sparkcache.deployment-profile"
        ] == "deepseek-v4-flash-0731-tp4-dcp1"


def test_cluster_preflight_accepts_one_homogeneous_four_rank_ring() -> None:
    plan = validate_cluster(
        [_source(rank) for rank in (3, 0, 2, 1)],
        checkpoint_sha256=CHECKPOINT,
        api_port=8100,
        master_port=29600,
    )
    assert plan["schema"] == "sparkcache-deepseek0731-tp4-cluster-plan/v1"
    assert plan["image"] == IMAGE
    assert plan["master_addr"] == "192.168.0.10"
    assert [record["rank"] for record in plan["ranks"]] == [0, 1, 2, 3]
    assert [record["api_port"] for record in plan["ranks"]] == [
        8100,
        None,
        None,
        None,
    ]


def test_cluster_preflight_rejects_duplicate_physical_rank() -> None:
    with pytest.raises(ProfileTransformError, match="exactly once"):
        validate_cluster(
            [_source(rank) for rank in (0, 1, 2, 2)],
            checkpoint_sha256=CHECKPOINT,
            api_port=8100,
            master_port=29600,
        )


def test_cluster_preflight_rejects_image_drift() -> None:
    inspections = [_source(rank) for rank in range(4)]
    inspections[3]["Image"] = "sha256:" + "c" * 64
    with pytest.raises(ProfileTransformError, match="not homogeneous"):
        validate_cluster(
            inspections,
            checkpoint_sha256=CHECKPOINT,
            api_port=8100,
            master_port=29600,
        )


def test_cluster_preflight_rejects_collective_port_drift() -> None:
    inspections = [_source(rank) for rank in range(4)]
    for rank, inspection in enumerate(inspections):
        inspection["Config"]["Env"].append(
            f"SPARK_TP4_COLLECTIVE_PORT={31000 + (rank == 3)}"
        )
    with pytest.raises(ProfileTransformError, match="port assignments differ"):
        validate_cluster(
            inspections,
            checkpoint_sha256=CHECKPOINT,
            api_port=8100,
            master_port=29600,
        )


@pytest.mark.parametrize("degree", (2, 4))
def test_transform_rejects_hma_dcp_greater_than_one(degree: int) -> None:
    inspection = _source()
    inspection["Config"]["Cmd"].extend(
        ("--decode-context-parallel-size", str(degree))
    )
    with pytest.raises(ProfileTransformError, match="DCP1"):
        transform_inspection(inspection, checkpoint_sha256=CHECKPOINT)


def test_transform_rejects_generic_fp8_for_external_cache() -> None:
    inspection = _source()
    command = inspection["Config"]["Cmd"]
    command[command.index("--kv-cache-dtype") + 1] = "fp8"
    with pytest.raises(ProfileTransformError, match="fp8_ds_mla"):
        transform_inspection(inspection, checkpoint_sha256=CHECKPOINT)


@pytest.mark.parametrize(
    "argument",
    ("--enable-expert-parallel", "--disable-hybrid-kv-cache-manager"),
)
def test_transform_rejects_incompatible_runtime_flags(argument: str) -> None:
    inspection = _source()
    inspection["Config"]["Cmd"].append(argument)
    with pytest.raises(ProfileTransformError):
        transform_inspection(inspection, checkpoint_sha256=CHECKPOINT)


def test_transform_rejects_lmcache_source_configuration() -> None:
    inspection = _source()
    inspection["Config"]["Cmd"].extend(
        (
            "--kv-transfer-config",
            '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both"}',
        )
    )
    with pytest.raises(ProfileTransformError, match="LMCache"):
        transform_inspection(inspection, checkpoint_sha256=CHECKPOINT)


def test_transform_strips_inherited_glm_attestation_state() -> None:
    inspection = _source()
    inherited = {
        "SPARKRING_MODEL_REPOSITORY": "zai-org/GLM-5.2",
        "SPARKRING_MODEL_CONFIG_SHA256": "c" * 64,
        "SPARKRING_ATTEST_MODEL_REVISION": "old-glm-revision",
        "SPARK_Q40_STATE": "/opt/spark-vllm/q40.json",
    }
    inspection["Config"]["Env"].extend(
        f"{name}={value}" for name, value in inherited.items()
    )
    transformed = transform_inspection(
        inspection,
        checkpoint_sha256=CHECKPOINT,
    )
    environment = _environment(transformed)
    assert set(inherited).isdisjoint(environment)
    explicit_unset = set(environment["SPARKRING_EXPLICITLY_UNSET"].split(","))
    assert set(inherited) <= explicit_unset


def test_transform_rejects_private_runtime_mounts() -> None:
    inspection = _source()
    inspection["Mounts"].append(
        {
            "Type": "bind",
            "Source": "/var/tmp/private-scheduler.py",
            "Destination": "/opt/venv/lib/python3.12/site-packages/"
            "vllm/v1/core/sched/scheduler.py",
            "RW": False,
        }
    )
    with pytest.raises(ProfileTransformError, match="non-portable runtime mounts"):
        transform_inspection(inspection, checkpoint_sha256=CHECKPOINT)


def test_transform_rejects_collective_port_collisions() -> None:
    inspection = _source()
    inspection["Config"]["Env"].append("SPARK_TP4_CONTROL_PORT0=11100")
    with pytest.raises(ProfileTransformError, match="collides"):
        transform_inspection(
            inspection,
            checkpoint_sha256=CHECKPOINT,
            api_port=11100,
        )


def test_transform_does_not_mutate_the_source_inspection() -> None:
    inspection = _source()
    original = copy.deepcopy(inspection)
    transform_inspection(inspection, checkpoint_sha256=CHECKPOINT)
    assert inspection == original


@pytest.mark.parametrize("checkpoint", ("", "A" * 64, "1" * 63))
def test_checkpoint_identity_must_be_canonical_sha256(checkpoint: str) -> None:
    with pytest.raises(ProfileTransformError, match="lowercase"):
        transform_inspection(_source(), checkpoint_sha256=checkpoint)
