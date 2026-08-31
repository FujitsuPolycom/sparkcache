"""Pure launch transformation for DeepSeek-V4-Flash-0731 TP4/DCP1."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from deploy.deployment_contract import (
    DeploymentContractError,
    compact_json,
    drop_option,
    environment_map,
    integer_option,
    one_option,
    option_values,
    optional_one_option,
    validate_port,
    vllm_arguments,
)


class ProfileTransformError(DeploymentContractError):
    """The source inspection or requested deployment state is unsupported."""


PROFILE_PATH = Path(__file__).with_name("tp4_profile.json")
PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
if PROFILE.get("schema") != "sparkcache-deployment-profile/v1":
    raise RuntimeError(f"unsupported deployment profile: {PROFILE_PATH}")

PROFILE_ID = str(PROFILE["profile_id"])
DEFAULT_CACHE_ROOT = str(PROFILE["cache"]["root"])
MAX_BYTES = int(PROFILE["cache"]["max_bytes"])
LOW_WATERMARK_BYTES = int(PROFILE["cache"]["low_watermark_bytes"])
TTL_SECONDS = int(PROFILE["cache"]["ttl_seconds"])

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PORT_ENVIRONMENT_RE = re.compile(r"(?:^|_)PORT\d*\Z")


def _require_sha256(value: str, role: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ProfileTransformError(f"{role} must be 64 lowercase hexadecimal digits")
    return value


_compact_json = compact_json
_vllm_args = vllm_arguments


def _environment_map(values: Iterable[str]) -> dict[str, str]:
    return environment_map(values, error_type=ProfileTransformError)


def _option_values(arguments: list[str], option: str) -> list[str]:
    return option_values(arguments, option, error_type=ProfileTransformError)


def _drop_option(arguments: list[str], option: str) -> list[str]:
    return drop_option(arguments, option, error_type=ProfileTransformError)


def _one(arguments: list[str], option: str) -> str:
    return one_option(arguments, option, error_type=ProfileTransformError)


def _optional_one(arguments: list[str], option: str) -> str | None:
    return optional_one_option(arguments, option, error_type=ProfileTransformError)


def _integer(arguments: list[str], option: str) -> int:
    return integer_option(arguments, option, error_type=ProfileTransformError)


def _port(value: int | None, role: str) -> int | None:
    return validate_port(value, role, error_type=ProfileTransformError)


def _validate_source_mounts(inspection: dict[str, Any]) -> None:
    model_path = PROFILE["model"]["container_path"]
    mounts = inspection.get("Mounts", ())
    model_mounts = [
        mount
        for mount in mounts
        if isinstance(mount, dict) and mount.get("Destination") == model_path
    ]
    if (
        len(model_mounts) != 1
        or model_mounts[0].get("Type") != "bind"
        or model_mounts[0].get("RW") is not False
    ):
        raise ProfileTransformError(
            "source DeepSeek model path must have one read-only bind mount"
        )
    allowed = {model_path, "/root/.cache/huggingface"}
    unexpected = sorted(
        str(mount.get("Destination"))
        for mount in mounts
        if not isinstance(mount, dict) or mount.get("Destination") not in allowed
    )
    if unexpected:
        raise ProfileTransformError(
            "source DeepSeek inspection contains non-portable runtime mounts: "
            + ", ".join(unexpected)
        )
    source = str(model_mounts[0].get("Source", ""))
    path = PurePosixPath(source)
    if not path.is_absolute() or ".." in path.parts or str(path) != source:
        raise ProfileTransformError("source model bind must use an absolute host path")


def _validate_environment(values: Iterable[str]) -> list[str]:
    environment = _environment_map(values)
    required = {
        "VLLM_NCCL_SO_PATH": "/opt/sparkring/nccl/libnccl.so.2",
        "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        "VLLM_USE_B12X_MOE": "1",
        "VLLM_USE_B12X_SPARSE_INDEXER": "1",
        "VLLM_DSPARK_IMPL": "upstream",
        "VLLM_DSPARK_REPLICATE_MARKOV_W1": "1",
        "VLLM_DSPARK_REPLICATE_MARKOV_W2": "1",
        "HF_HUB_OFFLINE": "1",
    }
    for name, expected in required.items():
        if environment.get(name) != expected:
            raise ProfileTransformError(
                f"source DeepSeek environment requires {name}={expected}"
            )
    preload = environment.get("LD_PRELOAD", "")
    for library in (
        "/usr/local/cuda/compat/libcuda.so.1",
        "/opt/sparkring/nccl/libnccl.so.2",
    ):
        if library not in preload.split(":"):
            raise ProfileTransformError(
                f"source DeepSeek environment LD_PRELOAD lacks {library}"
            )
    forbidden_names = {
        name
        for name, value in environment.items()
        if "lmcache" in name.lower() or "lmcache" in value.lower()
    }
    if forbidden_names:
        raise ProfileTransformError(
            "source DeepSeek environment contains foreign cache/model state: "
            + ", ".join(sorted(forbidden_names))
        )
    python_path = environment.get("PYTHONPATH", "")
    required_paths = ["/opt/sparkcache-src"]
    paths = [path for path in python_path.split(":") if path]
    environment["PYTHONPATH"] = ":".join(
        [*required_paths, *(path for path in paths if path not in required_paths)]
    )
    removed = {
        name
        for name in environment
        if name == "SPARK_CONTEXT_CACHE_ENABLE"
        or name.startswith("SPARK_Q40_")
        or name.startswith("SPARKRING_ATTEST_MODEL_")
        or name.startswith("SPARKRING_MODEL_")
    }
    for name in removed:
        environment.pop(name, None)
    explicit_unset = {
        name
        for name in environment.get("SPARKRING_EXPLICITLY_UNSET", "").split(",")
        if name
    }
    explicit_unset.update(removed)
    environment["SPARKRING_EXPLICITLY_UNSET"] = ",".join(sorted(explicit_unset))
    return [f"{name}={value}" for name, value in sorted(environment.items())]


def _validate_arguments(arguments: list[str]) -> int:
    serving = PROFILE["serving"]
    model_path = PROFILE["model"]["container_path"]
    if arguments[:2] != ["serve", model_path]:
        raise ProfileTransformError(
            f"source command must begin with 'serve {model_path}'"
        )
    exact = {
        "--tensor-parallel-size": serving["tensor_parallel_size"],
        "--nnodes": serving["nodes"],
        "--distributed-executor-backend": "mp",
        "--dtype": serving["dtype"],
        "--max-model-len": serving["max_model_len"],
        "--max-num-seqs": serving["max_num_seqs"],
        "--kv-cache-memory-bytes": serving["kv_cache_bytes_per_rank"],
        "--kv-cache-dtype": serving["kv_cache_dtype"],
        "--tokenizer-mode": serving["tokenizer_mode"],
        "--served-model-name": serving["served_model_name"],
    }
    for option, expected in exact.items():
        if _one(arguments, option) != str(expected):
            raise ProfileTransformError(
                f"source DeepSeek command requires {option} {expected}"
            )
    try:
        utilization = float(_one(arguments, "--gpu-memory-utilization"))
    except ValueError as error:
        raise ProfileTransformError(
            "source --gpu-memory-utilization must be numeric"
        ) from error
    if utilization != float(serving["gpu_memory_utilization"]):
        raise ProfileTransformError(
            "source DeepSeek command requires gpu-memory-utilization 0.70"
        )
    dcp = _optional_one(arguments, "--decode-context-parallel-size")
    if dcp not in (None, "1"):
        raise ProfileTransformError(
            "DeepSeek hybrid-memory-allocator block pages require DCP1"
        )
    pp = _optional_one(arguments, "--pipeline-parallel-size")
    if pp not in (None, "1"):
        raise ProfileTransformError("DeepSeek SparkCache requires PP1")
    block_size = _optional_one(arguments, "--block-size")
    if block_size not in (None, str(serving["block_size"])):
        raise ProfileTransformError("DeepSeek SparkCache requires block size 256")
    if "--enable-expert-parallel" in arguments:
        raise ProfileTransformError("DeepSeek e266 profile rejects expert parallelism")
    for forbidden in (
        "--disable-hybrid-kv-cache-manager",
        "--enforce-eager",
        "--disable-prefix-caching",
    ):
        if forbidden in arguments:
            raise ProfileTransformError(
                f"source command contains unsupported {forbidden}"
            )
    for required_flag in ("--enable-auto-tool-choice",):
        if arguments.count(required_flag) != 1:
            raise ProfileTransformError(f"source command requires {required_flag}")
    if _one(arguments, "--tool-call-parser") != "deepseek_v4":
        raise ProfileTransformError(
            "source DeepSeek command requires tool-call-parser deepseek_v4"
        )
    try:
        kernel = json.loads(_one(arguments, "--kernel-config"))
        speculative = json.loads(_one(arguments, "--speculative-config"))
    except json.JSONDecodeError as error:
        raise ProfileTransformError(
            "source DeepSeek JSON argument is invalid"
        ) from error
    if kernel.get("enable_cutedsl_warmup") is not False:
        raise ProfileTransformError("source DeepSeek kernel config must disable warmup")
    expected_speculative = {
        "method": serving["speculation_method"],
        "num_speculative_tokens": serving["speculation_tokens"],
        "moe_backend": serving["speculation_moe_backend"],
    }
    if any(
        speculative.get(key) != value for key, value in expected_speculative.items()
    ):
        raise ProfileTransformError("source DeepSeek DSpark configuration differs")
    if speculative.get("draft_sample_method", "greedy") != "greedy":
        raise ProfileTransformError("source DeepSeek DSpark sampling must be greedy")
    rank = _integer(arguments, "--node-rank")
    if not 0 <= rank < int(serving["nodes"]):
        raise ProfileTransformError("source DeepSeek node rank is out of range")
    headless = "--headless" in arguments
    if headless is (rank == 0):
        raise ProfileTransformError(
            "rank 0 must serve the API and ranks 1-3 must be headless"
        )
    if rank == 0:
        if _one(arguments, "--host") != "0.0.0.0":
            raise ProfileTransformError("rank 0 must listen on 0.0.0.0")
        _port(int(_one(arguments, "--port")), "source api port")
    elif _option_values(arguments, "--host") or _option_values(arguments, "--port"):
        raise ProfileTransformError("headless DeepSeek ranks must not bind the API")
    _port(int(_one(arguments, "--master-port")), "source master port")
    if not _one(arguments, "--master-addr"):
        raise ProfileTransformError("source DeepSeek master address is empty")
    existing_transfer = _optional_one(arguments, "--kv-transfer-config")
    if existing_transfer is not None and "lmcache" in existing_transfer.lower():
        raise ProfileTransformError("source command contains LMCache configuration")
    return rank


def build_kv_transfer_config(checkpoint_sha256: str) -> dict[str, Any]:
    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    cache = PROFILE["cache"]
    return {
        "kv_connector": "SparkContextCacheConnector",
        "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": cache["load_failure_policy"],
        "kv_connector_extra_config": {
            "spark_cache_root": cache["root"],
            "spark_cache_model_profile": PROFILE["cache_model_profile"],
            "spark_cache_target_checkpoint_sha256": checkpoint_sha256,
            "spark_cache_draft_policy": cache["draft_policy"],
            "spark_cache_scheduler_probe": "none",
            "spark_cache_store": True,
            "spark_cache_restore": True,
            "spark_cache_streaming_snapshots": False,
            "spark_cache_cuda_restore": False,
            "spark_cache_max_bytes": cache["max_bytes"],
            "spark_cache_low_watermark_bytes": cache["low_watermark_bytes"],
            "spark_cache_ttl_seconds": cache["ttl_seconds"],
            "spark_cache_min_span_tokens": cache["min_span_tokens"],
            "spark_cache_max_span_tokens": cache["max_span_tokens"],
        },
    }


def _reserved_ports(environment: Iterable[str]) -> frozenset[int]:
    reserved: set[int] = set()
    for name, raw in _environment_map(environment).items():
        if name in {"PORT", "MASTER_PORT"} or not _PORT_ENVIRONMENT_RE.search(name):
            continue
        try:
            value = int(raw)
        except ValueError as error:
            raise ProfileTransformError(
                f"environment port {name} is invalid"
            ) from error
        validated = _port(value, name)
        assert validated is not None
        reserved.add(validated)
    return frozenset(reserved)


def transform_inspection(
    inspection: dict[str, Any],
    *,
    checkpoint_sha256: str,
    api_port: int | None = None,
    master_port: int | None = None,
) -> dict[str, Any]:
    """Return one exact TP4/DCP1 source inspection with SparkCache enabled."""

    try:
        source_command = list(inspection["Config"]["Cmd"])
        source_environment = list(inspection["Config"].get("Env", ()))
    except (KeyError, TypeError) as error:
        raise ProfileTransformError("invalid Docker inspection record") from error
    source_image = str(inspection.get("Image", ""))
    if not source_image.startswith("sha256:") or len(source_image) != 71:
        raise ProfileTransformError("source image must be an immutable Docker image ID")
    _validate_source_mounts(inspection)
    arguments = _vllm_args(source_command)
    rank = _validate_arguments(arguments)
    transformed_environment = _validate_environment(source_environment)
    reserved = _reserved_ports(source_environment)
    api_port = _port(api_port, "api_port")
    master_port = _port(master_port, "master_port")
    effective_api = api_port or (int(_one(arguments, "--port")) if rank == 0 else None)
    effective_master = master_port or int(_one(arguments, "--master-port"))
    if effective_api is not None and effective_api == effective_master:
        raise ProfileTransformError("api_port and master_port must differ")
    for role, value in (("api_port", effective_api), ("master_port", effective_master)):
        if value is not None and value in reserved:
            raise ProfileTransformError(f"{role} {value} collides with a runtime port")
    result = _drop_option(arguments, "--kv-transfer-config")
    result = _drop_option(result, "--master-port")
    if rank == 0:
        result = _drop_option(result, "--port")
    if not _option_values(result, "--block-size"):
        result.extend(("--block-size", str(PROFILE["serving"]["block_size"])))
    result.extend(("--master-port", str(effective_master)))
    if rank == 0 and effective_api is not None:
        result.extend(("--port", str(effective_api)))
    result.extend(
        (
            "--kv-transfer-config",
            _compact_json(build_kv_transfer_config(checkpoint_sha256)),
        )
    )
    environment = _environment_map(transformed_environment)
    environment["MASTER_PORT"] = str(effective_master)
    if rank == 0 and effective_api is not None:
        environment["PORT"] = str(effective_api)
    transformed = copy.deepcopy(inspection)
    transformed["Config"]["Cmd"] = result
    transformed["Config"]["Env"] = [
        f"{name}={value}" for name, value in sorted(environment.items())
    ]
    labels = dict(transformed["Config"].get("Labels") or {})
    labels["org.sparkcache.deployment-profile"] = PROFILE_ID
    transformed["Config"]["Labels"] = labels
    return transformed
