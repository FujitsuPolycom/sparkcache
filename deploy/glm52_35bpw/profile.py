"""Launch transformation for the fixed-MTP4 GLM-5.2 3.5-bpw serving recipe.

SparkRing identifies this recipe as ``R7``. The input is one ``docker inspect``
record from its serving container. The output preserves rank, transport, JIT,
and topology settings owned by that container while replacing the model/cache
settings owned by ``profile.json``. No Docker, SSH, CUDA, or vLLM import occurs
in this module.
"""

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
    option_values,
    validate_port,
    vllm_arguments,
)


class ProfileTransformError(DeploymentContractError):
    """The source inspection or requested feature set is unsafe to launch."""


PROFILE_PATH = Path(__file__).with_name("profile.json")
PROFILE = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
if PROFILE.get("schema") != "sparkcache-deployment-profile/v1":
    raise RuntimeError(f"unsupported deployment profile: {PROFILE_PATH}")

PROFILE_ID = str(PROFILE["profile_id"])
DEFAULT_CHECKPOINT_SHA256 = str(PROFILE["model"]["index_sha256"])
DEFAULT_CACHE_ROOT = str(PROFILE["cache"]["root"])
MAX_BYTES = int(PROFILE["cache"]["max_bytes"])
LOW_WATERMARK_BYTES = int(PROFILE["cache"]["low_watermark_bytes"])
TTL_SECONDS = int(PROFILE["cache"]["ttl_seconds"])
SPARKCACHE_SOURCE_SHA256 = str(PROFILE["sparkcache"]["source_sha256"])
DEFAULT_STREAMING_LEASE_CONTRACT = (
    "/opt/sparkcache-src/sparkcache/runtime_patches/"
    "vllm-kv-block-lease-contract-e2666d9a6.json"
)

_SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_PORT_ENVIRONMENT_RE = re.compile(r"(?:^|_)PORT\d*\Z")


def _require_sha256(value: str, role: str) -> str:
    if _SHA256_RE.fullmatch(value) is None:
        raise ProfileTransformError(f"{role} must be 64 lowercase hexadecimal digits")
    return value


def _deployment_path(value: str, role: str) -> str:
    path = PurePosixPath(value)
    if not value or not path.is_absolute() or ".." in path.parts:
        raise ProfileTransformError(f"{role} must be an absolute container path")
    return value


def _cache_root(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not path.is_absolute()
        or len(path.parts) < 3
        or path.parts[1] != "cache"
        or ".." in path.parts
        or str(path) != value
    ):
        raise ProfileTransformError(
            "cache_root must be a normalized model-specific child of /cache"
        )
    return value


def _port(value: int | None, role: str) -> int | None:
    return validate_port(value, role, error_type=ProfileTransformError)


def _reserved_collective_ports(values: Iterable[str]) -> frozenset[int]:
    environment = _environment_map(values)
    reserved: set[int] = set()
    for name, raw_value in environment.items():
        if name in {"PORT", "MASTER_PORT"} or not _PORT_ENVIRONMENT_RE.search(name):
            continue
        try:
            value = int(raw_value)
        except ValueError as error:
            raise ProfileTransformError(
                f"source GLM-5.2 serving recipe R7 environment port {name} is not an integer"
            ) from error
        validated = _port(value, name)
        assert validated is not None
        if name == "SPARK_TP4_ALLGATHER_BASE_PORT":
            for slot in range(8):
                for endpoint in range(2):
                    expanded = _port(
                        validated + slot * 10 + endpoint,
                        name,
                    )
                    assert expanded is not None
                    reserved.add(expanded)
        else:
            reserved.add(validated)
    return frozenset(reserved)


def _option_values(arguments: list[str], option: str) -> list[str]:
    return option_values(arguments, option, error_type=ProfileTransformError)


def _drop_option(arguments: list[str], option: str) -> list[str]:
    return drop_option(arguments, option, error_type=ProfileTransformError)


def _single_existing_port(arguments: list[str], option: str) -> int | None:
    values = _option_values(arguments, option)
    if not values:
        return None
    if len(values) != 1:
        raise ProfileTransformError(f"source command must have at most one {option}")
    try:
        return _port(int(values[0]), option)
    except ValueError as error:
        raise ProfileTransformError(f"source {option} is not an integer") from error


_compact_json = compact_json
_vllm_args = vllm_arguments


def _require_option(arguments: list[str], option: str, expected: Any) -> None:
    if _option_values(arguments, option) != [str(expected)]:
        raise ProfileTransformError(
            f"source GLM-5.2 serving recipe R7 requires exactly one {option} {expected}"
        )


def _load_json_option(arguments: list[str], option: str) -> dict[str, Any]:
    values = _option_values(arguments, option)
    if len(values) != 1:
        raise ProfileTransformError(
            f"source GLM-5.2 serving recipe R7 requires exactly one {option}"
        )
    try:
        value = json.loads(values[0])
    except json.JSONDecodeError as error:
        raise ProfileTransformError(f"source {option} is not valid JSON") from error
    if not isinstance(value, dict):
        raise ProfileTransformError(f"source {option} must be a JSON object")
    return value


def _validate_r7_arguments(arguments: list[str]) -> None:
    """Validate the accepted serving contract without rebuilding its profile."""

    serving = PROFILE["serving"]
    model = PROFILE["model"]
    if arguments[1] != model["container_path"]:
        raise ProfileTransformError(
            f"source GLM-5.2 serving recipe R7 model path must be {model['container_path']}"
        )
    required_options = (
        ("--distributed-executor-backend", "mp"),
        ("--nnodes", 4),
        ("--tensor-parallel-size", serving["tensor_parallel_size"]),
        ("--decode-context-parallel-size", serving["decode_context_parallel_size"]),
        ("--dcp-comm-backend", serving["dcp_backend"]),
        (
            "--dcp-kv-cache-interleave-size",
            serving["dcp_kv_cache_interleave_size"],
        ),
        ("--quantization", "exl3"),
        ("--moe-backend", "b12x"),
        ("--attention-backend", "B12X_MLA_SPARSE"),
        ("--kv-cache-dtype", serving["kv_cache_dtype"]),
        ("--max-model-len", serving["max_model_len"]),
        ("--kv-cache-memory-bytes", serving["kv_cache_bytes_per_rank"]),
        ("--max-num-seqs", serving["max_num_seqs"]),
        ("--max-num-batched-tokens", serving["max_num_batched_tokens"]),
        ("--load-format", "instanttensor"),
        ("--gpu-memory-utilization", 0.85),
        ("--host", "0.0.0.0"),
        ("--served-model-name", serving["served_model_name"]),
        ("--max-cudagraph-capture-size", serving["max_query_rows"]),
    )
    for option, expected in required_options:
        _require_option(arguments, option, expected)
    node_ranks = _option_values(arguments, "--node-rank")
    if len(node_ranks) != 1:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 requires one --node-rank"
        )
    try:
        node_rank = int(node_ranks[0])
    except ValueError as error:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 node rank is not an integer"
        ) from error
    if not 0 <= node_rank < 4:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 node rank must be in [0, 4)"
        )
    master_addresses = _option_values(arguments, "--master-addr")
    if len(master_addresses) != 1 or not master_addresses[0]:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 requires one master address"
        )

    speculative = _load_json_option(arguments, "--speculative-config")
    expected_speculation = {
        "model": arguments[1],
        "method": "mtp",
        "num_speculative_tokens": int(serving["mtp_tokens"]),
        "draft_tensor_parallel_size": int(serving["tensor_parallel_size"]),
        "quantization": "exl3",
        "moe_backend": "b12x",
        "attention_backend": "B12X_MLA_SPARSE",
        "use_local_argmax_reduction": False,
        "draft_sample_method": "greedy",
    }
    for key, expected in expected_speculation.items():
        actual = speculative.get(key)
        if actual != expected or (
            key == "use_local_argmax_reduction" and actual is not False
        ):
            raise ProfileTransformError(
                f"source GLM-5.2 serving recipe R7 speculative config requires {key}={expected!r}"
            )
    if "adaptive_speculative_tokens_window" in speculative:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 speculative config must use"
            " fixed MTP4, not adaptive depth"
        )

    compilation = _load_json_option(arguments, "--compilation-config")
    if compilation.get("cudagraph_mode") != serving["execution_mode"]:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 compilation config must use"
            " FULL_AND_PIECEWISE"
        )
    if compilation.get("cudagraph_capture_sizes") != list(
        range(1, int(serving["max_query_rows"]) + 1)
    ):
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 compilation config must preserve"
            " CUDA graph sizes Q1 through Q40"
        )

    for flag in (
        "--enable-chunked-prefill",
        "--no-async-scheduling",
    ):
        if arguments.count(flag) != 1:
            raise ProfileTransformError(
                f"source GLM-5.2 serving recipe R7 requires exactly one {flag}"
            )
    if "--disable-prefix-caching" in arguments:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 must use its native prefix-cache default or"
            " --enable-prefix-caching"
        )
    for preserved_flag in (
        "--enable-prefix-caching",
        "--disable-hybrid-kv-cache-manager",
    ):
        if arguments.count(preserved_flag) > 1:
            raise ProfileTransformError(
                f"source GLM-5.2 serving recipe R7 has duplicate {preserved_flag}"
            )
    block_sizes = _option_values(arguments, "--block-size")
    if block_sizes not in ([], [str(serving["block_size"])]):
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 requires implicit block size 64 or exactly one"
            " --block-size 64"
        )
    if "--enforce-eager" in arguments:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 must retain CUDA graphs"
        )


def build_kv_transfer_config(
    *,
    checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    cache_root: str = DEFAULT_CACHE_ROOT,
    streaming_snapshots: bool = False,
    streaming_native_library: str | None = None,
    streaming_native_library_sha256: str | None = None,
    streaming_timing: bool = False,
    native_restore: bool = False,
    native_restore_library: str | None = None,
    native_restore_library_sha256: str | None = None,
    native_restore_arena_bytes: int = 128 * 1024 * 1024,
    native_restore_io_workers: int = 8,
) -> dict[str, Any]:
    """Build the complete connector argument for one deployment instance."""

    checkpoint_sha256 = _require_sha256(checkpoint_sha256, "checkpoint_sha256")
    cache_root = _cache_root(cache_root)
    extra: dict[str, Any] = {
        "spark_cache_root": cache_root,
        "spark_cache_model_profile": str(PROFILE["cache_model_profile"]),
        "spark_cache_target_checkpoint_sha256": checkpoint_sha256,
        "spark_cache_draft_policy": "colocated_target",
        "spark_cache_store": True,
        "spark_cache_restore": True,
        "spark_cache_scheduler_probe": "none",
        "spark_cache_streaming_snapshots": bool(streaming_snapshots),
        "spark_cache_native_restore": bool(native_restore),
        "spark_cache_max_bytes": MAX_BYTES,
        "spark_cache_low_watermark_bytes": LOW_WATERMARK_BYTES,
        "spark_cache_ttl_seconds": TTL_SECONDS,
        "spark_cache_min_span_tokens": 256,
        "spark_cache_max_span_tokens": int(PROFILE["serving"]["max_model_len"]),
    }

    if streaming_snapshots:
        if streaming_native_library is None or streaming_native_library_sha256 is None:
            raise ProfileTransformError(
                "streaming snapshots require their native library path and SHA-256"
            )
        extra.update(
            {
                "spark_cache_streaming_native_library": _deployment_path(
                    streaming_native_library, "streaming_native_library"
                ),
                "spark_cache_streaming_native_library_sha256": _require_sha256(
                    streaming_native_library_sha256,
                    "streaming_native_library_sha256",
                ),
                "spark_cache_streaming_lease_contract": (
                    DEFAULT_STREAMING_LEASE_CONTRACT
                ),
                "spark_cache_streaming_timing": int(bool(streaming_timing)),
            }
        )
    elif any(
        value is not None
        for value in (streaming_native_library, streaming_native_library_sha256)
    ) or streaming_timing:
        raise ProfileTransformError(
            "streaming native options require streaming_snapshots=True"
        )

    if native_restore:
        if native_restore_library is None or native_restore_library_sha256 is None:
            raise ProfileTransformError(
                "native restore requires its placement library path and SHA-256"
            )
        if native_restore_arena_bytes not in {
            64 * 1024 * 1024,
            128 * 1024 * 1024,
            256 * 1024 * 1024,
        }:
            raise ProfileTransformError(
                "native_restore_arena_bytes must be 64, 128, or 256 MiB"
            )
        if not 1 <= native_restore_io_workers <= 32:
            raise ProfileTransformError("native_restore_io_workers must be in [1, 32]")
        extra.update(
            {
                "spark_cache_native_library": _deployment_path(
                    native_restore_library, "native_restore_library"
                ),
                "spark_cache_native_library_sha256": _require_sha256(
                    native_restore_library_sha256,
                    "native_restore_library_sha256",
                ),
                "spark_cache_native_arena_bytes": native_restore_arena_bytes,
                "spark_cache_native_io_workers": native_restore_io_workers,
            }
        )
    elif any(
        value is not None
        for value in (native_restore_library, native_restore_library_sha256)
    ):
        raise ProfileTransformError(
            "native restore library options require native_restore=True"
        )

    return {
        "kv_connector": "SparkContextCacheConnector",
        "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": extra,
    }


def transform_vllm_args(
    source_command: list[str],
    kv_transfer_config: dict[str, Any],
    *,
    api_port: int | None = None,
    master_port: int | None = None,
    reserved_ports: frozenset[int] = frozenset(),
) -> list[str]:
    """Return the exact GLM-5.2 recipe-R7 command with rank/transport args."""

    arguments = _vllm_args(list(source_command))
    if len(arguments) < 2 or arguments[0] != "serve" or arguments[1].startswith("-"):
        raise ProfileTransformError(
            "source command must begin with 'serve MODEL_PATH'"
        )
    api_port = _port(api_port, "api_port")
    master_port = _port(master_port, "master_port")
    effective_api_port = api_port or _single_existing_port(arguments, "--port")
    effective_master_port = master_port or _single_existing_port(
        arguments, "--master-port"
    )
    if (
        effective_api_port is not None
        and effective_master_port is not None
        and effective_api_port == effective_master_port
    ):
        raise ProfileTransformError("api_port and master_port must differ")
    for role, value in (
        ("api_port", effective_api_port),
        ("master_port", effective_master_port),
    ):
        if value is not None and value in reserved_ports:
            raise ProfileTransformError(
                f"{role} {value} collides with a GLM-5.2 recipe-R7 collective port"
            )

    _validate_r7_arguments(arguments)
    add_explicit_block_size = not _option_values(arguments, "--block-size")
    result = list(arguments)
    result = _drop_option(result, "--kv-transfer-config")
    if api_port is not None:
        result = _drop_option(result, "--port")
    if master_port is not None:
        result = _drop_option(result, "--master-port")
    if add_explicit_block_size:
        result.extend(("--block-size", str(PROFILE["serving"]["block_size"])))
    result.extend(("--kv-transfer-config", _compact_json(kv_transfer_config)))
    if api_port is not None:
        result.extend(("--port", str(api_port)))
    if master_port is not None:
        result.extend(("--master-port", str(master_port)))
    if any("lmcache" in argument.lower() for argument in result):
        raise ProfileTransformError(
            "source command contains an unsupported LMCache-specific argument"
        )
    return result


def _environment_map(values: Iterable[str]) -> dict[str, str]:
    return environment_map(
        values,
        require_unique=False,
        error_type=ProfileTransformError,
    )


def _transform_environment(
    source: Iterable[str], *, api_port: int | None, master_port: int | None
) -> list[str]:
    environment = _environment_map(source)
    explicit_unset = {
        name
        for name in environment.get("SPARKRING_EXPLICITLY_UNSET", "").split(",")
        if name
    }
    serving = PROFILE["serving"]
    required = {
        "SPARKRING_ATTEST_MODEL_REPOSITORY": str(PROFILE["model"]["repository"]),
        "SPARKRING_ATTEST_MODEL_REVISION": str(PROFILE["model"]["revision"]),
        "SPARKRING_ATTEST_MODEL_CONFIG_SHA256": str(
            PROFILE["model"]["config_sha256"]
        ),
        "SPARKRING_ATTEST_MODEL_INDEX_SHA256": str(
            PROFILE["model"]["index_sha256"]
        ),
        "KV_FP8_ROPE": "1",
        "VLLM_NVFP4_MLA_DYNAMIC_SCALE": "1",
        "VLLM_EXL3_PREFILL_CAPACITY": str(serving["max_num_batched_tokens"]),
        "VLLM_SPARK_MAX_QUERY_ROWS": str(serving["max_query_rows"]),
        "VLLM_SPARK_MTP_MODE_ID": "fixed-mtp4",
        "VLLM_SPARK_MTP_TOKENS": str(serving["mtp_tokens"]),
        "SPARK_ADAPTIVE_MTP_CONTROL": "0",
        "VLLM_SPARK_MTP_ADAPTIVE_WINDOW": "0",
        "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT": "0",
        "VLLM_B12X_MLA_CKV_GATHER": "1",
        "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS": str(
            serving["max_model_len"]
        ),
        "VLLM_SPARK_TP4_MODE": "custom",
        "SPARK_TP4_LIBRARY": "/opt/sparkring/spark_transport/"
        "libspark_transport_capi.so",
        "ONLINE_QUANT": "exl3-b6",
        "VLLM_EXL3_ONLINE_CACHE_MODE": "readwrite",
        "VLLM_EXL3_ONLINE_TRELLIS_BITS": "6",
        "SPARK_Q40_EXACT_STATE_CHECKPOINT": str(PROFILE["model"]["revision"]),
    }
    for name, expected in required.items():
        if environment.get(name) != expected:
            raise ProfileTransformError(
                f"source GLM-5.2 serving recipe R7 environment requires {name}={expected}"
            )
        if name in explicit_unset:
            raise ProfileTransformError(
                f"source GLM-5.2 serving recipe R7 entrypoint would unset required variable {name}"
            )
    public_q40 = environment.get(
        "SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256"
    ) == "8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2"
    if not public_q40 or "SPARK_Q35_EXACT_STATE_EXPECTED_EXL3_SHA256" in environment:
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 environment must select the public target-only"
            " exact-Q40 state"
        )

    removed = {
        name
        for name, value in environment.items()
        if name == "SPARK_CONTEXT_CACHE_ENABLE"
        or "lmcache" in name.lower()
        or (
            name not in {"PYTHONPATH", "SPARKRING_EXPLICITLY_UNSET"}
            and "lmcache" in value.lower()
        )
    }
    for name in removed:
        environment.pop(name, None)

    python_path = environment.get("PYTHONPATH", "")
    paths = [path for path in python_path.split(":") if "lmcache" not in path.lower()]
    required_paths = ["/opt/sparkcache-src"]
    environment["PYTHONPATH"] = ":".join(
        [*required_paths, *(path for path in paths if path not in required_paths)]
    )

    if api_port is not None:
        environment["PORT"] = str(api_port)
    if master_port is not None:
        environment["MASTER_PORT"] = str(master_port)

    explicit_unset.update(removed)
    explicit_unset.discard("MASTER_PORT")
    environment["SPARKRING_EXPLICITLY_UNSET"] = ",".join(sorted(explicit_unset))
    return [f"{name}={value}" for name, value in sorted(environment.items())]


def transform_inspection(
    inspection: dict[str, Any],
    *,
    checkpoint_sha256: str = DEFAULT_CHECKPOINT_SHA256,
    cache_root: str = DEFAULT_CACHE_ROOT,
    streaming_snapshots: bool = False,
    streaming_native_library: str | None = None,
    streaming_native_library_sha256: str | None = None,
    streaming_timing: bool = False,
    native_restore: bool = False,
    native_restore_library: str | None = None,
    native_restore_library_sha256: str | None = None,
    native_restore_arena_bytes: int = 128 * 1024 * 1024,
    native_restore_io_workers: int = 8,
    api_port: int | None = None,
    master_port: int | None = None,
) -> dict[str, Any]:
    """Return a Docker inspection record carrying the complete GLM contract."""

    try:
        source_command = list(inspection["Config"]["Cmd"])
        source_environment = list(inspection["Config"].get("Env", ()))
    except (KeyError, TypeError) as error:
        raise ProfileTransformError("invalid docker inspect record") from error
    model_mounts = [
        mount
        for mount in inspection.get("Mounts", ())
        if isinstance(mount, dict)
        and mount.get("Destination") == PROFILE["model"]["container_path"]
    ]
    if (
        len(model_mounts) != 1
        or model_mounts[0].get("Type") != "bind"
        or model_mounts[0].get("RW") is not False
    ):
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 model path must have one read-only bind mount"
        )
    environment_map = _environment_map(source_environment)
    source_image = str(inspection.get("Image", ""))
    if (
        not source_image.startswith("sha256:")
        or environment_map.get("SPARKRING_IMAGE_DIGEST") != source_image
        or environment_map.get("SPARK_Q40_EXACT_STATE_IMAGE_ID") != source_image
    ):
        raise ProfileTransformError(
            "source GLM-5.2 serving recipe R7 image must match its CUDA-graph"
            " Q40 and SparkRing image identity"
        )
    kv_transfer_config = build_kv_transfer_config(
        checkpoint_sha256=checkpoint_sha256,
        cache_root=cache_root,
        streaming_snapshots=streaming_snapshots,
        streaming_native_library=streaming_native_library,
        streaming_native_library_sha256=streaming_native_library_sha256,
        streaming_timing=streaming_timing,
        native_restore=native_restore,
        native_restore_library=native_restore_library,
        native_restore_library_sha256=native_restore_library_sha256,
        native_restore_arena_bytes=native_restore_arena_bytes,
        native_restore_io_workers=native_restore_io_workers,
    )
    transformed = copy.deepcopy(inspection)
    transformed["Config"]["Cmd"] = transform_vllm_args(
        source_command,
        kv_transfer_config,
        api_port=api_port,
        master_port=master_port,
        reserved_ports=_reserved_collective_ports(source_environment),
    )
    transformed["Config"]["Env"] = _transform_environment(
        source_environment,
        api_port=api_port,
        master_port=master_port,
    )
    labels = dict(transformed["Config"].get("Labels") or {})
    labels["org.sparkcache.deployment-profile"] = PROFILE_ID
    transformed["Config"]["Labels"] = labels
    return transformed
