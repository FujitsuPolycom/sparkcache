"""Configuration parsing and immutable cache-identity construction.

Both the scheduler and worker connector roles share one parsing path.
``parse_connector_config`` reads all SparkCache settings from a
``VllmConfig``-like object — extra-config keys, environment variables,
parallel degrees, model profile, and KV-cache group topology. It validates
the fail-closed deployment contract and returns an immutable
:class:`ConnectorConfig` carrying every value the connector needs at
construction.

The immutable cache-identity base (``identity_base``) is built here so both
roles construct identical wire bytes for the same deployment. ``SHA256_RE``
is the shared lowercase digest syntax used by construction and streaming
commit validation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from sparkcache.spark_context_cache_profiles import (
    ProfileError,
    resolve_profile,
)
from sparkcache.spark_context_cache_store import (
    CacheIdentity,
    CapacityPolicy,
)
from sparkcache.streaming.feature_gate import (
    ENVIRONMENT_KEY as _STREAMING_SNAPSHOTS_ENV,
    EXTRA_CONFIG_KEY as _STREAMING_SNAPSHOTS_CONFIG,
    is_enabled as _streaming_snapshots_enabled,
)

SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
_NATIVE_ARENA_BYTES = frozenset(
    {64 * 1024 * 1024, 128 * 1024 * 1024, 256 * 1024 * 1024}
)


def _nonnegative_config_int(value: Any, label: str) -> int:
    try:
        if isinstance(value, (bool, float)):
            raise ValueError("Boolean and floating values are not integer limits")
        parsed = int(value)
    except (OverflowError, TypeError, ValueError) as error:
        raise RuntimeError(f"spark-context-cache: {label} must be an integer") from error
    if parsed < 0:
        raise RuntimeError(f"spark-context-cache: {label} must be non-negative")
    return parsed


def _freeze_config_value(value: Any) -> Any:
    """Recursively freeze connector metadata shared across runtime roles."""

    if isinstance(value, Mapping):
        return MappingProxyType(
            {key: _freeze_config_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_config_value(item) for item in value)
    return value


def _cache_spec_reuse_policy(spec: Any) -> tuple[str, int | None]:
    """Classify one layer's supported persistent-cache reuse semantics."""

    class_names = {base.__name__ for base in type(spec).__mro__}
    if "SlidingWindowSpec" in class_names:
        window = int(getattr(spec, "sliding_window", 0) or 0)
        if window <= 1:
            raise RuntimeError(
                "spark-context-cache: sliding KV-cache window must exceed one token"
            )
        return "sliding", window
    if "FullAttentionSpec" in class_names:
        return "full", None
    if "MambaSpec" in class_names:
        mode = str(getattr(spec, "mamba_cache_mode", "none"))
        if mode != "align":
            raise RuntimeError(
                "spark-context-cache: recurrent block-page restore requires"
                f" mamba_cache_mode 'align' (configured: {mode!r})"
            )
        return "recurrent_align", None
    raise RuntimeError(
        "spark-context-cache: unsupported block-page KV-cache spec "
        f"{type(spec).__name__}"
    )


def _group_reuse_policy(
    spec: Any, layer_names: Sequence[str]
) -> tuple[str, int | None]:
    """Resolve one engine group's common supported reuse policy."""

    if not layer_names:
        raise RuntimeError("spark-context-cache: KV-cache group has no layers")
    per_layer = getattr(spec, "kv_cache_specs", None)
    policies = {
        _cache_spec_reuse_policy(
            per_layer[name]
            if isinstance(per_layer, dict) and name in per_layer
            else spec
        )
        for name in layer_names
    }
    if len(policies) != 1:
        rendered = sorted(
            policy if window is None else f"{policy}:{window}"
            for policy, window in policies
        )
        raise RuntimeError(
            "spark-context-cache: one KV-cache group mixes incompatible reuse"
            f" policies: {rendered}"
        )
    return policies.pop()


def _recurrent_state_identity(
    spec: Any, layer_names: Sequence[str]
) -> dict[str, Any] | None:
    """Return identity fields that affect Mamba-align checkpoint semantics."""

    per_layer = getattr(spec, "kv_cache_specs", None)
    layer_specs = tuple(
        per_layer[name]
        if isinstance(per_layer, dict) and name in per_layer
        else spec
        for name in layer_names
    )
    recurrent = tuple(
        layer_spec
        for layer_spec in layer_specs
        if "MambaSpec" in {base.__name__ for base in type(layer_spec).__mro__}
    )
    if not recurrent:
        return None
    fields = (
        "mamba_cache_mode",
        "tokens_per_state",
        "num_speculative_blocks",
        "num_prefill_checkpoint_blocks",
    )
    values = {
        field: {getattr(layer_spec, field, None) for layer_spec in recurrent}
        for field in fields
    }
    inconsistent = [field for field, choices in values.items() if len(choices) != 1]
    if inconsistent:
        raise RuntimeError(
            "spark-context-cache: recurrent group mixes incompatible state"
            " geometry: " + ", ".join(inconsistent)
        )
    return {field: next(iter(choices)) for field, choices in values.items()}


def kv_group_topology(kv_cache_config: Any) -> tuple[dict[str, Any], ...]:
    groups = tuple(getattr(kv_cache_config, "kv_cache_groups", ()) or ())
    topology = []
    for group_index, group in enumerate(groups):
        spec = getattr(group, "kv_cache_spec", None)
        layers = tuple(sorted(getattr(group, "layer_names", ()) or ()))
        reuse_policy, reuse_window_tokens = _group_reuse_policy(spec, layers)
        group_identity = {
            "group": group_index,
            "spec": type(spec).__name__,
            "block_size": int(getattr(spec, "block_size", 0) or 0),
            "storage_block_size": int(getattr(spec, "storage_block_size", 0) or 0),
            "page_size_bytes": int(getattr(spec, "page_size_bytes", 0) or 0),
            "reuse_policy": reuse_policy,
            "reuse_window_tokens": reuse_window_tokens,
            "eagle": bool(getattr(group, "is_eagle_group", False)),
            "layers": layers,
        }
        recurrent_state = _recurrent_state_identity(spec, layers)
        if recurrent_state is not None:
            group_identity["recurrent_state"] = recurrent_state
        topology.append(group_identity)
    return tuple(topology)


def kv_group_topology_digest(kv_cache_config: Any) -> str:
    encoded = json.dumps(
        kv_group_topology(kv_cache_config),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class ConnectorConfig:
    """Immutable result of parsing all SparkCache connector settings.

    Every core connector field is determined solely by the vLLM config,
    extra-config keys, environment variables, and model profile supplied at
    construction. The optional streaming factory owns its native artifact,
    vLLM-root, lease-contract, and timing settings and validates them while the
    connector is still starting. Both scheduler and worker roles receive the
    same :class:`ConnectorConfig` for the same deployment, so cache-identity
    wire bytes are identical.
    """

    tp_degree: int
    dcp_degree: int
    block_size: int
    profile: Any
    storage_mode: str
    group_topology: tuple[Mapping[str, Any], ...]
    chunk_tokens: int
    root: str
    capacity_policy: CapacityPolicy
    min_span: int
    max_span: int
    store_enabled: bool
    restore_enabled: bool
    streaming_snapshots_enabled: bool
    native_restore_enabled: bool
    native_library_path: str
    native_library_sha256: str
    native_arena_bytes: int
    native_io_workers: int
    scheduler_probe: str
    identity_base: Mapping[str, Any]
    load_thread_limit: int
    max_pending_restores: int

    def build_identity(self, shard_rank: int, tp_shard_rank: int) -> CacheIdentity:
        """Construct a :class:`CacheIdentity` for a DCP shard rank.

        The connector resolves scheduler-versus-worker physical rank ownership
        before calling this interface. Both shard ranks are therefore explicit
        and no unowned storage namespace can be produced.
        """
        return CacheIdentity(
            dcp_shard_rank=shard_rank,
            tp_shard_rank=tp_shard_rank,
            **self.identity_base,
        )


def parse_connector_config(
    vllm_config: Any,
    kv_transfer_config: Any,
    kv_cache_config: Any,
) -> ConnectorConfig:
    """Parse and validate all SparkCache settings at construction time.

    Explicit validation failures raise :class:`RuntimeError` with the
    ``spark-context-cache:`` prefix. Direct integer conversion for span and
    load-thread settings retains Python's :class:`ValueError` behavior.
    """

    block_size = vllm_config.cache_config.block_size
    parallel = vllm_config.parallel_config
    tp_degree = max(1, getattr(parallel, "tensor_parallel_size", 1))
    dcp_degree = max(1, getattr(parallel, "decode_context_parallel_size", 1))
    if tp_degree % dcp_degree:
        raise RuntimeError(
            "spark-context-cache: decode context parallel size must divide"
            f" tensor parallel size (tp={tp_degree},"
            f" dcp={dcp_degree})"
        )
    pp_degree = max(1, getattr(parallel, "pipeline_parallel_size", 1))
    if pp_degree > 1:
        # CacheIdentity has no pipeline-stage field. Two stages would otherwise
        # publish different tensors into one manifest namespace.
        raise RuntimeError(
            "spark-context-cache: pipeline parallelism is unsupported;"
            f" pipeline_parallel_size={pp_degree}"
        )
    extra = kv_transfer_config.get_from_extra_config
    profile_name = str(
        extra(
            "spark_cache_model_profile",
            os.environ.get("SPARK_CONTEXT_CACHE_MODEL_PROFILE", "glm52-nvfp4"),
        )
    )
    try:
        profile = resolve_profile(profile_name)
    except ProfileError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    storage_mode = profile.storage_mode
    group_topology = kv_group_topology(kv_cache_config)
    if storage_mode == "block_pages_v1" and not group_topology:
        raise RuntimeError(
            "spark-context-cache: block-page storage requires KV-cache groups"
        )
    chunk_tokens = profile.chunk_tokens
    load_policy = str(
        getattr(kv_transfer_config, "kv_load_failure_policy", "recompute")
    )
    if load_policy != "recompute":
        # Every restore failure must become a clean re-prefill. Any other vLLM
        # policy can surface a failed restore as a request error.
        raise RuntimeError(
            "spark-context-cache: kv_load_failure_policy must be"
            f" 'recompute' (configured: {load_policy!r})"
        )
    root = extra(
        "spark_cache_root",
        os.environ.get("SPARK_CONTEXT_CACHE_ROOT", "/cache/context"),
    )
    max_bytes = _nonnegative_config_int(
        extra(
            "spark_cache_max_bytes",
            os.environ.get("SPARK_CONTEXT_CACHE_MAX_BYTES", "0"),
        ),
        "spark_cache_max_bytes",
    )
    default_low_bytes = max_bytes * 9 // 10 if max_bytes else 0
    low_watermark_bytes = _nonnegative_config_int(
        extra(
            "spark_cache_low_watermark_bytes",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_LOW_WATERMARK_BYTES",
                str(default_low_bytes),
            ),
        ),
        "spark_cache_low_watermark_bytes",
    )
    ttl_seconds = _nonnegative_config_int(
        extra(
            "spark_cache_ttl_seconds",
            os.environ.get("SPARK_CONTEXT_CACHE_TTL_SECONDS", "0"),
        ),
        "spark_cache_ttl_seconds",
    )
    try:
        capacity_policy = CapacityPolicy(
            max_bytes=max_bytes,
            low_watermark_bytes=low_watermark_bytes,
            ttl_seconds=ttl_seconds,
        )
    except ValueError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    min_span = int(
        extra(
            "spark_cache_min_span_tokens",
            os.environ.get("SPARK_CONTEXT_CACHE_MIN_SPAN", "1024"),
        )
    )
    model_max = int(getattr(vllm_config.model_config, "max_model_len", 0) or 0)
    default_max_span = str(model_max if model_max > 0 else 1 << 30)
    max_span = int(
        extra(
            "spark_cache_max_span_tokens",
            os.environ.get("SPARK_CONTEXT_CACHE_MAX_SPAN", default_max_span),
        )
    )
    store_enabled = extra(
        "spark_cache_store",
        os.environ.get("SPARK_CONTEXT_CACHE_STORE", "1"),
    ) in (1, "1", True, "true")
    restore_enabled = extra(
        "spark_cache_restore",
        os.environ.get("SPARK_CONTEXT_CACHE_RESTORE", "1"),
    ) in (1, "1", True, "true")
    try:
        streaming_snapshots_enabled = _streaming_snapshots_enabled(
            extra(
                _STREAMING_SNAPSHOTS_CONFIG,
                os.environ.get(_STREAMING_SNAPSHOTS_ENV, "0"),
            )
        )
    except ValueError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    if storage_mode == "block_pages_v1" and streaming_snapshots_enabled:
        raise RuntimeError(
            "spark-context-cache: block-page storage does not support"
            " streaming snapshots"
        )
    native_restore_enabled = extra(
        "spark_cache_native_restore",
        os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_RESTORE", "0"),
    ) in (1, "1", True, "true")
    native_library_path = str(
        extra(
            "spark_cache_native_library",
            os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_LIBRARY", ""),
        )
        or ""
    )
    native_library_sha256 = str(
        extra(
            "spark_cache_native_library_sha256",
            os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_LIBRARY_SHA256", ""),
        )
        or ""
    )
    native_arena_raw = str(
        extra(
            "spark_cache_native_arena_bytes",
            os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_ARENA_BYTES", ""),
        )
        or ""
    )
    try:
        native_arena_bytes = int(native_arena_raw) if native_arena_raw else 0
    except ValueError as error:
        if native_restore_enabled:
            raise RuntimeError(
                "spark-context-cache: native restore requires an integer arena size"
            ) from error
        native_arena_bytes = 0
    native_workers_raw = extra(
        "spark_cache_native_io_workers",
        os.environ.get("SPARK_CONTEXT_CACHE_NATIVE_IO_WORKERS", "8"),
    )
    try:
        native_io_workers = int(native_workers_raw)
    except (TypeError, ValueError) as error:
        if native_restore_enabled:
            raise RuntimeError(
                "spark-context-cache: native restore requires an integer"
                " IO worker count"
            ) from error
        native_io_workers = 8
    if native_restore_enabled:
        library_path = Path(native_library_path)
        if (
            not native_library_path
            or not library_path.is_absolute()
            or SHA256_RE.fullmatch(native_library_sha256) is None
            or native_arena_bytes not in _NATIVE_ARENA_BYTES
        ):
            raise RuntimeError(
                "spark-context-cache: native restore requires an"
                " absolute library path, a 64-character lowercase"
                " SHA-256, and arena bytes equal to 64, 128, or 256 MiB"
            )
        if not 1 <= native_io_workers <= 32:
            raise RuntimeError(
                "spark-context-cache: native restore IO workers must be in [1, 32]"
            )
    draft_policy = extra(
        "spark_cache_draft_policy",
        os.environ.get(
            "SPARK_CONTEXT_CACHE_DRAFT_POLICY",
            profile.default_draft_kv_policy,
        ),
    )
    target_id = str(
        extra(
            "spark_cache_target_checkpoint_sha256",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_TARGET_CHECKPOINT_SHA256",
                "",
            ),
        )
        or ""
    )
    if SHA256_RE.fullmatch(target_id) is None:
        raise RuntimeError(
            "spark-context-cache: target checkpoint identity must be a"
            " 64-character lowercase SHA-256; set"
            " spark_cache_target_checkpoint_sha256"
        )
    draft_id = str(
        extra(
            "spark_cache_draft_checkpoint_sha256",
            os.environ.get(
                "SPARK_CONTEXT_CACHE_DRAFT_CHECKPOINT_SHA256",
                "",
            ),
        )
        or ""
    )
    if str(draft_policy) == "colocated_target":
        if draft_id and draft_id != target_id:
            raise RuntimeError(
                "spark-context-cache: colocated_target draft state must"
                " use the target checkpoint identity; omit"
                " spark_cache_draft_checkpoint_sha256"
            )
        draft_id = target_id
    elif SHA256_RE.fullmatch(draft_id) is None:
        raise RuntimeError(
            "spark-context-cache: separate draft checkpoint identity must"
            " be a 64-character lowercase SHA-256; set"
            " spark_cache_draft_checkpoint_sha256"
        )
    quantization_layout = profile.quantization_layout
    if storage_mode == "block_pages_v1":
        quantization_layout += ":" + kv_group_topology_digest(kv_cache_config)
    record_schema = (
        ("target_ckv", "logical_positions")
        if storage_mode == "block_pages_v1"
        else ()
    )
    identity_base: dict[str, Any] = dict(
        target_checkpoint=target_id,
        draft_checkpoint=draft_id,
        quantization_layout=quantization_layout,
        rope_layout=profile.rope_layout,
        tp_degree=tp_degree,
        dcp_degree=dcp_degree,
        chunk_tokens=chunk_tokens,
        boundary_hidden_policy=profile.boundary_hidden_policy,
        draft_kv_policy=str(draft_policy),
    )
    if record_schema:
        identity_base["record_schema"] = record_schema
    try:
        profile.validate_for_deployment(
            dcp_degree=dcp_degree,
            block_size=block_size,
            min_span_tokens=min_span,
            native_restore=native_restore_enabled,
        )
    except ProfileError as error:
        raise RuntimeError(f"spark-context-cache: {error}") from error
    # "tp0" probes rank 0's local manifest store and therefore requires the
    # scheduler to share that worker's cache root. "none" admits on worker
    # quorum alone when the scheduler has no rank-local cache filesystem.
    scheduler_probe = str(
        extra(
            "spark_cache_scheduler_probe",
            os.environ.get("SPARK_CONTEXT_CACHE_SCHEDULER_PROBE", "tp0"),
        )
    )
    if scheduler_probe not in ("tp0", "none"):
        raise RuntimeError(
            "spark-context-cache: spark_cache_scheduler_probe must be"
            f" 'tp0' or 'none' (configured: {scheduler_probe!r})"
        )
    load_thread_limit = min(
        2,
        max(
            1,
            int(
                extra(
                    "spark_cache_load_threads",
                    os.environ.get("SPARK_CONTEXT_CACHE_LOAD_THREADS", "1"),
                )
            ),
        ),
    )
    if native_restore_enabled:
        load_thread_limit = 1
    max_pending_restores_raw = extra(
        "spark_cache_max_pending_restores",
        os.environ.get("SPARK_CONTEXT_CACHE_MAX_PENDING_RESTORES", "64"),
    )
    try:
        if isinstance(max_pending_restores_raw, bool):
            raise ValueError("Boolean values are not integer limits")
        max_pending_restores = int(max_pending_restores_raw)
    except (TypeError, ValueError) as error:
        raise RuntimeError(
            "spark-context-cache: spark_cache_max_pending_restores"
            " must be an integer"
        ) from error
    if max_pending_restores < 1:
        raise RuntimeError(
            "spark-context-cache: spark_cache_max_pending_restores"
            " must be at least 1"
        )
    return ConnectorConfig(
        tp_degree=tp_degree,
        dcp_degree=dcp_degree,
        block_size=block_size,
        profile=profile,
        storage_mode=storage_mode,
        group_topology=tuple(_freeze_config_value(group) for group in group_topology),
        chunk_tokens=chunk_tokens,
        root=root,
        capacity_policy=capacity_policy,
        min_span=min_span,
        max_span=max_span,
        store_enabled=store_enabled,
        restore_enabled=restore_enabled,
        streaming_snapshots_enabled=streaming_snapshots_enabled,
        native_restore_enabled=native_restore_enabled,
        native_library_path=native_library_path,
        native_library_sha256=native_library_sha256,
        native_arena_bytes=native_arena_bytes,
        native_io_workers=native_io_workers,
        scheduler_probe=scheduler_probe,
        identity_base=_freeze_config_value(identity_base),
        load_thread_limit=load_thread_limit,
        max_pending_restores=max_pending_restores,
    )
