"""Construct the GLM-5.3 Flash SparkCache connector configuration."""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any


SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
DEFAULT_MAX_BYTES = 200 * 1024**3
DEFAULT_LOW_WATERMARK_BYTES = 180 * 1024**3
# GLM-5.3 resolves each hybrid attention/Mamba page to 2,304 tokens. Retaining
# every eighth boundary preserves useful conversation checkpoints while
# avoiding dense recurrent-state retention. Zero keeps only the newest replay
# boundary and makes an externally restored prefix undiscoverable after its
# leader computes a tail under DFlash/EAGLE.
PREFIX_CACHE_RETENTION_INTERVAL = 8 * 2304


class ProfileError(ValueError):
    """The GLM-5.3 deployment configuration is incomplete or unsafe."""


def required_hybrid_prefix_arguments() -> tuple[str, ...]:
    """vLLM arguments required for reusable GLM hybrid prefix boundaries."""

    return (
        "--mamba-cache-mode",
        "align",
        "--prefix-cache-retention-interval",
        str(PREFIX_CACHE_RETENTION_INTERVAL),
    )


def immutable_revision_identity(repository: str, revision: str) -> str:
    """Hash one content-addressed model repository revision for CacheIdentity."""

    if not repository or not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ProfileError("model identity requires a repository and 40-hex revision")
    value = f"hf-revision-v1\0{repository}\0{revision}".encode("utf-8")
    return hashlib.sha256(value).hexdigest()


def _sha256(value: str, label: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise ProfileError(f"{label} must be a 64-character lowercase SHA-256")
    return value


def build_kv_transfer_config(
    *,
    target_checkpoint_sha256: str,
    draft_checkpoint_sha256: str,
    cache_root: str = "/cache/jit/sparkcache-context",
    min_span_tokens: int = 4096,
    max_span_tokens: int = 524288,
    max_bytes: int = DEFAULT_MAX_BYTES,
    low_watermark_bytes: int = DEFAULT_LOW_WATERMARK_BYTES,
) -> dict[str, Any]:
    """Return the complete vLLM KV-Connector-V1 configuration.

    The draft identity may describe an embedded MTP checkpoint or an external
    DFlash checkpoint. Draft cache state is recomputed after restore, but the
    identity prevents target-prefix entries from crossing draft-model changes.
    """

    target = _sha256(target_checkpoint_sha256, "target checkpoint identity")
    draft = _sha256(draft_checkpoint_sha256, "draft checkpoint identity")
    if not cache_root.startswith("/"):
        raise ProfileError("cache_root must be an absolute container path")
    if min_span_tokens < 256 or min_span_tokens % 256:
        raise ProfileError("min_span_tokens must be a positive multiple of 256")
    if max_span_tokens < min_span_tokens or max_span_tokens % 256:
        raise ProfileError(
            "max_span_tokens must be a multiple of 256 not below min_span_tokens"
        )
    if max_bytes <= 0 or not 0 <= low_watermark_bytes < max_bytes:
        raise ProfileError("capacity requires 0 <= low watermark < maximum bytes")
    return {
        "kv_connector": "SparkContextCacheConnector",
        "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
        "kv_role": "kv_both",
        "kv_load_failure_policy": "recompute",
        "kv_connector_extra_config": {
            "spark_cache_root": cache_root,
            "spark_cache_model_profile": "glm53-flash-hybrid",
            "spark_cache_target_checkpoint_sha256": target,
            "spark_cache_draft_checkpoint_sha256": draft,
            "spark_cache_draft_policy": "separate",
            "spark_cache_store": True,
            "spark_cache_restore": True,
            "spark_cache_scheduler_probe": "none",
            "spark_cache_streaming_snapshots": False,
            "spark_cache_native_restore": False,
            "spark_cache_max_bytes": max_bytes,
            "spark_cache_low_watermark_bytes": low_watermark_bytes,
            "spark_cache_ttl_seconds": 0,
            "spark_cache_min_span_tokens": min_span_tokens,
            "spark_cache_max_span_tokens": max_span_tokens,
        },
    }


def compact_json(config: dict[str, Any]) -> str:
    """Encode a connector configuration as one deterministic CLI argument."""

    return json.dumps(config, sort_keys=True, separators=(",", ":"))
