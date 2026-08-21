#!/bin/bash
set -euo pipefail

: "${SPARKCACHE_TARGET_CHECKPOINT_SHA256:?set the immutable checkpoint SHA-256}"

export LMCACHE=0
export PYTHONPATH="/opt/sparkcache-src:${PYTHONPATH:-}"

profile="${SPARKCACHE_MODEL_PROFILE:-deepseek-v4-fp8-hma}"
cache_root="${SPARKCACHE_ROOT:-/cache/sparkcache-dsv4}"
probe="${SPARKCACHE_SCHEDULER_PROBE:-none}"
sparkcache_config="$(python3 - <<'PY'
import json
import os

max_bytes = int(os.environ.get("SPARKCACHE_MAX_BYTES", "0"))
low_watermark_bytes = int(os.environ.get(
    "SPARKCACHE_LOW_WATERMARK_BYTES",
    str(max_bytes * 9 // 10 if max_bytes else 0),
))

print(json.dumps({
    "kv_connector": "SparkContextCacheConnector",
    "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
    "kv_role": "kv_both",
    "kv_load_failure_policy": "recompute",
    "kv_connector_extra_config": {
        "spark_cache_root": os.environ.get(
            "SPARKCACHE_ROOT", "/cache/sparkcache-dsv4"
        ),
        "spark_cache_model_profile": os.environ.get(
            "SPARKCACHE_MODEL_PROFILE", "deepseek-v4-fp8-hma"
        ),
        "spark_cache_target_checkpoint_sha256": os.environ[
            "SPARKCACHE_TARGET_CHECKPOINT_SHA256"
        ],
        "spark_cache_draft_policy": "colocated_target",
        "spark_cache_scheduler_probe": os.environ.get(
            "SPARKCACHE_SCHEDULER_PROBE", "none"
        ),
        "spark_cache_store": True,
        "spark_cache_restore": True,
        "spark_cache_max_bytes": max_bytes,
        "spark_cache_low_watermark_bytes": low_watermark_bytes,
        "spark_cache_ttl_seconds": int(os.environ.get(
            "SPARKCACHE_TTL_SECONDS", "0"
        )),
        "spark_cache_min_span_tokens": int(os.environ.get(
            "SPARKCACHE_MIN_SPAN_TOKENS", "256"
        )),
        "spark_cache_max_span_tokens": int(os.environ.get(
            "SPARKCACHE_MAX_SPAN_TOKENS", "524288"
        )),
        "spark_cache_streaming_snapshots": False,
        "spark_cache_native_restore": False,
    },
}, separators=(",", ":")))
PY
)"

args=()
while (($#)); do
    case "$1" in
        --kv-transfer-config)
            shift
            (($#)) && shift
            ;;
        *)
            args+=("$1")
            shift
            ;;
    esac
done
args+=(--kv-transfer-config "$sparkcache_config")

echo "spark-context-cache: baked runtime profile=${profile} root=${cache_root} probe=${probe}"
exec /opt/venv/bin/vllm "${args[@]}"
