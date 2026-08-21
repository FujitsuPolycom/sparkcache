"""Create or run one SparkCache-enabled DeepSeek-0731 TP4/DCP1 rank."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from deploy.deployment_contract import (
    launch_container as launch,
    read_single_inspection,
    source_tree_sha256,
    validate_overlay_receipt,
)
from deploy.deepseek_v4.tp4_profile import (
    DEFAULT_CACHE_ROOT,
    LOW_WATERMARK_BYTES,
    MAX_BYTES,
    PROFILE,
    PROFILE_ID,
    TTL_SECONDS,
    ProfileTransformError,
    transform_inspection,
)


SPARKCACHE_SOURCE_PATH = "/opt/sparkcache-src/sparkcache"
VLLM_SCHEDULER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
)
VLLM_CONFIG_PATH = "/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py"
VLLM_ENTRYPOINT = "/opt/venv/bin/vllm"
JIT_ROOT = "/cache/jit"
EXPECTED_SCHEDULER_SHA256 = str(
    PROFILE["vllm"]["scheduler_hma_postimage_sha256"]
)
EXPECTED_VLLM_CONFIG_SHA256 = str(PROFILE["vllm"]["config_postimage_sha256"])
EXPECTED_SOURCE_SHA256 = str(PROFILE["sparkcache"]["source_sha256"])


def _inspection(path: Path) -> dict:
    return read_single_inspection(path, error_type=ProfileTransformError)


def _validate_overlays(
    receipt_path: Path,
    scheduler_path: Path,
    config_path: Path,
    source_path: Path,
) -> None:
    expected = {
        "scheduler.py": EXPECTED_SCHEDULER_SHA256,
        "vllm.py": EXPECTED_VLLM_CONFIG_SHA256,
    }
    validate_overlay_receipt(
        receipt_path,
        role="DeepSeek-V4 vLLM",
        schema="sparkcache-deepseek0731-tp4-vllm-overlays/v1",
        expected_files=expected,
        file_paths={"scheduler.py": scheduler_path, "vllm.py": config_path},
        source_sha256=EXPECTED_SOURCE_SHA256,
        error_type=ProfileTransformError,
    )
    if not source_path.is_dir():
        raise ProfileTransformError("SparkCache source directory is absent")
    try:
        source_digest = source_tree_sha256(source_path)
    except RuntimeError as error:
        raise ProfileTransformError(str(error)) from error
    if source_digest != EXPECTED_SOURCE_SHA256:
        raise ProfileTransformError("SparkCache source tree differs from TP4 profile")


def _require_directory(path: Path, role: str) -> str:
    resolved = path.resolve()
    if not resolved.is_dir():
        raise ProfileTransformError(f"{role} directory is absent: {resolved}")
    return resolved.as_posix()


def _require_disjoint(left: Path, right: Path) -> None:
    left = left.resolve()
    right = right.resolve()
    if left == right or left in right.parents or right in left.parents:
        raise ProfileTransformError(
            "cache-host-path and jit-host-path must be disjoint directories"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--cache-host-path", required=True, type=Path)
    parser.add_argument("--jit-host-path", required=True, type=Path)
    parser.add_argument("--sparkcache-source-host-path", required=True, type=Path)
    parser.add_argument("--scheduler-overlay-host-path", required=True, type=Path)
    parser.add_argument("--vllm-config-overlay-host-path", required=True, type=Path)
    parser.add_argument("--vllm-overlay-receipt-host-path", required=True, type=Path)
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--master-port", type=int)
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        inspection = _inspection(args.inspect)
        if args.image != inspection.get("Image"):
            raise ProfileTransformError(
                "--image must equal the immutable image ID in --inspect"
            )
        _require_disjoint(args.cache_host_path, args.jit_host_path)
        cache_host = _require_directory(args.cache_host_path, "cache host")
        jit_host = _require_directory(args.jit_host_path, "JIT host")
        source_host = _require_directory(
            args.sparkcache_source_host_path,
            "SparkCache source host",
        )
        _validate_overlays(
            args.vllm_overlay_receipt_host_path,
            args.scheduler_overlay_host_path,
            args.vllm_config_overlay_host_path,
            args.sparkcache_source_host_path,
        )
        transformed = transform_inspection(
            inspection,
            checkpoint_sha256=args.checkpoint_sha256,
            api_port=args.api_port,
            master_port=args.master_port,
        )
    except (OSError, json.JSONDecodeError, ProfileTransformError) as error:
        parser.error(str(error))

    labels = dict(transformed["Config"].get("Labels") or {})
    labels.update(
        {
            "org.sparkcache.managed": "true",
            "org.sparkcache.profile": PROFILE_ID,
        }
    )
    launch(
        transformed,
        args.image,
        args.name,
        args.checkpoint_sha256,
        create_only=args.create_only,
        sparkcache_root=DEFAULT_CACHE_ROOT,
        max_bytes=MAX_BYTES,
        low_watermark_bytes=LOW_WATERMARK_BYTES,
        ttl_seconds=TTL_SECONDS,
        preserve_all_binds=True,
        entrypoint=VLLM_ENTRYPOINT,
        labels=labels,
        extra_binds=(
            (cache_host, DEFAULT_CACHE_ROOT, False),
            (jit_host, JIT_ROOT, False),
            (source_host, SPARKCACHE_SOURCE_PATH, True),
            (
                args.scheduler_overlay_host_path.resolve().as_posix(),
                VLLM_SCHEDULER_PATH,
                True,
            ),
            (
                args.vllm_config_overlay_host_path.resolve().as_posix(),
                VLLM_CONFIG_PATH,
                True,
            ),
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
