"""Launch one rank of the SparkCache-enabled GLM-5.2 serving recipe (``R7``)."""

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
from deploy.glm52_35bpw.profile import (
    DEFAULT_CACHE_ROOT,
    LOW_WATERMARK_BYTES,
    MAX_BYTES,
    TTL_SECONDS,
    ProfileTransformError,
    SPARKCACHE_SOURCE_SHA256,
    transform_inspection,
)


SPARKCACHE_SOURCE_PATH = "/opt/sparkcache-src/sparkcache"
VLLM_SCHEDULER_PATH = (
    "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py"
)
VLLM_CONFIG_PATH = "/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py"
EXPECTED_SCHEDULER_SHA256 = (
    "d4ebec211b027b6c7f64574f79374237de0f5fde0c5c03f20f1cb1596ffadc3a"
)
EXPECTED_VLLM_CONFIG_SHA256 = (
    "71c4f9e622dd8b3d665f2a2b5fb932206516ddb82873ff89283c63aa80696005"
)


def _validate_overlay_inputs(
    receipt_path: Path,
    scheduler_path: Path,
    config_path: Path,
) -> str:
    expected = {
        "scheduler.py": EXPECTED_SCHEDULER_SHA256,
        "vllm.py": EXPECTED_VLLM_CONFIG_SHA256,
    }
    return validate_overlay_receipt(
        receipt_path,
        role="GLM-5.2 vLLM",
        schema="sparkcache-glm52-r7-vllm-overlays/v1",
        expected_files=expected,
        file_paths={"scheduler.py": scheduler_path, "vllm.py": config_path},
        source_sha256=SPARKCACHE_SOURCE_SHA256,
        error_type=ProfileTransformError,
    )


def _validate_sparkcache_source(path: Path, expected_sha256: str) -> None:
    required = (
        "spark_context_cache_connector.py",
        "spark_context_cache_profiles.py",
        "spark_context_cache_store.py",
        "persistent_context_cache/cache_manifest.py",
        "streaming/factory.py",
    )
    if not path.is_dir() or any(not (path / item).is_file() for item in required):
        raise ProfileTransformError(
            "sparkcache source path lacks the required connector package"
        )
    try:
        actual = source_tree_sha256(path)
    except RuntimeError as error:
        raise ProfileTransformError(str(error)) from error
    if actual != expected_sha256:
        raise ProfileTransformError("sparkcache source tree differs from receipt")


def _inspection(path: Path) -> dict:
    return read_single_inspection(path, error_type=ProfileTransformError)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inspect", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument(
        "--checkpoint-sha256",
        required=True,
        help="immutable checkpoint or verified artifact-manifest identity",
    )
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument(
        "--cache-host-path",
        required=True,
        help="rank-local host directory bound read/write at --cache-root",
    )
    parser.add_argument(
        "--sparkcache-source-host-path",
        required=True,
        help="host SparkCache package directory bound read-only",
    )
    parser.add_argument(
        "--scheduler-overlay-host-path",
        required=True,
        help="host scheduler.py carrying SparkCache patch 011",
    )
    parser.add_argument(
        "--vllm-config-overlay-host-path",
        required=True,
        help="host vllm.py carrying SparkCache patch 020",
    )
    parser.add_argument(
        "--vllm-overlay-receipt-host-path",
        required=True,
        help="receipt.json emitted beside the exact vLLM overlay files",
    )
    parser.add_argument("--api-port", type=int)
    parser.add_argument("--master-port", type=int)
    parser.add_argument(
        "--streaming-snapshots",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--streaming-native-library")
    parser.add_argument("--streaming-native-library-sha256")
    parser.add_argument("--streaming-timing", action="store_true")
    parser.add_argument(
        "--cuda-restore",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument("--cuda-placement-library")
    parser.add_argument("--cuda-placement-library-sha256")
    parser.add_argument(
        "--cuda-placement-arena-bytes",
        type=int,
        default=None,
    )
    parser.add_argument("--cuda-restore-io-workers", type=int)
    parser.add_argument(
        "--native-restore",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--native-restore-library", help=argparse.SUPPRESS)
    parser.add_argument("--native-restore-library-sha256", help=argparse.SUPPRESS)
    parser.add_argument(
        "--native-restore-arena-bytes",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--native-restore-io-workers",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        source_inspection = _inspection(args.inspect)
    except (OSError, json.JSONDecodeError, ProfileTransformError) as error:
        parser.error(str(error))
    if args.image != source_inspection.get("Image"):
        parser.error("--image must equal the immutable image ID in --inspect")
    try:
        source_digest = _validate_overlay_inputs(
            Path(args.vllm_overlay_receipt_host_path),
            Path(args.scheduler_overlay_host_path),
            Path(args.vllm_config_overlay_host_path),
        )
        _validate_sparkcache_source(
            Path(args.sparkcache_source_host_path),
            source_digest,
        )
        transformed = transform_inspection(
            source_inspection,
            checkpoint_sha256=args.checkpoint_sha256,
            cache_root=args.cache_root,
            streaming_snapshots=args.streaming_snapshots,
            streaming_native_library=args.streaming_native_library,
            streaming_native_library_sha256=(args.streaming_native_library_sha256),
            streaming_timing=args.streaming_timing,
            cuda_restore=args.cuda_restore,
            cuda_placement_library=args.cuda_placement_library,
            cuda_placement_library_sha256=args.cuda_placement_library_sha256,
            cuda_placement_arena_bytes=args.cuda_placement_arena_bytes,
            cuda_restore_io_workers=args.cuda_restore_io_workers,
            native_restore=args.native_restore,
            native_restore_library=args.native_restore_library,
            native_restore_library_sha256=args.native_restore_library_sha256,
            native_restore_arena_bytes=args.native_restore_arena_bytes,
            native_restore_io_workers=args.native_restore_io_workers,
            api_port=args.api_port,
            master_port=args.master_port,
        )
    except (OSError, json.JSONDecodeError, ProfileTransformError) as error:
        parser.error(str(error))

    launch(
        transformed,
        args.image,
        args.name,
        args.checkpoint_sha256,
        create_only=args.create_only,
        sparkcache_root=args.cache_root,
        max_bytes=MAX_BYTES,
        low_watermark_bytes=LOW_WATERMARK_BYTES,
        ttl_seconds=TTL_SECONDS,
        extra_binds=(
            (args.cache_host_path, args.cache_root, False),
            (
                args.sparkcache_source_host_path,
                SPARKCACHE_SOURCE_PATH,
                True,
            ),
            (
                args.scheduler_overlay_host_path,
                VLLM_SCHEDULER_PATH,
                True,
            ),
            (
                args.vllm_config_overlay_host_path,
                VLLM_CONFIG_PATH,
                True,
            ),
        ),
        preserve_all_binds=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
