"""Launch a baked image from one transformed Docker inspection record."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any, Mapping, Sequence

from deploy.deployment_contract import (
    launch_container,
    normalized_posix_path as _bind_path,
    vllm_arguments as _vllm_args,
)

__all__ = ("_bind_path", "_vllm_args", "launch", "main")


def launch(
    inspection: dict[str, Any],
    image: str,
    name: str,
    checkpoint_sha256: str,
    create_only: bool = False,
    sparkcache_root: str | None = None,
    max_bytes: int | None = None,
    low_watermark_bytes: int | None = None,
    ttl_seconds: int | None = None,
    extra_binds: Sequence[tuple[str, str] | tuple[str, str, bool]] = (),
    preserve_all_binds: bool = False,
    entrypoint: str | None = None,
    labels: Mapping[str, str] | None = None,
) -> None:
    """Compatibility interface for the model-neutral container launcher."""

    launch_container(
        inspection,
        image,
        name,
        checkpoint_sha256,
        create_only=create_only,
        sparkcache_root=sparkcache_root,
        max_bytes=max_bytes,
        low_watermark_bytes=low_watermark_bytes,
        ttl_seconds=ttl_seconds,
        extra_binds=extra_binds,
        preserve_all_binds=preserve_all_binds,
        entrypoint=entrypoint,
        labels=labels,
        runner=subprocess.run,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inspect", required=True, type=Path)
    parser.add_argument("--image", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--sparkcache-root")
    parser.add_argument("--max-bytes", type=int)
    parser.add_argument("--low-watermark-bytes", type=int)
    parser.add_argument("--ttl-seconds", type=int)
    parser.add_argument("--create-only", action="store_true")
    args = parser.parse_args()
    records = json.loads(args.inspect.read_text(encoding="utf-8"))
    if len(records) != 1:
        raise RuntimeError("inspect file must contain one container")
    launch(
        records[0],
        args.image,
        args.name,
        args.checkpoint_sha256,
        create_only=args.create_only,
        sparkcache_root=args.sparkcache_root,
        max_bytes=args.max_bytes,
        low_watermark_bytes=args.low_watermark_bytes,
        ttl_seconds=args.ttl_seconds,
    )


if __name__ == "__main__":
    main()
