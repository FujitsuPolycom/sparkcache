#!/usr/bin/env python3
"""Build the GLM-5.3 image after resolving its immutable parent identity."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")


def build_command(
    *,
    repository: Path,
    base_image: str,
    base_image_id: str,
    source_sha256: str,
    output_image: str,
) -> list[str]:
    if SHA256_ID.fullmatch(base_image_id) is None:
        raise ValueError("base_image_id must be sha256 followed by 64 lowercase hex")
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("source_sha256 must be 64 lowercase hex characters")
    inspected = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", base_image],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if inspected != base_image_id:
        raise RuntimeError(
            f"base image identity mismatch: expected {base_image_id}, got {inspected}"
        )
    return [
        "docker",
        "build",
        "-f",
        str(repository / "deploy/glm53_flash/Containerfile"),
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"BASE_IMAGE_ID={base_image_id}",
        "--build-arg",
        f"SPARKCACHE_SOURCE_SHA256={source_sha256}",
        "-t",
        output_image,
        str(repository),
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--base-image-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument("--output-image", required=True)
    args = parser.parse_args()
    command = build_command(
        repository=args.repository.resolve(),
        base_image=args.base_image,
        base_image_id=args.base_image_id,
        source_sha256=args.source_sha256,
        output_image=args.output_image,
    )
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
