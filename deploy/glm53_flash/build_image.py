#!/usr/bin/env python3
"""Build the GLM-5.3 image after resolving its immutable parent identity."""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path


SHA256_ID = re.compile(r"sha256:[0-9a-f]{64}\Z")
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")


def build_command(
    *,
    repository: Path,
    base_image: str,
    base_image_id: str,
    source_sha256: str,
    sparkcache_revision: str,
    base_image_licenses: str,
    output_image: str,
    containerfile: str = "deploy/glm53_flash/Containerfile",
) -> list[str]:
    if SHA256_ID.fullmatch(base_image_id) is None:
        raise ValueError("base_image_id must be sha256 followed by 64 lowercase hex")
    if re.fullmatch(r"[0-9a-f]{64}", source_sha256) is None:
        raise ValueError("source_sha256 must be 64 lowercase hex characters")
    if GIT_COMMIT.fullmatch(sparkcache_revision) is None:
        raise ValueError("sparkcache_revision must be a 40-character Git commit")
    if not base_image_licenses.strip():
        raise ValueError("base_image_licenses must identify the parent image terms")
    recipe = (repository / containerfile).resolve()
    try:
        recipe.relative_to(repository.resolve())
    except ValueError as exc:
        raise ValueError("containerfile must remain inside the repository") from exc
    if repository.exists() and not recipe.is_file():
        raise ValueError(f"containerfile is absent: {containerfile}")
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
        str(recipe),
        "--build-arg",
        f"BASE_IMAGE={base_image}",
        "--build-arg",
        f"BASE_IMAGE_ID={base_image_id}",
        "--build-arg",
        f"SPARKCACHE_SOURCE_SHA256={source_sha256}",
        "--build-arg",
        f"SPARKCACHE_REVISION={sparkcache_revision}",
        "--build-arg",
        f"BASE_IMAGE_LICENSES={base_image_licenses}",
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
    parser.add_argument("--sparkcache-revision")
    parser.add_argument(
        "--base-image-licenses",
        default=(
            "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0 "
            "AND BSD-3-Clause"
        ),
    )
    parser.add_argument("--output-image", required=True)
    parser.add_argument(
        "--containerfile",
        default="deploy/glm53_flash/Containerfile",
        help="repository-relative SparkCache overlay recipe",
    )
    args = parser.parse_args()
    repository = args.repository.resolve()
    sparkcache_revision = args.sparkcache_revision
    if sparkcache_revision is None:
        sparkcache_revision = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    command = build_command(
        repository=repository,
        base_image=args.base_image,
        base_image_id=args.base_image_id,
        source_sha256=args.source_sha256,
        sparkcache_revision=sparkcache_revision,
        base_image_licenses=args.base_image_licenses,
        output_image=args.output_image,
        containerfile=args.containerfile,
    )
    subprocess.run(command, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
