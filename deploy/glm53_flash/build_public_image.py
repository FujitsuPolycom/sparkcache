#!/usr/bin/env python3
"""Build and attest the SparkCache GLM-5.3 overlay from a registry parent."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
from pathlib import Path
from typing import Any, Iterable

from deploy.deployment_contract.source import source_tree_sha256
from deploy.glm53_flash.build_image import build_command


RECEIPT_SCHEMA = "sparkcache-glm53-public-image/v1"
IMMUTABLE_IMAGE = re.compile(r"[^\s@]+@sha256:[0-9a-f]{64}\Z")
EXPECTED_LICENSES = (
    "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0 AND BSD-3-Clause"
)


class PublicBuildError(RuntimeError):
    """Raised when the public image cannot prove an immutable source boundary."""


def run(argv: Iterable[str], *, cwd: Path | None = None) -> str:
    arguments = list(argv)
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode:
        detail = completed.stderr.strip() or completed.stdout.strip()
        raise PublicBuildError(f"command failed ({' '.join(arguments)}): {detail}")
    return completed.stdout.strip()


def inspect_image(image: str) -> dict[str, Any]:
    documents = json.loads(run(("docker", "image", "inspect", image)))
    if not isinstance(documents, list) or len(documents) != 1:
        raise PublicBuildError("Docker returned an unexpected image inspection")
    return documents[0]


def require_clean_sources(repository: Path) -> str:
    revision = run(("git", "-C", str(repository), "rev-parse", "HEAD"))
    paths = ("sparkcache", "deploy/glm53_flash", "patches/vllm-da4d7be", "LICENSE")
    if run(("git", "-C", str(repository), "status", "--porcelain", "--", *paths)):
        raise PublicBuildError("SparkCache image inputs differ from the checked-out revision")
    return revision


def build_public_image(
    *,
    repository: Path,
    base_image: str,
    output_image: str,
) -> dict[str, Any]:
    if IMMUTABLE_IMAGE.fullmatch(base_image) is None:
        raise PublicBuildError("base image must be a registry reference ending in @sha256:<64 hex>")
    run(("docker", "pull", "--platform", "linux/arm64", base_image))
    parent = inspect_image(base_image)
    if parent.get("Architecture") != "arm64" or parent.get("Os") != "linux":
        raise PublicBuildError("parent image must use linux/arm64")
    parent_labels = parent.get("Config", {}).get("Labels") or {}
    parent_licenses = parent_labels.get("org.opencontainers.image.licenses")
    if parent_licenses != EXPECTED_LICENSES:
        raise PublicBuildError(
            "parent license label drift: expected "
            f"{EXPECTED_LICENSES!r}, got {parent_licenses!r}"
        )
    revision = require_clean_sources(repository)
    source_sha256 = source_tree_sha256(repository / "sparkcache")
    command = build_command(
        repository=repository,
        base_image=base_image,
        base_image_id=parent["Id"],
        source_sha256=source_sha256,
        sparkcache_revision=revision,
        base_image_licenses=parent_licenses,
        output_image=output_image,
    )
    run(command, cwd=repository)
    output = inspect_image(output_image)
    labels = output.get("Config", {}).get("Labels") or {}
    expected_labels = {
        "org.opencontainers.image.base.name": base_image,
        "org.opencontainers.image.licenses": parent_licenses,
        "org.opencontainers.image.revision": revision,
        "org.sparkcache.deployment-profile": "glm53-flash-hybrid",
        "org.sparkcache.parent-image-id": parent["Id"],
        "org.sparkcache.source-revision": revision,
        "org.sparkcache.source-sha256": source_sha256,
    }
    for name, expected in expected_labels.items():
        if labels.get(name) != expected:
            raise PublicBuildError(
                f"output label {name} drift: expected {expected!r}, got {labels.get(name)!r}"
            )
    return {
        "schema": RECEIPT_SCHEMA,
        "status": "implemented",
        "parent": {
            "reference": base_image,
            "image_id": parent["Id"],
            "repo_digests": sorted(parent.get("RepoDigests") or []),
        },
        "sparkcache": {
            "revision": revision,
            "source_sha256": source_sha256,
        },
        "image": {
            "reference": output_image,
            "image_id": output["Id"],
            "repo_digests": sorted(output.get("RepoDigests") or []),
            "size_bytes": output.get("Size"),
            "labels": dict(sorted(labels.items())),
        },
        "limitation": (
            "This receipt verifies construction. Four-rank TP4/DCP1 serving is "
            "unqualified until a live receipt names one registry digest."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--base-image", required=True)
    parser.add_argument("--output-image", required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    try:
        receipt = build_public_image(
            repository=args.repository.resolve(),
            base_image=args.base_image,
            output_image=args.output_image,
        )
    except (OSError, KeyError, json.JSONDecodeError, PublicBuildError) as exc:
        parser.error(str(exc))
    rendered = json.dumps(receipt, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
