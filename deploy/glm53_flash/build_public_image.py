#!/usr/bin/env python3
"""Build and attest the SparkCache GLM-5.3 overlay from a registry parent."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from deploy.deployment_contract.source import source_tree_sha256  # noqa: E402
from deploy.glm53_flash.build_image import build_command  # noqa: E402


RECEIPT_SCHEMA = "sparkcache-glm53-public-image/v1"
IMMUTABLE_IMAGE = re.compile(
    r"ghcr\.io/fujitsupolycom/sparkring-glm53-runtime@sha256:[0-9a-f]{64}\Z"
)
GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")
EXPECTED_LICENSES = (
    "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0 AND BSD-3-Clause"
)
EXPECTED_PARENT_LABELS = {
    "org.opencontainers.image.source": "https://github.com/FujitsuPolycom/sparkring",
    "org.opencontainers.image.licenses": EXPECTED_LICENSES,
    "org.jovian.architecture": "linux-arm64-sm121",
    "org.jovian.vllm.commit": "da4d7be6c97434f6942292ed8abbf4b32dc44355",
    "org.jovian.b12x.commit": "2fcf23a0ce269be27b2e03fece73d46e90e6aeea",
    "org.jovian.transport": "sparkring-nccl-2.30.7-source-built",
    "org.sparkring.nccl.commit": "73cf112295c33aee2b895f329f592f2a9b4b0f97",
    "org.sparkring.nccl.patched-tree": "abdeb053b94c3f6d472cd55ae2b79ca821299009",
    "org.sparkring.nccl.patch-sha256": (
        "6709063fa1c25055ae77a9397dea5d89643f8211d25e7990bdd11597d08c0dde"
    ),
}
IMAGE_INPUT_PATHS = (
    "sparkcache",
    "deploy/glm53_flash",
    "deploy/deployment_contract/source.py",
    "patches/vllm-da4d7be",
    "LICENSE",
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
    if run(
        (
            "git",
            "-C",
            str(repository),
            "status",
            "--porcelain",
            "--",
            *IMAGE_INPUT_PATHS,
        )
    ):
        raise PublicBuildError("SparkCache image inputs differ from the checked-out revision")
    return revision


def validate_parent(parent: dict[str, Any]) -> dict[str, str]:
    """Return the exact labels after proving the SparkRing runtime contract."""

    if parent.get("Architecture") != "arm64" or parent.get("Os") != "linux":
        raise PublicBuildError("parent image must use linux/arm64")
    labels = parent.get("Config", {}).get("Labels") or {}
    if not isinstance(labels, dict):
        raise PublicBuildError("parent image labels must be a JSON object")
    for name, expected in EXPECTED_PARENT_LABELS.items():
        observed = labels.get(name)
        if observed != expected:
            raise PublicBuildError(
                f"parent label {name} drift: expected {expected!r}, got {observed!r}"
            )
    patterned = {
        "org.opencontainers.image.revision": GIT_COMMIT,
        "org.sparkring.source-receipt-sha256": SHA256,
    }
    for name, pattern in patterned.items():
        observed = labels.get(name)
        if not isinstance(observed, str) or pattern.fullmatch(observed) is None:
            raise PublicBuildError(f"parent label {name} is not an immutable identity")
    return {name: str(labels[name]) for name in (*EXPECTED_PARENT_LABELS, *patterned)}


def build_public_image(
    *,
    repository: Path,
    base_image: str,
    output_image: str,
) -> dict[str, Any]:
    if IMMUTABLE_IMAGE.fullmatch(base_image) is None:
        raise PublicBuildError(
            "base image must be an immutable FujitsuPolycom SparkRing GLM-5.3"
            " runtime reference"
        )
    run(("docker", "pull", "--platform", "linux/arm64", base_image))
    parent = inspect_image(base_image)
    parent_labels = validate_parent(parent)
    parent_licenses = parent_labels["org.opencontainers.image.licenses"]
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
            "labels": parent_labels,
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
