"""Verify SparkCache release archives and an isolated wheel installation."""

from __future__ import annotations

import argparse
from email.parser import Parser
import json
import os
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import tarfile
import tempfile
import venv
import zipfile


REQUIRED_WHEEL_MEMBERS = frozenset(
    {
        "sparkcache/__init__.py",
        "sparkcache/persistent_context_cache/cache_manifest.py",
        "sparkcache/spark_context_cache_codec.py",
        "sparkcache/spark_context_cache_config.py",
        "sparkcache/spark_context_cache_connector.py",
        "sparkcache/spark_context_cache_hybrid.py",
        "sparkcache/spark_context_cache_profiles.py",
        "sparkcache/spark_context_cache_store.py",
        "sparkcache/streaming/__init__.py",
        "sparkcache/streaming/feature_gate.py",
        "sparkcache/runtime_patches/vllm-kv-block-lease-contract.json",
        "sparkcache/runtime_patches/vllm-kv-block-lease-contract-e2666d9a6.json",
    }
)

LIFECYCLE_SNAPSHOT_LABEL = re.compile(
    r"(?:^|[_./-])(prototype|pilot|phase|current|new|old|latest)(?:[_./-]|$)",
    re.IGNORECASE,
)
CONDITIONAL_PUBLICATION_CLAIMS = (
    re.compile(
        r"\bpublication (?:has|have) not (?:yet )?been "
        r"(?:performed|completed)\b",
        re.IGNORECASE,
    ),
    re.compile(
        r"\b(?:after|when|once)\b[^.\n]{0,120}"
        r"\b(?:is|are|has been|have been) published\b",
        re.IGNORECASE,
    ),
)


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_archive_contract(
    members: set[str],
    metadata: str,
    expected_version: str,
    artifact: Path,
) -> None:
    lifecycle_labeled = sorted(
        member
        for member in members
        if "snapshot" in member.casefold()
        and LIFECYCLE_SNAPSHOT_LABEL.search(member)
    )
    if lifecycle_labeled:
        raise RuntimeError(
            f"{artifact.name} contains lifecycle-labeled snapshot paths: "
            + ", ".join(lifecycle_labeled)
        )

    parsed = Parser().parsestr(metadata)
    actual_version = parsed.get("Version")
    if actual_version != expected_version:
        raise RuntimeError(
            f"{artifact.name} metadata version {actual_version!r} does not "
            f"match {expected_version!r}"
        )
    for pattern in CONDITIONAL_PUBLICATION_CLAIMS:
        match = pattern.search(metadata)
        if match is not None:
            raise RuntimeError(
                f"{artifact.name} metadata contains a conditional publication "
                f"claim: {match.group(0)!r}"
            )
    released_version_claim = re.compile(
        rf"\bversion\s+[`'\"]?{re.escape(expected_version)}[`'\"]?\s+"
        rf"(?:is|has been)\s+(?:now\s+)?published\b",
        re.IGNORECASE,
    ).search(metadata)
    if released_version_claim is not None:
        raise RuntimeError(
            f"{artifact.name} metadata claims its own version is published: "
            f"{released_version_claim.group(0)!r}"
        )


def verify_wheel(wheel: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
        metadata_members = sorted(
            member for member in members if member.endswith(".dist-info/METADATA")
        )
        if len(metadata_members) != 1:
            raise RuntimeError(
                f"wheel must contain exactly one METADATA file, found "
                f"{len(metadata_members)}"
            )
        metadata = archive.read(metadata_members[0]).decode("utf-8")
    verify_archive_contract(members, metadata, expected_version, wheel)
    missing = sorted(REQUIRED_WHEEL_MEMBERS - members)
    if missing:
        raise RuntimeError(f"wheel omits required files: {', '.join(missing)}")
    repository_only = sorted(
        member
        for member in members
        if member.endswith(".py")
        and (
            Path(member).name.startswith("test")
            or member.startswith("sparkcache/native/app/")
            or member.startswith("sparkcache/native/experiments/")
            or member.startswith("sparkcache/native/tests/")
        )
    )
    if repository_only:
        raise RuntimeError(
            "wheel contains repository-only modules: " + ", ".join(repository_only)
        )

    with tempfile.TemporaryDirectory(prefix="sparkcache-wheel-") as directory:
        root = Path(directory)
        venv.create(root / "venv", with_pip=True)
        python = _venv_python(root / "venv")
        subprocess.run(
            [
                str(python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "--no-deps",
                str(wheel.resolve()),
            ],
            cwd=root,
            check=True,
        )
        probe = """
import importlib.resources
import importlib.util
import json
import sparkcache
from sparkcache.spark_context_cache_config import ConnectorConfig
from sparkcache.spark_context_cache_store import ManifestStore

contract = importlib.resources.files("sparkcache.runtime_patches").joinpath(
    "vllm-kv-block-lease-contract.json"
)
result = {
    "version": sparkcache.__version__,
    "config": ConnectorConfig.__name__,
    "store": ManifestStore.__name__,
    "connector_present": importlib.util.find_spec(
        "sparkcache.spark_context_cache_connector"
    ) is not None,
    "contract_present": contract.is_file(),
}
print(json.dumps(result, sort_keys=True))
"""
        completed = subprocess.run(
            [str(python), "-I", "-c", probe],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
    result = json.loads(completed.stdout)
    expected = {
        "connector_present": True,
        "config": "ConnectorConfig",
        "contract_present": True,
        "store": "ManifestStore",
        "version": expected_version,
    }
    if result != expected:
        raise RuntimeError(f"isolated install mismatch: expected {expected}, got {result}")


def verify_sdist(sdist: Path, expected_version: str) -> None:
    with tarfile.open(sdist, mode="r:*") as archive:
        regular_members = [member for member in archive.getmembers() if member.isfile()]
        members = {member.name for member in regular_members}
        metadata_members = [
            member
            for member in regular_members
            if PurePosixPath(member.name).name == "PKG-INFO"
            and len(PurePosixPath(member.name).parts) == 2
        ]
        if len(metadata_members) != 1:
            raise RuntimeError(
                f"source distribution must contain exactly one top-level "
                f"PKG-INFO file, found {len(metadata_members)}"
            )
        extracted = archive.extractfile(metadata_members[0])
        if extracted is None:
            raise RuntimeError("source distribution PKG-INFO is not readable")
        metadata = extracted.read().decode("utf-8")
    verify_archive_contract(members, metadata, expected_version, sdist)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify SparkCache wheel and source-distribution contracts."
    )
    parser.add_argument("distributions", nargs="+", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    for distribution in args.distributions:
        if distribution.name.endswith(".whl"):
            verify_wheel(distribution, args.version)
        elif distribution.name.endswith(".tar.gz"):
            verify_sdist(distribution, args.version)
        else:
            raise RuntimeError(
                f"unsupported distribution type: {distribution.name}"
            )
        print(f"verified SparkCache distribution: {distribution.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
