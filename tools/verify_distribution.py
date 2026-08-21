"""Verify SparkCache wheel contents and an isolated base-package install."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
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


def _venv_python(root: Path) -> Path:
    return root / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def verify_wheel(wheel: Path, expected_version: str) -> None:
    with zipfile.ZipFile(wheel) as archive:
        members = set(archive.namelist())
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify required wheel files and an isolated SparkCache install."
    )
    parser.add_argument("wheel", type=Path)
    parser.add_argument("--version", required=True)
    args = parser.parse_args()
    verify_wheel(args.wheel, args.version)
    print(f"verified SparkCache distribution: {args.wheel.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
