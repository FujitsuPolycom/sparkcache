"""Prepare a private derived-image context from a validated live container.

The generated context is ignored by Git because it can contain maintained
runtime overlays whose redistribution status differs from SparkCache's.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
from pathlib import Path
from typing import Any

_DATA_DESTINATIONS = (
    "/cache",
    "/l2cache",
    "/models",
    "/root/.cache/huggingface",
    "/wheels",
)
_SKIP_DESTINATIONS = ("/leg3pair-inner.sh",)


def _run(*args: str) -> str:
    return subprocess.check_output(args, text=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha256(path)))
    return digest.hexdigest()


def _normalize_timestamps(root: Path) -> None:
    for path in [root, *sorted(root.rglob("*"))]:
        os.utime(path, ns=(0, 0), follow_symlinks=False)


def _write_rootfs_tar(output: Path, repo_root: Path) -> None:
    rootfs = output / "rootfs"
    shutil.copytree(output / "sparkcache", rootfs / "opt/sparkcache-src/sparkcache")
    shutil.copytree(output / "overlay", rootfs / "opt/sparkcache-overlay")
    binary = rootfs / "opt/sparkcache/bin"
    binary.mkdir(parents=True)
    shutil.copy2(
        repo_root / "deploy/deepseek_v4/install_overlay.py",
        binary / "install_overlay.py",
    )
    shutil.copy2(
        repo_root / "deploy/deepseek_v4/entrypoint.sh",
        binary / "entrypoint.sh",
    )

    def normalize(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid = info.gid = 0
        info.uname = info.gname = "root"
        info.mtime = 0
        return info

    with tarfile.open(output / "rootfs.tar", "w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(rootfs.iterdir()):
            tar.add(path, arcname=path.name, recursive=True, filter=normalize)
    shutil.rmtree(rootfs)


def prepare(container: str, repo_root: Path, output: Path) -> None:
    if output.exists():
        raise RuntimeError(f"refusing to overwrite context: {output}")
    inspection: list[dict[str, Any]] = json.loads(_run("docker", "inspect", container))
    if len(inspection) != 1:
        raise RuntimeError("docker inspect returned an unexpected record count")
    record = inspection[0]
    output.mkdir(parents=True)
    shutil.copytree(
        repo_root / "sparkcache",
        output / "sparkcache",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "build"),
    )
    _normalize_timestamps(output / "sparkcache")
    overlay = output / "overlay"
    artifacts = overlay / "artifacts"
    artifacts.mkdir(parents=True)
    manifest_records = []
    mounts = sorted(
        record.get("Mounts", []), key=lambda mount: str(mount.get("Destination", ""))
    )
    for mount in mounts:
        destination = str(mount.get("Destination", ""))
        if destination in _SKIP_DESTINATIONS or destination.startswith(
            _DATA_DESTINATIONS
        ):
            continue
        source = Path(str(mount.get("Source", "")))
        if not source.exists():
            raise RuntimeError(f"overlay source is absent: {source}")
        artifact = f"{len(manifest_records):03d}"
        staged = artifacts / artifact
        if source.is_dir():
            shutil.copytree(source, staged)
            kind = "directory"
            digest = _tree_sha256(staged)
        else:
            shutil.copy2(source, staged)
            kind = "file"
            digest = _sha256(staged)
        manifest_records.append(
            {
                "artifact": artifact,
                "destination": destination,
                "kind": kind,
                "sha256": digest,
            }
        )
    (overlay / "manifest.json").write_text(
        json.dumps(
            {"schema": "sparkcache-private-overlay/v1", "artifacts": manifest_records},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    image = record["Image"]
    (output / "build-metadata.json").write_text(
        json.dumps(
            {
                "schema": "sparkcache-derived-image/v1",
                "source_container": container,
                "source_image_id": image,
                "artifact_count": len(manifest_records),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    _write_rootfs_tar(output, repo_root)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--container", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    prepare(args.container, args.repo_root.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
