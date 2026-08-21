"""Install a hash-attested private runtime overlay into a derived image."""

from __future__ import annotations

import hashlib
import json
import shutil
import sys
from pathlib import Path


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


def install(context: Path) -> None:
    manifest_path = context / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sparkcache-private-overlay/v1":
        raise RuntimeError("unsupported private-overlay manifest")
    for record in manifest.get("artifacts", []):
        source = context / "artifacts" / record["artifact"]
        destination = Path(record["destination"])
        kind = record["kind"]
        actual = _tree_sha256(source) if kind == "directory" else _sha256(source)
        if actual != record["sha256"]:
            raise RuntimeError(
                f"overlay artifact hash mismatch for {record['destination']}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        if kind == "directory":
            shutil.copytree(source, destination, dirs_exist_ok=True)
        elif kind == "file":
            shutil.copy2(source, destination)
        else:
            raise RuntimeError(f"unknown overlay artifact kind {kind!r}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: install_overlay.py CONTEXT")
    install(Path(sys.argv[1]))
