"""Apply one exact-source vLLM runtime patch for a SparkCache image."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


class RuntimePatchError(RuntimeError):
    """The installed vLLM tree cannot be proven compatible with a patch."""


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def apply_runtime_patch(*, root: Path, manifest_path: Path) -> None:
    """Apply a patch only when every source file has its exact preimage."""

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "sparkcache-vllm-runtime-patch/v1":
        raise RuntimePatchError("runtime patch manifest schema is unsupported")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise RuntimePatchError("runtime patch manifest has no files")

    observed = {name: _sha256(root / name) for name in files}
    preimages = {name: record["preimage_sha256"] for name, record in files.items()}
    postimages = {name: record["postimage_sha256"] for name, record in files.items()}
    if observed == postimages:
        return
    if observed != preimages:
        mismatches = sorted(name for name in files if observed[name] != preimages[name])
        raise RuntimePatchError(
            "installed vLLM source differs from the complete accepted preimage: "
            + ", ".join(mismatches)
        )

    patch_path = manifest_path.with_name(str(manifest["patch"]))
    if _sha256(patch_path) != manifest.get("patch_sha256"):
        raise RuntimePatchError("runtime patch digest differs from the manifest")
    subprocess.run(
        ["patch", "--batch", "--forward", "--fuzz=0", "-p1", "-i", patch_path],
        cwd=root,
        check=True,
    )
    after = {name: _sha256(root / name) for name in files}
    if after != postimages:
        mismatches = sorted(name for name in files if after[name] != postimages[name])
        raise RuntimePatchError(
            "patched vLLM source differs from the complete accepted postimage: "
            + ", ".join(mismatches)
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    apply_runtime_patch(root=args.root, manifest_path=args.manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
