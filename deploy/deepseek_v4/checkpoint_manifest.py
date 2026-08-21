"""Build or verify a complete DeepSeek checkpoint content manifest."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
from typing import Any


SCHEMA = "sparkcache-checkpoint-manifest/v1"
_IGNORED_TOP_LEVEL = frozenset({".cache", ".git"})


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inventory(root: Path) -> list[Path]:
    root = root.resolve()
    if not root.is_dir():
        raise RuntimeError(f"checkpoint root is absent: {root}")
    files = []
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if relative.parts and relative.parts[0] in _IGNORED_TOP_LEVEL:
            continue
        if path.is_symlink():
            raise RuntimeError(
                f"checkpoint must be self-contained; symlink found: {relative}"
            )
        if path.is_file():
            files.append(path)
        elif path.exists() and not path.is_dir():
            raise RuntimeError(f"checkpoint contains a special file: {relative}")
    return sorted(files, key=lambda path: path.relative_to(root).as_posix())


def build_manifest(
    root: Path,
    *,
    repository: str,
    revision: str,
    workers: int = 4,
) -> dict[str, Any]:
    root = root.resolve()
    if not repository or not revision:
        raise RuntimeError("repository and revision must be nonempty")
    if not 1 <= workers <= 32:
        raise RuntimeError("workers must be in [1, 32]")
    paths = _inventory(root)
    if not paths:
        raise RuntimeError("checkpoint contains no files")
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        digests = list(pool.map(_sha256, paths))
    files = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest,
        }
        for path, digest in zip(paths, digests)
    ]
    payload = {
        "schema": SCHEMA,
        "repository": repository,
        "revision": revision,
        "files": files,
    }
    return {
        **payload,
        "file_count": len(files),
        "total_bytes": sum(record["bytes"] for record in files),
        "checkpoint_sha256": hashlib.sha256(_canonical(payload)).hexdigest(),
    }


def verify_manifest(
    root: Path,
    manifest: dict[str, Any],
    *,
    workers: int = 4,
) -> dict[str, Any]:
    if manifest.get("schema") != SCHEMA:
        raise RuntimeError("checkpoint manifest schema differs")
    expected = build_manifest(
        root,
        repository=str(manifest.get("repository", "")),
        revision=str(manifest.get("revision", "")),
        workers=workers,
    )
    for key in (
        "files",
        "file_count",
        "total_bytes",
        "checkpoint_sha256",
    ):
        if manifest.get(key) != expected[key]:
            raise RuntimeError(f"checkpoint manifest differs at {key}")
    return expected


def _write_new(path: Path, document: dict[str, Any]) -> None:
    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = path.open("x", encoding="utf-8")
    with descriptor:
        json.dump(document, descriptor, indent=2, sort_keys=True)
        descriptor.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    build.add_argument("--root", required=True, type=Path)
    build.add_argument("--repository", required=True)
    build.add_argument("--revision", required=True)
    build.add_argument("--output", required=True, type=Path)
    build.add_argument("--workers", type=int, default=4)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", required=True, type=Path)
    verify.add_argument("--manifest", required=True, type=Path)
    verify.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    if args.command == "build":
        root = args.root.resolve()
        output = args.output.resolve()
        if output == root or root in output.parents:
            parser.error("--output must be outside --root")
        document = build_manifest(
            root,
            repository=args.repository,
            revision=args.revision,
            workers=args.workers,
        )
        try:
            _write_new(output, document)
        except FileExistsError:
            parser.error(f"refusing to overwrite manifest: {output}")
    else:
        try:
            document = json.loads(args.manifest.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            parser.error(f"cannot read checkpoint manifest: {error}")
        document = verify_manifest(
            args.root,
            document,
            workers=args.workers,
        )
    print(
        json.dumps(
            {
                "checkpoint_sha256": document["checkpoint_sha256"],
                "file_count": document["file_count"],
                "total_bytes": document["total_bytes"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
