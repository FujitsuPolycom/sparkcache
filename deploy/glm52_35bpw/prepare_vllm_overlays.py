"""Prepare vLLM overlays from the GLM-5.2 fixed-MTP4 serving recipe (``R7``)."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deploy.deployment_contract import (
    apply_verified_patch,
    file_sha256,
    source_tree_sha256,
)


@dataclass(frozen=True)
class OverlaySpec:
    source: str
    patch: str
    output: str
    preimage_sha256: str
    postimage_sha256: str


OVERLAYS = (
    OverlaySpec(
        source="vllm/v1/core/sched/scheduler.py",
        patch=(
            "patches/vllm-e2666d9a6/"
            "011-sparkcache-glm52-async-rollback.patch"
        ),
        output="scheduler.py",
        preimage_sha256=(
            "1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce"
        ),
        postimage_sha256=(
            "d4ebec211b027b6c7f64574f79374237de0f5fde0c5c03f20f1cb1596ffadc3a"
        ),
    ),
    OverlaySpec(
        source="vllm/config/vllm.py",
        patch="patches/vllm-e2666d9a6/020-sparkcache-vmm-exemption.patch",
        output="vllm.py",
        preimage_sha256=(
            "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f"
        ),
        postimage_sha256=(
            "71c4f9e622dd8b3d665f2a2b5fb932206516ddb82873ff89283c63aa80696005"
        ),
    ),
)


_sha256 = file_sha256


def prepare(vllm_root: Path, repository: Path, output: Path) -> dict[str, object]:
    """Copy or patch exact preimages, publishing output only after verification."""

    vllm_root = vllm_root.resolve()
    repository = repository.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite overlay output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, str]] = []
    with tempfile.TemporaryDirectory(
        prefix="sparkcache-r7-overlays-",
        dir=output.parent,
    ) as temporary:
        work = Path(temporary)
        for spec in OVERLAYS:
            source = vllm_root / spec.source
            if not source.is_file():
                raise RuntimeError(
                    f"GLM-5.2 serving recipe R7 vLLM source is missing: {source}"
                )
            staged = work / spec.source
            staged.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, staged)
            before = _sha256(staged)
            if before == spec.preimage_sha256:
                patch_source = repository / spec.patch
                apply_verified_patch(
                    work=work,
                    patch_source=patch_source,
                    staged_source=staged,
                    expected_preimage_sha256=spec.preimage_sha256,
                    expected_postimage_sha256=spec.postimage_sha256,
                    patch_name=f".{spec.output}.patch",
                    role="GLM-5.2 vLLM overlay",
                )
                disposition = "patched"
            elif before == spec.postimage_sha256:
                disposition = "already_patched"
            else:
                raise RuntimeError(
                    f"unsupported GLM-5.2 recipe-R7 preimage for {spec.source}: {before}"
                )
            after = _sha256(staged)
            if after != spec.postimage_sha256:
                raise RuntimeError(
                    f"GLM-5.2 recipe-R7 overlay postimage differs for {spec.source}: {after}"
                )
            records.append(
                {
                    "source": spec.source,
                    "output": spec.output,
                    "sha256": after,
                    "disposition": disposition,
                }
            )

        output.mkdir()
        for spec in OVERLAYS:
            shutil.copy2(work / spec.source, output / spec.output)
        receipt = {
            "schema": "sparkcache-glm52-r7-vllm-overlays/v1",
            "sparkcache_source_sha256": source_tree_sha256(
                repository / "sparkcache"
            ),
            "files": records,
        }
        (output / "receipt.json").write_text(
            json.dumps(receipt, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vllm-root", required=True, type=Path)
    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.vllm_root, args.repository, args.output),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
