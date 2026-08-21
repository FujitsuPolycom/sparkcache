"""Prepare exact SparkCache overlays for the DeepSeek-0731 TP4 runtime."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from deploy.deployment_contract import (
    apply_verified_patch,
    file_sha256,
    source_tree_sha256,
)


@dataclass(frozen=True)
class PatchStep:
    patch: str
    preimage_sha256: str
    postimage_sha256: str


@dataclass(frozen=True)
class OverlayChain:
    source: str
    output: str
    steps: tuple[PatchStep, ...]


OVERLAYS = (
    OverlayChain(
        source="vllm/v1/core/sched/scheduler.py",
        output="scheduler.py",
        steps=(
            PatchStep(
                patch=(
                    "patches/vllm-e2666d9a6/"
                    "011-sparkcache-glm52-async-rollback.patch"
                ),
                preimage_sha256=(
                    "1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce"
                ),
                postimage_sha256=(
                    "d4ebec211b027b6c7f64574f79374237de0f5fde0c5c03f20f1cb1596ffadc3a"
                ),
            ),
            PatchStep(
                patch=(
                    "patches/vllm-e2666d9a6/"
                    "031-sparkcache-stock-hma-load-failure.patch"
                ),
                preimage_sha256=(
                    "d4ebec211b027b6c7f64574f79374237de0f5fde0c5c03f20f1cb1596ffadc3a"
                ),
                postimage_sha256=(
                    "2f34aa9d65a495a86d814c90f654fbe1ff754cfdbecd204b98d513652ca3e06d"
                ),
            ),
        ),
    ),
    OverlayChain(
        source="vllm/config/vllm.py",
        output="vllm.py",
        steps=(
            PatchStep(
                patch="patches/vllm-e2666d9a6/020-sparkcache-vmm-exemption.patch",
                preimage_sha256=(
                    "fbc581651521d8f5fb753be7bb9baa24deddac5dcc7cef5da27d6a6b9d99af5f"
                ),
                postimage_sha256=(
                    "71c4f9e622dd8b3d665f2a2b5fb932206516ddb82873ff89283c63aa80696005"
                ),
            ),
        ),
    ),
)


_sha256 = file_sha256


def _apply_step(
    work: Path,
    repository: Path,
    step: PatchStep,
    index: int,
    staged_source: Path,
) -> None:
    apply_verified_patch(
        work=work,
        patch_source=repository / step.patch,
        staged_source=staged_source,
        expected_preimage_sha256=step.preimage_sha256,
        expected_postimage_sha256=step.postimage_sha256,
        patch_name=f".step-{index:02d}.patch",
        role="DeepSeek-V4 vLLM overlay",
    )


def _prepare_chain(
    chain: OverlayChain,
    *,
    vllm_root: Path,
    repository: Path,
    work: Path,
) -> dict[str, object]:
    source = vllm_root / chain.source
    if not source.is_file():
        raise RuntimeError(f"DeepSeek vLLM source is missing: {source}")
    staged = work / chain.source
    staged.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, staged)
    initial = _sha256(staged)
    hashes = [chain.steps[0].preimage_sha256]
    hashes.extend(step.postimage_sha256 for step in chain.steps)
    if initial not in hashes:
        raise RuntimeError(
            f"unsupported DeepSeek vLLM preimage for {chain.source}: {initial}"
        )
    start = hashes.index(initial)
    applied: list[str] = []
    for index, step in enumerate(chain.steps[start:], start=start):
        _apply_step(work, repository, step, index, staged)
        applied.append(step.patch)
    final = _sha256(staged)
    expected = chain.steps[-1].postimage_sha256
    if final != expected:
        raise RuntimeError(
            f"DeepSeek final overlay differs for {chain.source}: {final}"
        )
    return {
        "source": chain.source,
        "output": chain.output,
        "initial_sha256": initial,
        "sha256": final,
        "applied_patches": applied,
    }


def prepare(vllm_root: Path, repository: Path, output: Path) -> dict[str, object]:
    """Create exact scheduler/config overlays and a source-bound receipt."""

    vllm_root = vllm_root.resolve()
    repository = repository.resolve()
    output = output.resolve()
    if output.exists():
        raise RuntimeError(f"refusing to overwrite overlay output: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="sparkcache-deepseek0731-tp4-overlays-",
        dir=output.parent,
    ) as temporary:
        work = Path(temporary)
        subprocess.run(["git", "init", "-q"], cwd=work, check=True)
        records = [
            _prepare_chain(
                chain,
                vllm_root=vllm_root,
                repository=repository,
                work=work,
            )
            for chain in OVERLAYS
        ]
        output.mkdir()
        for chain in OVERLAYS:
            shutil.copy2(work / chain.source, output / chain.output)
        receipt = {
            "schema": "sparkcache-deepseek0731-tp4-vllm-overlays/v1",
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
