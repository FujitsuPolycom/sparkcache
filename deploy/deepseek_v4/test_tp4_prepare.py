"""Tests for the exact DeepSeek TP4 vLLM overlay builder."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import pytest

from deploy.deepseek_v4 import tp4_prepare_vllm_overlays as overlays


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _patch(path: Path, before: str, after: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(
        (
            "diff --git a/vllm/example.py b/vllm/example.py\r\n"
            "--- a/vllm/example.py\r\n"
            "+++ b/vllm/example.py\r\n"
            "@@ -1 +1 @@\r\n"
            f"-{before}\r\n"
            f"+{after}\r\n"
        ).encode()
    )


def _fixture(monkeypatch, tmp_path: Path, initial: bytes = b"old\n"):
    repository = tmp_path / "repository"
    package = repository / "sparkcache"
    package.mkdir(parents=True)
    (package / "connector.py").write_text("# source\n", encoding="utf-8")
    _patch(repository / "patches/one.patch", "old", "middle")
    _patch(repository / "patches/two.patch", "middle", "new")
    chain = overlays.OverlayChain(
        source="vllm/example.py",
        output="example.py",
        steps=(
            overlays.PatchStep(
                "patches/one.patch",
                _digest(b"old\n"),
                _digest(b"middle\n"),
            ),
            overlays.PatchStep(
                "patches/two.patch",
                _digest(b"middle\n"),
                _digest(b"new\n"),
            ),
        ),
    )
    monkeypatch.setattr(overlays, "OVERLAYS", (chain,))
    vllm_root = tmp_path / "vllm-root"
    source = vllm_root / "vllm/example.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(initial)
    return repository, vllm_root


def test_prepare_applies_both_exact_steps_and_normalizes_patch_eol(
    monkeypatch, tmp_path: Path
) -> None:
    repository, vllm_root = _fixture(monkeypatch, tmp_path)
    output = tmp_path / "output"
    receipt = overlays.prepare(vllm_root, repository, output)

    assert (output / "example.py").read_bytes() == b"new\n"
    assert receipt["schema"] == "sparkcache-deepseek0731-tp4-vllm-overlays/v1"
    assert receipt["files"][0]["applied_patches"] == [
        "patches/one.patch",
        "patches/two.patch",
    ]
    assert receipt["files"][0]["sha256"] == _digest(b"new\n")
    assert (output / "receipt.json").is_file()


def test_prepare_isolated_from_an_unrelated_parent_git_checkout(
    monkeypatch, tmp_path: Path
) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    repository, vllm_root = _fixture(monkeypatch, tmp_path)
    output = tmp_path / "output"

    overlays.prepare(vllm_root, repository, output)

    assert (output / "example.py").read_bytes() == b"new\n"


def test_prepare_resumes_only_from_an_exact_intermediate(
    monkeypatch, tmp_path: Path
) -> None:
    repository, vllm_root = _fixture(
        monkeypatch,
        tmp_path,
        initial=b"middle\n",
    )
    receipt = overlays.prepare(vllm_root, repository, tmp_path / "output")
    assert receipt["files"][0]["applied_patches"] == ["patches/two.patch"]


def test_prepare_accepts_an_exact_final_without_reapplication(
    monkeypatch, tmp_path: Path
) -> None:
    repository, vllm_root = _fixture(monkeypatch, tmp_path, initial=b"new\n")
    receipt = overlays.prepare(vllm_root, repository, tmp_path / "output")
    assert receipt["files"][0]["applied_patches"] == []


def test_prepare_rejects_an_unrecognized_scheduler_lineage(
    monkeypatch, tmp_path: Path
) -> None:
    repository, vllm_root = _fixture(monkeypatch, tmp_path, initial=b"adaptive\n")
    with pytest.raises(RuntimeError, match="unsupported DeepSeek vLLM preimage"):
        overlays.prepare(vllm_root, repository, tmp_path / "output")


def test_prepare_refuses_to_overwrite_output(monkeypatch, tmp_path: Path) -> None:
    repository, vllm_root = _fixture(monkeypatch, tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        overlays.prepare(vllm_root, repository, output)
