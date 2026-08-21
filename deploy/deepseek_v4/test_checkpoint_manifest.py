"""Tests for complete immutable checkpoint manifests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.deepseek_v4.checkpoint_manifest import (
    build_manifest,
    main,
    verify_manifest,
)


def _checkpoint(root: Path) -> None:
    root.mkdir()
    (root / "config.json").write_text('{"model_type":"deepseek_v4"}\n')
    (root / "model-00001-of-00002.safetensors").write_bytes(b"weights-a")
    (root / "model-00002-of-00002.safetensors").write_bytes(b"weights-b")
    (root / "tokenizer").mkdir()
    (root / "tokenizer/tokenizer.json").write_text('{"version":1}\n')


def test_manifest_is_deterministic_and_verifies_complete_tree(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    first = build_manifest(
        root,
        repository="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="revision-a",
        workers=2,
    )
    second = build_manifest(
        root,
        repository="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="revision-a",
        workers=1,
    )

    assert first == second
    assert first["file_count"] == 4
    assert first["files"] == sorted(first["files"], key=lambda item: item["path"])
    assert len(first["checkpoint_sha256"]) == 64
    assert verify_manifest(root, first, workers=2) == first


def test_manifest_detects_changed_missing_and_extra_files(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    manifest = build_manifest(
        root,
        repository="deepseek-ai/DeepSeek-V4-Flash-0731",
        revision="revision-a",
    )

    (root / "config.json").write_text('{"model_type":"other"}\n')
    with pytest.raises(RuntimeError, match="differs"):
        verify_manifest(root, manifest)
    (root / "config.json").write_text('{"model_type":"deepseek_v4"}\n')
    (root / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(RuntimeError, match="differs"):
        verify_manifest(root, manifest)
    (root / "model-00002-of-00002.safetensors").write_bytes(b"weights-b")
    (root / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(RuntimeError, match="differs"):
        verify_manifest(root, manifest)


def test_manifest_rejects_symlinks(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    link = root / "linked-config.json"
    try:
        link.symlink_to(root / "config.json")
    except OSError:
        pytest.skip("symlink creation is unavailable")
    with pytest.raises(RuntimeError, match="symlink"):
        build_manifest(
            root,
            repository="deepseek-ai/DeepSeek-V4-Flash-0731",
            revision="revision-a",
        )


def test_git_metadata_is_not_part_of_checkpoint_identity(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    first = build_manifest(root, repository="repo", revision="revision")
    (root / ".git").mkdir()
    (root / ".git/index").write_bytes(b"mutable-git-state")
    second = build_manifest(root, repository="repo", revision="revision")
    assert first == second


def test_download_cache_is_not_part_of_checkpoint_identity(tmp_path: Path) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    first = build_manifest(root, repository="repo", revision="revision")
    (root / ".cache/huggingface/download").mkdir(parents=True)
    (root / ".cache/huggingface/download/config.json.lock").write_bytes(b"")
    (root / ".cache/huggingface/trees").mkdir()
    (root / ".cache/huggingface/trees/revision.json").write_text("{}\n")
    second = build_manifest(root, repository="repo", revision="revision")
    assert first == second


def test_cli_builds_once_and_verifies(tmp_path: Path, monkeypatch, capsys) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    output = tmp_path / "manifest.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "checkpoint_manifest.py",
            "build",
            "--root",
            str(root),
            "--repository",
            "deepseek-ai/DeepSeek-V4-Flash-0731",
            "--revision",
            "revision-a",
            "--output",
            str(output),
        ],
    )
    assert main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["file_count"] == 4
    monkeypatch.setattr(
        "sys.argv",
        [
            "checkpoint_manifest.py",
            "verify",
            "--root",
            str(root),
            "--manifest",
            str(output),
        ],
    )
    assert main() == 0


def test_cli_rejects_manifest_output_inside_checkpoint(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "model"
    _checkpoint(root)
    monkeypatch.setattr(
        "sys.argv",
        [
            "checkpoint_manifest.py",
            "build",
            "--root",
            str(root),
            "--repository",
            "repo",
            "--revision",
            "revision",
            "--output",
            str(root / "manifest.json"),
        ],
    )
    with pytest.raises(SystemExit):
        main()
