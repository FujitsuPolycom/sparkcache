"""Tests for the guarded DeepSeek TP4 rank launcher."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.deepseek_v4 import tp4_launch
from deploy.deepseek_v4.test_tp4_profile import CHECKPOINT, IMAGE, _source


def _arguments(tmp_path: Path) -> list[str]:
    inspection = tmp_path / "inspect.json"
    inspection.write_text(json.dumps([_source(0)]), encoding="utf-8")
    paths = {}
    for name in ("cache", "jit", "source"):
        path = tmp_path / name
        path.mkdir()
        paths[name] = path
    scheduler = tmp_path / "scheduler.py"
    config = tmp_path / "vllm.py"
    receipt = tmp_path / "receipt.json"
    for path in (scheduler, config, receipt):
        path.write_text("fixture\n", encoding="utf-8")
    return [
        "--inspect",
        str(inspection),
        "--image",
        IMAGE,
        "--name",
        "deepseek0731-sparkcache-r0",
        "--checkpoint-sha256",
        CHECKPOINT,
        "--cache-host-path",
        str(paths["cache"]),
        "--jit-host-path",
        str(paths["jit"]),
        "--sparkcache-source-host-path",
        str(paths["source"]),
        "--scheduler-overlay-host-path",
        str(scheduler),
        "--vllm-config-overlay-host-path",
        str(config),
        "--vllm-overlay-receipt-host-path",
        str(receipt),
        "--api-port",
        "8100",
        "--master-port",
        "29600",
        "--create-only",
    ]


def test_launcher_requires_and_passes_disjoint_rank_local_binds(
    monkeypatch, tmp_path: Path
) -> None:
    captured = {}
    monkeypatch.setattr(tp4_launch, "_validate_overlays", lambda *args: None)
    monkeypatch.setattr(
        tp4_launch,
        "launch",
        lambda *args, **kwargs: captured.update(args=args, kwargs=kwargs),
    )

    assert tp4_launch.main(_arguments(tmp_path)) == 0

    assert captured["args"][1:] == (
        IMAGE,
        "deepseek0731-sparkcache-r0",
        CHECKPOINT,
    )
    kwargs = captured["kwargs"]
    assert kwargs["create_only"] is True
    assert kwargs["preserve_all_binds"] is True
    assert kwargs["entrypoint"] == "/opt/venv/bin/vllm"
    assert kwargs["labels"]["org.sparkcache.managed"] == "true"
    binds = {container: (host, read_only) for host, container, read_only in kwargs["extra_binds"]}
    assert binds["/cache/sparkcache-deepseek0731-tp4-dcp1"][1] is False
    assert binds["/cache/jit"][1] is False
    assert binds["/opt/sparkcache-src/sparkcache"][1] is True
    command = captured["args"][0]["Config"]["Cmd"]
    assert command[command.index("--port") + 1] == "8100"
    assert command[command.index("--master-port") + 1] == "29600"


def test_launcher_rejects_an_image_other_than_the_inspection(monkeypatch, tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    arguments[arguments.index("--image") + 1] = "sha256:" + "c" * 64
    monkeypatch.setattr(tp4_launch, "launch", lambda *args, **kwargs: None)
    with pytest.raises(SystemExit):
        tp4_launch.main(arguments)


def test_launcher_rejects_overlapping_cache_and_jit_roots(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    cache = Path(arguments[arguments.index("--cache-host-path") + 1])
    nested = cache / "jit"
    nested.mkdir()
    arguments[arguments.index("--jit-host-path") + 1] = str(nested)
    with pytest.raises(SystemExit):
        tp4_launch.main(arguments)


def test_launcher_rejects_missing_rank_local_root(tmp_path: Path) -> None:
    arguments = _arguments(tmp_path)
    missing = tmp_path / "absent"
    arguments[arguments.index("--cache-host-path") + 1] = str(missing)
    with pytest.raises(SystemExit):
        tp4_launch.main(arguments)
