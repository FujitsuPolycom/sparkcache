from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


def _load_harness():
    path = Path(__file__).with_name("measure_page_tail_interference.py")
    spec = importlib.util.spec_from_file_location("page_tail_interference", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cli_compares_equal_concurrency_and_writes_receipt(
    tmp_path: Path,
    monkeypatch,
) -> None:
    harness = _load_harness()
    calls: list[str] = []

    def sentinel(_endpoint, _model, nonce, _timeout):
        calls.append(nonce)
        return 2.0 if nonce.startswith("overlap-") else 1.0

    monkeypatch.setattr(harness, "sentinel", sentinel)
    monkeypatch.setattr(
        harness,
        "chat",
        lambda *_args, **_kwargs: (
            {"choices": [{"message": {"content": "GOLD"}}]},
            12.5,
        ),
    )
    output = tmp_path / "receipt.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "measure_page_tail_interference.py",
            "--endpoint",
            "http://127.0.0.1:8000",
            "--concurrency",
            "3",
            "--output",
            str(output),
        ],
    )

    assert harness.main() == 0

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert receipt["schema"] == "sparkcache-page-tail-interference/v1"
    assert receipt["baseline_seconds"] == [1.0, 1.0, 1.0]
    assert receipt["overlap_seconds"] == [2.0, 2.0, 2.0]
    assert receipt["baseline_median_seconds"] == 1.0
    assert receipt["overlap_median_seconds"] == 2.0
    assert len([value for value in calls if value.startswith("baseline-")]) == 3
    assert len([value for value in calls if value.startswith("overlap-")]) == 3
