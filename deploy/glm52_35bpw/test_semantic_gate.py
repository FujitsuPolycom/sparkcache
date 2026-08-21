from __future__ import annotations

from pathlib import Path

from deploy.glm52_35bpw.semantic_gate import run_hit_after_quorum


def test_hit_gate_passes_an_explicit_record_count(monkeypatch, tmp_path: Path) -> None:
    options: dict[str, int] = {}
    monkeypatch.setattr(
        "deploy.glm52_35bpw.semantic_gate._request",
        lambda *_args, **_kwargs: {
            "choices": [{"message": {"content": "2"}}]
        },
    )

    def hit(_endpoint: str, _model: str, _reference: Path, **values: int) -> dict:
        options.update(values)
        return {"content": "SPARKCACHE_OK:9540", "post_restore_canary": "42"}

    monkeypatch.setattr("deploy.glm52_35bpw.semantic_gate.run_hit", hit)
    run_hit_after_quorum(
        "http://stack",
        "glm-5.2-exl3-r7-3.5bpw",
        tmp_path / "reference.json",
        records=12000,
    )
    assert options == {
        "long_max_tokens": 256,
        "short_max_tokens": 128,
        "records": 12000,
    }
