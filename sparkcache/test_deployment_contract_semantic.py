from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from deploy.deployment_contract.semantic import (
    AssistantCompletion,
    SemanticGateInconclusive,
    assistant_completion,
    assistant_content,
    run_semantic_hit,
    run_semantic_miss,
)
from deploy.glm52_35bpw import semantic_gate as glm_semantic_gate
from deploy.deepseek_v4 import semantic_gate as deepseek_semantic_gate


def _response(
    *,
    finish_reason: str | None = "stop",
    reasoning: str | None = None,
    reasoning_content: str | None = None,
    content: str | None = None,
) -> dict:
    message = {
        "reasoning": reasoning,
        "reasoning_content": reasoning_content,
        "content": content,
    }
    return {
        "choices": [
            {
                "finish_reason": finish_reason,
                "message": message,
            }
        ]
    }


def _run_miss(response: dict, reference: Path) -> dict:
    return run_semantic_miss(
        "http://stack",
        "model",
        reference,
        request=lambda *_args: response,
    )


def test_semantic_miss_reports_empty_assistant_body_as_inconclusive(
    tmp_path: Path,
) -> None:
    with pytest.raises(SemanticGateInconclusive) as raised:
        _run_miss(_response(content=" \n "), tmp_path / "reference.json")

    assert raised.value.as_result() == {
        "assistant_body_present": False,
        "check": "long semantic miss",
        "finish_reason": "stop",
        "reason": "completion has no non-whitespace assistant body",
        "status": "INCONCLUSIVE",
    }


def test_semantic_miss_treats_reasoning_only_response_as_semantic_failure(
    tmp_path: Path,
) -> None:
    response = _response(reasoning_content=" SPARKCACHE_OK:9540 ", content=None)

    with pytest.raises(RuntimeError, match="long semantic miss failed: ''"):
        _run_miss(response, tmp_path / "reference.json")

    assert assistant_completion(response).body == "SPARKCACHE_OK:9540"
    assert assistant_completion(response).finish_reason == "stop"


def test_semantic_miss_reports_length_truncation_as_inconclusive(
    tmp_path: Path,
) -> None:
    with pytest.raises(SemanticGateInconclusive) as raised:
        _run_miss(
            _response(finish_reason="length", content="SPARKCACHE_OK:9540"),
            tmp_path / "reference.json",
        )

    assert raised.value.as_result() == {
        "assistant_body_present": True,
        "check": "long semantic miss",
        "finish_reason": "length",
        "reason": "completion token limit reached",
        "status": "INCONCLUSIVE",
    }


def test_semantic_miss_records_combined_body_in_v1_reference(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    result = _run_miss(
        _response(content="SPARKCACHE_OK:9540"),
        reference,
    )

    assert result["content"] == "SPARKCACHE_OK:9540"
    assert set(json.loads(reference.read_text(encoding="utf-8"))) == {
        "assistant_body",
        "content",
        "prompt_sha256",
        "records",
        "schema",
        "usage",
    }


def test_assistant_completion_combines_supported_fields_in_stable_order() -> None:
    response = _response(
        reasoning="first-",
        reasoning_content="second-",
        content="third",
    )

    completion = assistant_completion(response)
    assert completion.body == "first-second-third"
    assert assistant_content(response) == "third"


def test_semantic_miss_compares_exact_answer_separately_from_reasoning(
    tmp_path: Path,
) -> None:
    reference = tmp_path / "reference.json"
    result = _run_miss(
        _response(
            reasoning_content="deterministic-reasoning\n",
            content="SPARKCACHE_OK:9540",
        ),
        reference,
    )

    assert result["content"] == "SPARKCACHE_OK:9540"
    assert json.loads(reference.read_text(encoding="utf-8"))["assistant_body"] == (
        "deterministic-reasoning\nSPARKCACHE_OK:9540"
    )


def test_semantic_hit_rejects_changed_combined_body(tmp_path: Path) -> None:
    reference = tmp_path / "reference.json"
    _run_miss(
        _response(
            reasoning_content="stored-reasoning\n",
            content="SPARKCACHE_OK:9540",
        ),
        reference,
    )
    responses = iter(
        [
            _response(content="SPARKCACHE_CANARY_OK"),
            _response(
                reasoning_content="different-reasoning\n",
                content="SPARKCACHE_OK:9540",
            ),
        ]
    )

    with pytest.raises(RuntimeError, match="assistant body differs"):
        run_semantic_hit(
            "http://stack",
            "model",
            reference,
            request=lambda *_args: next(responses),
        )


@pytest.mark.parametrize(
    "semantic_module",
    (deepseek_semantic_gate, glm_semantic_gate),
)
def test_model_cli_reports_inconclusive_result(
    semantic_module,
    monkeypatch,
    capsys,
    tmp_path: Path,
) -> None:
    error = SemanticGateInconclusive(
        "long semantic miss",
        "completion token limit reached",
        completion=AssistantCompletion(
            body="SPARKCACHE_OK:9540",
            content="SPARKCACHE_OK:9540",
            finish_reason="length",
        ),
    )

    def inconclusive(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(semantic_module, "run_miss", inconclusive)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "semantic_gate.py",
            "miss",
            "--reference",
            str(tmp_path / "reference.json"),
        ],
    )

    with pytest.raises(SystemExit) as raised:
        semantic_module.main()

    assert raised.value.code == 2
    assert json.loads(capsys.readouterr().out) == error.as_result()
