from __future__ import annotations

import hashlib

from deploy.glm53_flash.qualification_request import (
    SEMANTIC_ANSWER,
    persistent_prompt,
    request_body,
    semantic_content_matches,
    semantic_prompt,
)


def test_persistent_prompt_has_stable_identity() -> None:
    prompt = persistent_prompt()
    assert prompt.count("benchmark ") == 8192
    assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == (
        "a8569c46a6cbf22bae4736c897023f7c552952440bf75e1ef6ebabe594f513cf"
    )


def test_semantic_prompt_names_the_expected_answer() -> None:
    assert semantic_prompt().endswith(SEMANTIC_ANSWER + " and no other text.")


def test_semantic_request_uses_parser_compatible_thinking_mode() -> None:
    body = request_body("semantic", "glm-model")
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["messages"] == [
        {"role": "user", "content": semantic_prompt()}
    ]

    persistent = request_body("persistent", "glm-model")
    assert "chat_template_kwargs" not in persistent


def test_semantic_validator_requires_exact_visible_content() -> None:
    assert semantic_content_matches(SEMANTIC_ANSWER)
    assert not semantic_content_matches(
        "internal reasoning</think>\n" + SEMANTIC_ANSWER
    )
    assert not semantic_content_matches(SEMANTIC_ANSWER + "\n")
    assert not semantic_content_matches(None)
