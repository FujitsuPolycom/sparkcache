from __future__ import annotations

import hashlib

from deploy.glm53_flash.qualification_request import (
    SEMANTIC_ANSWER,
    persistent_prompt,
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
