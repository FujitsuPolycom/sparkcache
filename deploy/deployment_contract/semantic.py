"""Deterministic HTTP store/restart/restore qualification mechanics."""

from __future__ import annotations

import hashlib
import json
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

EXPECTED_LONG = "SPARKCACHE_OK:9540"
EXPECTED_SHORT = "SPARKCACHE_CANARY_OK"
CANARY_PROMPT = "Respond with exactly SPARKCACHE_CANARY_OK and no other text."

# This identifier is a frozen compatibility interface used by existing
# DeepSeek-V4 and GLM-5.2 reference files. Its model-specific spelling does
# not constrain the model-neutral qualification mechanics.
SEMANTIC_REFERENCE_SCHEMA = "sparkcache-deepseek-semantic-reference/v1"

RequestFunction = Callable[[str, str, str, int], dict[str, Any]]
ContentReader = Callable[[dict[str, Any]], str]


def build_long_prompt(records: int = 384) -> str:
    """Build a stable multi-thousand-token fact-retrieval prompt."""

    if not 384 <= records <= 32768:
        raise ValueError("records must be in [384, 32768]")
    alpha_index = records * 17 // 384
    omega_index = records * 311 // 384
    index_width = max(4, len(str(records - 1)))
    archive = []
    for index in range(records):
        if index == alpha_index:
            text = "the alpha cache marker has value 3719"
        elif index == omega_index:
            text = "the omega cache marker has value 5821"
        else:
            text = "the archival copper marker is stable and carries no numeric value"
        archive.append(f"Archive record {index:0{index_width}d}: {text}.")
    return (
        "Read the archive below. Retain the two explicitly numeric cache marker "
        "values; all other records are padding.\n\n"
        + "\n".join(archive)
        + "\n\nAdd the alpha and omega cache marker values. Respond with exactly "
        "SPARKCACHE_OK:<sum>, with no other text."
    )


def request_chat(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    """Send one deterministic OpenAI-compatible chat request."""

    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0,
        },
        separators=(",", ":"),
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=900) as response:  # noqa: S310
        return json.load(response)


def assistant_content(response: dict[str, Any]) -> str:
    """Return normalized assistant content from one chat response."""

    try:
        return str(response["choices"][0]["message"]["content"]).strip()
    except (KeyError, IndexError, TypeError) as error:
        raise RuntimeError("completion response has no assistant content") from error


def run_semantic_miss(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    long_max_tokens: int = 32,
    records: int = 384,
    request: RequestFunction = request_chat,
    content_reader: ContentReader = assistant_content,
) -> dict[str, Any]:
    """Store the deterministic miss response and its prompt identity."""

    prompt = build_long_prompt(records)
    response = request(endpoint, model, prompt, long_max_tokens)
    content = content_reader(response)
    if content != EXPECTED_LONG:
        raise RuntimeError(f"long semantic miss failed: {content!r}")
    result = {
        "schema": SEMANTIC_REFERENCE_SCHEMA,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "records": records,
        "content": content,
        "usage": response.get("usage"),
    }
    reference.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def run_semantic_hit(
    endpoint: str,
    model: str,
    reference: Path,
    *,
    long_max_tokens: int = 32,
    short_max_tokens: int = 32,
    records: int | None = None,
    request: RequestFunction = request_chat,
    content_reader: ContentReader = assistant_content,
) -> dict[str, Any]:
    """Require prompt identity, restored output, and a post-restore canary."""

    expected = json.loads(reference.read_text(encoding="utf-8"))
    if expected.get("schema") != SEMANTIC_REFERENCE_SCHEMA:
        raise RuntimeError("semantic reference has an unsupported schema")
    expected_records = expected.get("records", 384)
    if records is not None and records != expected_records:
        raise RuntimeError("semantic reference record count differs")
    prompt = build_long_prompt(int(expected_records))
    digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
    if expected.get("prompt_sha256") != digest:
        raise RuntimeError("semantic reference prompt identity differs")
    prime_content = content_reader(
        request(endpoint, model, CANARY_PROMPT, short_max_tokens)
    )
    if prime_content != EXPECTED_SHORT:
        raise RuntimeError(
            "manifest-inventory publication canary failed: "
            f"{prime_content!r}"
        )
    content = content_reader(request(endpoint, model, prompt, long_max_tokens))
    if content != EXPECTED_LONG or content != expected.get("content"):
        raise RuntimeError(f"long semantic hit failed: {content!r}")
    canary_content = content_reader(
        request(endpoint, model, CANARY_PROMPT, short_max_tokens)
    )
    if canary_content != EXPECTED_SHORT:
        raise RuntimeError(f"post-restore semantic canary failed: {canary_content!r}")
    return {"content": content, "post_restore_canary": canary_content}
