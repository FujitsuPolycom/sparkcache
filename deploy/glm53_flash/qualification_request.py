#!/usr/bin/env python3
"""Run the persistent-prefix or semantic GLM-5.3 qualification request."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
import urllib.request
from pathlib import Path


SEMANTIC_ANSWER = "SPARKCACHE_GLM53_OK"


def persistent_prompt() -> str:
    """Return the deterministic text whose reusable span is 8,192 tokens."""

    return "benchmark " * 8192 + "\nRequest 0: summarize the repeated prefix briefly."


def semantic_prompt() -> str:
    """Return the uncached continued-generation canary input."""

    return f"Respond with exactly {SEMANTIC_ANSWER} and no other text."


def request_body(kind: str, model: str) -> dict[str, object]:
    """Build the request whose response is evaluated by this qualifier."""

    prompt = persistent_prompt() if kind == "persistent" else semantic_prompt()
    body: dict[str, object] = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 1.0 if kind == "persistent" else 0.0,
        "max_tokens": 64 if kind == "persistent" else 256,
    }
    if kind == "semantic":
        # GLM's chat parser separates reasoning from visible content when
        # thinking is enabled. Disabling it on this runtime can leave raw
        # reasoning and a closing </think> marker in message.content.
        body["chat_template_kwargs"] = {"enable_thinking": True}
    return body


def semantic_content_matches(content: object) -> bool:
    """Accept only the exact visible answer requested by the canary."""

    return isinstance(content, str) and content == SEMANTIC_ANSWER


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("persistent", "semantic"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=600.0)
    args = parser.parse_args()

    prompt = persistent_prompt() if args.kind == "persistent" else semantic_prompt()
    encoded = json.dumps(request_body(args.kind, args.model)).encode("utf-8")
    request = urllib.request.Request(
        args.endpoint.rstrip("/") + "/v1/chat/completions",
        data=encoded,
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=args.timeout) as response:
        body = json.load(response)
    elapsed = time.perf_counter() - started
    choice = body["choices"][0]
    raw_content = choice["message"].get("content")
    content = raw_content if isinstance(raw_content, str) else ""
    semantic_match = (
        semantic_content_matches(raw_content) if args.kind == "semantic" else None
    )
    receipt = {
        "schema": "sparkcache-glm53-qualification-request/v1",
        "kind": args.kind,
        "model": body.get("model"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "elapsed_seconds": round(elapsed, 3),
        "finish_reason": choice.get("finish_reason"),
        "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        "content_characters": len(content),
        "semantic_match": semantic_match,
        "usage": body.get("usage"),
    }
    finish_reason = choice.get("finish_reason")
    completion_ok = (
        finish_reason in {"stop", "length"}
        if args.kind == "persistent"
        else finish_reason == "stop"
    )
    validation_passed = completion_ok and semantic_match is not False
    receipt["validation_passed"] = validation_passed
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(receipt, indent=2))
    return 0 if validation_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
