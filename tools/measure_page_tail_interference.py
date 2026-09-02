#!/usr/bin/env python3
"""Measure short-request latency while a growing context is published."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import time
import urllib.request
from pathlib import Path


def chat(endpoint: str, body: dict, timeout: float) -> tuple[dict, float]:
    request = urllib.request.Request(
        endpoint.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    return result, time.perf_counter() - started


def sentinel(endpoint: str, model: str, nonce: str, timeout: float) -> float:
    result, elapsed = chat(
        endpoint,
        {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Request {nonce}. Your entire final answer must be exactly"
                        " OK and nothing else."
                    ),
                }
            ],
            "temperature": 0.0,
            "max_tokens": 128,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        timeout,
    )
    content = result["choices"][0]["message"].get("content")
    if content != "OK":
        raise RuntimeError(f"sentinel returned {content!r}")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--repetitions", type=int, default=262144)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        baseline = list(
            executor.map(
                lambda index: sentinel(
                    args.endpoint,
                    args.model,
                    f"baseline-{time.time_ns()}-{index}",
                    args.timeout,
                ),
                range(args.concurrency),
            )
        )
    result, long_elapsed = chat(
        args.endpoint,
        {
            "model": args.model,
            "messages": [
                {
                    "role": "user",
                    "content": (
                        "benchmark " * args.repetitions
                        + "\nRespond with exactly GOLD and no other text."
                    ),
                }
            ],
            "temperature": 0.0,
            "max_tokens": 256,
            "chat_template_kwargs": {"enable_thinking": True},
        },
        args.timeout,
    )
    if result["choices"][0]["message"].get("content") != "GOLD":
        raise RuntimeError("growing-context request returned the wrong codeword")

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=args.concurrency
    ) as executor:
        overlap = list(
            executor.map(
                lambda index: sentinel(
                    args.endpoint,
                    args.model,
                    f"overlap-{time.time_ns()}-{index}",
                    args.timeout,
                ),
                range(args.concurrency),
            )
        )

    receipt = {
        "schema": "sparkcache-page-tail-interference/v1",
        "context_repetitions": args.repetitions,
        "concurrency": args.concurrency,
        "long_request_seconds": round(long_elapsed, 3),
        "baseline_seconds": [round(value, 4) for value in baseline],
        "overlap_seconds": [round(value, 4) for value in overlap],
        "baseline_median_seconds": round(statistics.median(baseline), 4),
        "overlap_median_seconds": round(statistics.median(overlap), 4),
        "overlap_max_seconds": round(max(overlap), 4),
    }
    args.output.write_text(
        json.dumps(receipt, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
