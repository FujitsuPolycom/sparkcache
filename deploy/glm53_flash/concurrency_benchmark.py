#!/usr/bin/env python3
"""Issue a synchronized GLM-5.3 OpenAI-compatible request cohort.

The driver deliberately has no cache-management operations. ``cache_state`` is
an operator-supplied label describing the state prepared outside this process;
the driver only sends inference requests and records their results.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


SCHEMA = "sparkcache-glm53-concurrency-benchmark/v1"
CONCURRENCY_LEVELS = (2, 8, 16)
SCENARIOS = ("identical-prefix", "shared-trunk")
CACHE_STATES = ("hot", "cold", "uncontrolled")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Inputs that define one reproducible request cohort."""

    endpoint: str
    model: str
    concurrency: int
    scenario: str
    cache_state: str
    pretokenize: bool = False
    prefix_header: str = "SparkCache CUDA 128K restore test.\n"
    prefix_repetitions: int = 131_072
    tail_repetitions: int = 32
    max_tokens: int = 1
    timeout_seconds: float = 600.0
    api_key: str | None = None

    def validate(self) -> None:
        if self.concurrency not in CONCURRENCY_LEVELS:
            raise ValueError(f"concurrency must be one of {CONCURRENCY_LEVELS}")
        if self.scenario not in SCENARIOS:
            raise ValueError(f"scenario must be one of {SCENARIOS}")
        if self.cache_state not in CACHE_STATES:
            raise ValueError(f"cache_state must be one of {CACHE_STATES}")
        if self.prefix_repetitions < 1:
            raise ValueError("prefix_repetitions must be positive")
        if self.tail_repetitions < 0:
            raise ValueError("tail_repetitions cannot be negative")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")


@dataclass(frozen=True)
class TransportResult:
    """Minimal HTTP result retained by the benchmark."""

    http_status: int
    body: dict[str, Any]


Transport = Callable[[str, bytes, dict[str, str], float], TransportResult]


def build_prompts(config: BenchmarkConfig) -> list[str]:
    """Build stable prompts for identical-prefix or shared-trunk cohorts."""

    config.validate()
    prefix = config.prefix_header + "benchmark " * config.prefix_repetitions
    if config.scenario == "identical-prefix":
        prompt = prefix + "\nReturn one word."
        return [prompt] * config.concurrency

    return [
        prefix
        + (f"tail-{index:02d} " * config.tail_repetitions)
        + f"\nRequest {index:02d}: return one word."
        for index in range(config.concurrency)
    ]


def _request_body(
    config: BenchmarkConfig,
    prompt: str,
    token_ids: tuple[int, ...] | None = None,
) -> bytes:
    body: dict[str, Any] = {
        "model": config.model,
        "temperature": 0.0,
        "max_tokens": config.max_tokens,
        "stream": False,
    }
    if token_ids is None:
        body["messages"] = [{"role": "user", "content": prompt}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        body["prompt"] = list(token_ids)
    return json.dumps(body, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _tokenize_prompt(
    config: BenchmarkConfig,
    prompt: str,
    transport: Transport,
    headers: dict[str, str],
) -> tuple[int, ...]:
    payload = json.dumps(
        {
            "model": config.model,
            "messages": [{"role": "user", "content": prompt}],
            "chat_template_kwargs": {"enable_thinking": False},
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result = transport(
        config.endpoint.rstrip("/") + "/tokenize",
        payload,
        headers,
        config.timeout_seconds,
    )
    tokens = result.body.get("tokens")
    if not 200 <= result.http_status < 300 or not isinstance(tokens, list):
        raise RuntimeError("tokenization endpoint did not return token IDs")
    if not tokens or any(type(token) is not int or token < 0 for token in tokens):
        raise RuntimeError("tokenization endpoint returned invalid token IDs")
    return tuple(tokens)


def _urllib_transport(
    url: str, payload: bytes, headers: dict[str, str], timeout: float
) -> TransportResult:
    request = urllib.request.Request(url, data=payload, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            body = json.load(response)
            return TransportResult(http_status=response.status, body=body)
    except urllib.error.HTTPError as exc:
        try:
            body = json.load(exc)
        except (json.JSONDecodeError, UnicodeDecodeError):
            body = {}
        return TransportResult(http_status=exc.code, body=body)


def _nearest_rank(values: Sequence[float], percentile: float) -> float:
    if not values:
        raise ValueError("cannot calculate a percentile of no values")
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[index]


def _latency_summary(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    latencies = [float(result["elapsed_seconds"]) for result in results]
    succeeded = sum(bool(result["ok"]) for result in results)
    return {
        "request_count": len(results),
        "succeeded": succeeded,
        "failed": len(results) - succeeded,
        "min_seconds": round(min(latencies), 6),
        "p50_seconds": round(_nearest_rank(latencies, 0.50), 6),
        "p95_seconds": round(_nearest_rank(latencies, 0.95), 6),
        "max_seconds": round(max(latencies), 6),
    }


def run_benchmark(
    config: BenchmarkConfig,
    *,
    transport: Transport = _urllib_transport,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    """Run one cohort and return a stable, request-index-ordered receipt."""

    config.validate()
    prompts = build_prompts(config)
    start_barrier = threading.Barrier(config.concurrency + 1)
    headers = {"Content-Type": "application/json"}
    if config.api_key:
        headers["Authorization"] = f"Bearer {config.api_key}"
    encoded_prompts: dict[str, tuple[int, ...]] = {}
    if config.pretokenize:
        for prompt in dict.fromkeys(prompts):
            encoded_prompts[prompt] = _tokenize_prompt(
                config, prompt, transport, headers
            )
    url = config.endpoint.rstrip("/") + (
        "/v1/completions" if config.pretokenize else "/v1/chat/completions"
    )

    def issue(index: int) -> dict[str, Any]:
        prompt = prompts[index]
        token_ids = encoded_prompts.get(prompt)
        payload = _request_body(config, prompt, token_ids)
        start_barrier.wait()
        started = clock()
        try:
            response = transport(url, payload, headers, config.timeout_seconds)
            elapsed = clock() - started
            choices = response.body.get("choices")
            response_valid = isinstance(choices, list) and bool(choices)
            return {
                "request_index": index,
                "request_id": f"request-{index:02d}",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": len(token_ids) if token_ids is not None else None,
                "elapsed_seconds": round(elapsed, 6),
                "http_status": response.http_status,
                "response_valid": response_valid,
                "ok": 200 <= response.http_status < 300 and response_valid,
                "error_type": None,
            }
        except Exception as exc:  # receipt must retain every cohort member
            elapsed = clock() - started
            return {
                "request_index": index,
                "request_id": f"request-{index:02d}",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "prompt_tokens": len(token_ids) if token_ids is not None else None,
                "elapsed_seconds": round(elapsed, 6),
                "http_status": None,
                "response_valid": False,
                "ok": False,
                "error_type": type(exc).__name__,
            }

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=config.concurrency,
        thread_name_prefix="glm53-benchmark",
    ) as executor:
        futures = [executor.submit(issue, index) for index in range(config.concurrency)]
        start_barrier.wait()
        results = [future.result() for future in futures]

    results.sort(key=lambda result: int(result["request_index"]))
    receipt = {
        "schema": SCHEMA,
        "model": config.model,
        "scenario": config.scenario,
        "cache_state": config.cache_state,
        "input_mode": "pretokenized" if config.pretokenize else "chat",
        "concurrency": config.concurrency,
        "prefix_header": config.prefix_header,
        "prefix_repetitions": config.prefix_repetitions,
        "tail_repetitions": config.tail_repetitions,
        "max_tokens": config.max_tokens,
        "requests": results,
        "aggregate": _latency_summary(results),
    }
    receipt["validation_passed"] = receipt["aggregate"]["failed"] == 0
    return receipt


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a synchronized GLM-5.3 inference request cohort."
    )
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--concurrency", type=int, choices=CONCURRENCY_LEVELS, required=True
    )
    parser.add_argument("--scenario", choices=SCENARIOS, required=True)
    parser.add_argument("--cache-state", choices=CACHE_STATES, required=True)
    parser.add_argument(
        "--pretokenize",
        action="store_true",
        help="tokenize each unique chat prompt before starting timed completions",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--prefix-header", default="SparkCache CUDA 128K restore test.\n"
    )
    parser.add_argument("--prefix-repetitions", type=int, default=131_072)
    parser.add_argument("--tail-repetitions", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--api-key-env",
        default="OPENAI_API_KEY",
        help="environment variable containing an optional bearer token",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    config = BenchmarkConfig(
        endpoint=args.endpoint,
        model=args.model,
        concurrency=args.concurrency,
        scenario=args.scenario,
        cache_state=args.cache_state,
        pretokenize=args.pretokenize,
        prefix_header=args.prefix_header,
        prefix_repetitions=args.prefix_repetitions,
        tail_repetitions=args.tail_repetitions,
        max_tokens=args.max_tokens,
        timeout_seconds=args.timeout,
        api_key=os.environ.get(args.api_key_env),
    )
    receipt = run_benchmark(config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(receipt, indent=2) + "\n"
    args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if receipt["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
