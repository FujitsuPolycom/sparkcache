#!/usr/bin/env python3
"""Exercise busy-saver publication and require delayed KV ownership to drain."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import time
import urllib.request
import uuid
from pathlib import Path


def request_body(model: str, nonce: str, repetitions: int) -> dict[str, object]:
    return {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": f"{nonce} " + "cache " * repetitions + "Reply OK.",
            }
        ],
        "max_tokens": 1,
        "temperature": 0,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def _metric(text: str, name: str) -> float:
    matches = re.findall(
        rf"(?m)^{re.escape(name)}\{{[^\n]*\}}\s+([0-9.eE+-]+)$", text
    )
    if not matches:
        raise RuntimeError(f"metrics response does not contain {name}")
    return sum(float(value) for value in matches)


def parse_metrics(text: str) -> dict[str, float]:
    names = {
        "running": "vllm:num_requests_running",
        "waiting": "vllm:num_requests_waiting",
        "kv_usage": "vllm:kv_cache_usage_perc",
        "delayed_requests": "vllm:sparkcache_capture_delayed_requests",
        "delayed_rank_slots": "vllm:sparkcache_capture_delayed_rank_slots",
        "retained_pages": "vllm:sparkcache_capture_retained_manager_pages",
        "uncertain_ranks": "vllm:sparkcache_capture_ownership_uncertain_ranks",
    }
    return {key: _metric(text, name) for key, name in names.items()}


class Client:
    def __init__(self, endpoint: str, model: str) -> None:
        self.endpoint = endpoint.rstrip("/")
        self.model = model

    def chat(self, nonce: str, repetitions: int, timeout: float) -> float:
        request = urllib.request.Request(
            self.endpoint + "/v1/chat/completions",
            data=json.dumps(request_body(self.model, nonce, repetitions)).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.load(response)
        if not result.get("choices"):
            raise RuntimeError("validation request returned no completion")
        return time.monotonic() - started

    def metrics(self) -> dict[str, float]:
        with urllib.request.urlopen(self.endpoint + "/metrics", timeout=10) as response:
            return parse_metrics(response.read().decode())


def idle(metrics: dict[str, float]) -> bool:
    return all(
        metrics[name] == 0
        for name in (
            "running",
            "waiting",
            "kv_usage",
            "delayed_requests",
            "delayed_rank_slots",
            "retained_pages",
            "uncertain_ranks",
        )
    )


def run(args: argparse.Namespace) -> dict[str, object]:
    client = Client(args.endpoint, args.model)
    receipts = []
    for cycle in range(1, args.cycles + 1):
        nonce = uuid.uuid4().hex
        large = client.chat(f"async-store-large-{nonce}", args.large_repetitions, args.timeout)

        def small(index: int) -> float:
            return client.chat(
                f"async-store-small-{nonce}-{index}",
                args.small_repetitions,
                args.timeout,
            )

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=args.concurrency
        ) as executor:
            small_seconds = list(executor.map(small, range(args.concurrency)))
        deadline = time.monotonic() + args.drain_timeout
        while True:
            observed = client.metrics()
            if idle(observed):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "asynchronous store ownership did not drain: "
                    + json.dumps(observed, sort_keys=True)
                )
            time.sleep(0.5)
        receipts.append(
            {
                "cycle": cycle,
                "large_seconds": round(large, 3),
                "small_seconds": [round(value, 3) for value in small_seconds],
                "idle": observed,
            }
        )
    return {
        "schema": "sparkcache-async-store-completion-probe/v1",
        "status": "passed",
        "endpoint": args.endpoint,
        "model": args.model,
        "cycles": receipts,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", default="glm-5.3-flash")
    parser.add_argument("--cycles", type=int, default=3)
    parser.add_argument("--large-repetitions", type=int, default=131000)
    parser.add_argument("--small-repetitions", type=int, default=4096)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--drain-timeout", type=float, default=45)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
