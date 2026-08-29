from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from deploy.glm53_flash.concurrency_benchmark import (
    BenchmarkConfig,
    TransportResult,
    build_prompts,
    run_benchmark,
)


def _config(**overrides: Any) -> BenchmarkConfig:
    values: dict[str, Any] = {
        "endpoint": "http://127.0.0.1:8000",
        "model": "glm53-test",
        "concurrency": 2,
        "scenario": "identical-prefix",
        "cache_state": "hot",
        "prefix_repetitions": 4,
        "tail_repetitions": 2,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def test_prompt_shapes_are_stable_and_distinguish_scenarios() -> None:
    identical = build_prompts(_config())
    shared = build_prompts(_config(scenario="shared-trunk"))

    assert identical[0] == identical[1]
    assert shared[0] != shared[1]
    assert all(prompt.startswith("benchmark " * 4) for prompt in shared)
    assert "tail-00 tail-00" in shared[0]
    assert "tail-01 tail-01" in shared[1]


@pytest.mark.parametrize("concurrency", [2, 8, 16])
def test_all_supported_cohorts_start_together_and_receipt_is_ordered(
    concurrency: int,
) -> None:
    arrived = 0
    arrived_lock = threading.Lock()
    all_arrived = threading.Event()

    def transport(
        url: str, payload: bytes, headers: dict[str, str], timeout: float
    ) -> TransportResult:
        nonlocal arrived
        assert url.endswith("/v1/chat/completions")
        assert headers["Content-Type"] == "application/json"
        assert timeout == 600.0
        body = json.loads(payload)
        assert body["stream"] is False
        with arrived_lock:
            arrived += 1
            if arrived == concurrency:
                all_arrived.set()
        assert all_arrived.wait(2.0), "cohort requests were not concurrent"
        return TransportResult(200, {"choices": [{"message": {"content": "ok"}}]})

    receipt = run_benchmark(_config(concurrency=concurrency), transport=transport)

    assert [item["request_index"] for item in receipt["requests"]] == list(
        range(concurrency)
    )
    assert receipt["aggregate"]["request_count"] == concurrency
    assert receipt["aggregate"]["succeeded"] == concurrency
    assert receipt["validation_passed"] is True


def test_latency_aggregate_uses_nearest_rank_and_records_failures() -> None:
    clock_calls = threading.local()

    def clock() -> float:
        calls = getattr(clock_calls, "calls", 0)
        clock_calls.calls = calls + 1
        if calls == 0:
            return 0.0
        return 0.1 if threading.current_thread().name.endswith("_0") else 0.4

    def transport(
        url: str, payload: bytes, headers: dict[str, str], timeout: float
    ) -> TransportResult:
        if threading.current_thread().name.endswith("_0"):
            return TransportResult(200, {"choices": [{}]})
        return TransportResult(503, {"error": "busy"})

    receipt = run_benchmark(_config(), transport=transport, clock=clock)

    assert receipt["aggregate"] == {
        "request_count": 2,
        "succeeded": 1,
        "failed": 1,
        "min_seconds": 0.1,
        "p50_seconds": 0.1,
        "p95_seconds": 0.4,
        "max_seconds": 0.4,
    }
    assert receipt["validation_passed"] is False


def test_real_http_transport_posts_openai_compatible_requests() -> None:
    payloads: list[dict[str, Any]] = []
    both_arrived = threading.Event()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            length = int(self.headers["Content-Length"])
            payload = json.loads(self.rfile.read(length))
            with lock:
                payloads.append(payload)
                if len(payloads) == 2:
                    both_arrived.set()
            if not both_arrived.wait(2.0):
                self.send_error(500)
                return
            encoded = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, format: str, *args: object) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever)
    server_thread.start()
    try:
        endpoint = f"http://127.0.0.1:{server.server_address[1]}"
        receipt = run_benchmark(_config(endpoint=endpoint))
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join()

    assert receipt["validation_passed"] is True
    assert len(payloads) == 2
    assert all(payload["model"] == "glm53-test" for payload in payloads)
