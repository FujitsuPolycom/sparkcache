from __future__ import annotations

from tools import validate_async_store_completion as probe


def test_request_body_is_nonce_fronted_and_disables_thinking() -> None:
    body = probe.request_body("model", "nonce", 4)

    assert body["messages"][0]["content"].startswith("nonce ")
    assert body["chat_template_kwargs"] == {"enable_thinking": False}


def test_idle_requires_every_ownership_signal_to_clear() -> None:
    values = {
        "running": 0.0,
        "waiting": 0.0,
        "kv_usage": 0.0,
        "delayed_requests": 0.0,
        "delayed_rank_slots": 0.0,
        "retained_pages": 0.0,
        "uncertain_ranks": 0.0,
    }

    assert probe.idle(values)
    assert not probe.idle({**values, "retained_pages": 1.0})
