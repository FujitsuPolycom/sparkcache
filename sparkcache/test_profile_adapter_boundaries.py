"""Regression tests for the boundary between generic code and model profiles."""

from __future__ import annotations

from pathlib import Path


def test_generic_streaming_modules_do_not_embed_model_inventory() -> None:
    root = Path(__file__).resolve().parent
    generic_sources = (
        root / "streaming" / "factory.py",
        root / "streaming" / "publisher.py",
    )
    forbidden = (
        "glm52",
        "glm-5.2",
        "target_layers = 79",
        "indexer_layers = 22",
        "target_bytes_per_token = 368",
        "indexer_bytes_per_token = 132",
    )
    for path in generic_sources:
        text = path.read_text(encoding="utf-8").casefold()
        for fragment in forbidden:
            assert fragment not in text, f"{path.name} embeds {fragment!r}"


def test_model_inventory_is_owned_by_the_profile_adapter() -> None:
    root = Path(__file__).resolve().parent
    adapter = root / "profile_adapters" / "glm52_streaming.py"
    text = adapter.read_text(encoding="utf-8")
    assert "TARGET_LAYERS = 79" in text
    assert "INDEXER_LAYERS = 22" in text
    assert "TARGET_BYTES_PER_TOKEN = 368" in text
    assert "INDEXER_BYTES_PER_TOKEN = 132" in text
