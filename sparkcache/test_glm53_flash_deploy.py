from __future__ import annotations

import json
from pathlib import Path

import pytest

from deploy.glm53_flash.profile import (
    ProfileError,
    build_kv_transfer_config,
    compact_json,
    immutable_revision_identity,
)


ROOT = Path(__file__).parents[1]
TARGET_ID = "a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9"
DRAFT_ID = "b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b"


def test_target_revision_identity_is_deterministic() -> None:
    assert immutable_revision_identity(
        "local-inference-lab/GLM-5.3-Flash-NVFP4",
        "520de24eabf507659eaef7c70f14fd584527facc",
    ) == TARGET_ID


def test_connector_config_binds_target_and_draft_without_optional_native_paths() -> None:
    config = build_kv_transfer_config(
        target_checkpoint_sha256=TARGET_ID,
        draft_checkpoint_sha256=DRAFT_ID,
    )
    assert config["kv_load_failure_policy"] == "recompute"
    assert config["kv_role"] == "kv_both"
    extra = config["kv_connector_extra_config"]
    assert extra["spark_cache_model_profile"] == "glm53-flash-hybrid"
    assert extra["spark_cache_target_checkpoint_sha256"] == TARGET_ID
    assert extra["spark_cache_draft_checkpoint_sha256"] == DRAFT_ID
    assert extra["spark_cache_draft_policy"] == "separate"
    assert extra["spark_cache_streaming_snapshots"] is False
    assert extra["spark_cache_native_restore"] is False
    assert json.loads(compact_json(config)) == config


@pytest.mark.parametrize(
    ("field", "value"),
    (("target_checkpoint_sha256", "short"), ("draft_checkpoint_sha256", "A" * 64)),
)
def test_connector_config_rejects_unproven_checkpoint_identity(
    field: str, value: str
) -> None:
    arguments = {
        "target_checkpoint_sha256": TARGET_ID,
        "draft_checkpoint_sha256": DRAFT_ID,
    }
    arguments[field] = value
    with pytest.raises(ProfileError, match="64-character lowercase SHA-256"):
        build_kv_transfer_config(**arguments)


def test_image_recipe_pins_runtime_hashes_and_source_verifier() -> None:
    recipe = (ROOT / "deploy/glm53_flash/Containerfile").read_text("utf-8")
    receipts = json.loads(
        (ROOT / "patches/vllm-da4d7be/preimages.json").read_text("utf-8")
    )["020-sparkcache-vmm-exemption.patch"]
    assert receipts["accepted_preimage_sha256"]["jovian_glm53_runtime"] in recipe
    assert receipts["accepted_postimage_sha256"]["jovian_glm53_runtime"] in recipe
    assert "vllm-kv-block-lease-contract-da4d7be.json" in recipe
    assert 'test "$actual_source" = "$SPARKCACHE_SOURCE_SHA256"' in recipe
