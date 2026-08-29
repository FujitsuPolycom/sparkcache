from __future__ import annotations

import json
from pathlib import Path

from sparkcache.runtime_patches.verify_lease_contract import ContractError


CONTRACT = Path(__file__).with_name("vllm-kv-block-lease-contract-da4d7be.json")
ROOT = Path(__file__).parents[2]
PATCH_NAME = "030-sparkcache-hma-load-failure.patch"
PATCH = ROOT / "patches" / "vllm-da4d7be" / PATCH_NAME
PREIMAGES = PATCH.with_name("preimages.json")


def test_glm53_contract_is_full_commit_bound_and_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["vllm_commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert {record["path"] for record in contract["files"]} == {
        "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
        "vllm/distributed/kv_transfer/kv_connector/utils.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/sched/output.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "vllm/v1/kv_cache_interface.py",
    }
    assert all(
        set(record["accepted_sha256"]) == {
            "source_checkout",
            "jovian_glm53_runtime",
        }
        for record in contract["files"]
    )
    assert all(
        len(digest) == 64
        for record in contract["files"]
        for digest in record["accepted_sha256"].values()
    )
    assert all(record["required_symbols"] for record in contract["files"])


def test_contract_error_remains_public_for_verified_only_launchers() -> None:
    assert issubclass(ContractError, RuntimeError)


def test_glm53_image_applies_exact_hma_load_failure_recovery_patch() -> None:
    receipts = json.loads(PREIMAGES.read_text(encoding="utf-8"))
    receipt = receipts[PATCH_NAME]
    assert receipt["target_path"] == "vllm/v1/core/sched/scheduler.py"
    assert set(receipt["accepted_preimage_sha256"]) == {
        "source_checkout",
        "jovian_glm53_runtime",
    }
    assert set(receipt["accepted_postimage_sha256"]) == {
        "source_checkout",
        "jovian_glm53_runtime",
    }

    patch = PATCH.read_text(encoding="utf-8")
    assert "req_block_groups = self.kv_cache_manager.get_block_ids(req_id)" in patch
    assert "for group_block_ids in req_block_groups" in patch
    assert "request_block_ids.isdisjoint(invalid_block_ids)" in patch
    assert "request.num_computed_tokens = 0" in patch
    assert "blocks_to_evict.update(request_block_ids)" in patch

    recipe = (ROOT / "deploy/glm53_flash/Containerfile").read_text("utf-8")
    assert PATCH_NAME in recipe
    assert receipt["accepted_preimage_sha256"]["jovian_glm53_runtime"] in recipe
    assert receipt["accepted_postimage_sha256"]["jovian_glm53_runtime"] in recipe
