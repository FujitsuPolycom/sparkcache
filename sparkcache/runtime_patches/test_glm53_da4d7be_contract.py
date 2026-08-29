from __future__ import annotations

import json
from pathlib import Path

from sparkcache.runtime_patches.verify_lease_contract import ContractError


CONTRACT = Path(__file__).with_name("vllm-kv-block-lease-contract-da4d7be.json")


def test_glm53_contract_is_full_commit_bound_and_complete() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["vllm_commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert {record["path"] for record in contract["files"]} == {
        "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
        "vllm/distributed/kv_transfer/kv_connector/utils.py",
        "vllm/v1/core/sched/scheduler.py",
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


def test_contract_error_remains_public_for_fail_closed_launchers() -> None:
    assert issubclass(ContractError, RuntimeError)
