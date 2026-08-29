from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONTRACT = Path(__file__).with_name("vllm-kv-block-lease-contract-e10536a.json")
PATCH_ROOT = ROOT / "patches" / "vllm-e10536a"
PREIMAGES = PATCH_ROOT / "preimages.json"
CONTAINERFILE = ROOT / "deploy/glm53_flash/Containerfile.e10536a"
VLLM_COMMIT = "e10536aadf02a18fccddda7ec939c33147e8b0b3"


def test_e105_contract_attests_the_complete_sparkcache_vllm_surface() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    assert contract["vllm_commit"] == VLLM_COMMIT
    assert {record["path"] for record in contract["files"]} == {
        "vllm/distributed/kv_transfer/kv_connector/v1/base.py",
        "vllm/distributed/kv_transfer/kv_connector/utils.py",
        "vllm/v1/core/sched/scheduler.py",
        "vllm/v1/core/kv_cache_manager.py",
        "vllm/v1/core/block_pool.py",
        "vllm/v1/core/kv_cache_coordinator.py",
        "vllm/v1/core/sched/output.py",
        "vllm/v1/worker/gpu_model_runner.py",
        "vllm/v1/core/single_type_kv_cache_manager.py",
        "vllm/v1/kv_cache_interface.py",
    }
    assert all(
        set(record["accepted_sha256"]) == {"source_built_e105"}
        for record in contract["files"]
    )
    assert all(record["required_symbols"] for record in contract["files"])


def test_e105_overlay_has_exact_preimage_and_postimage_receipts() -> None:
    receipts = json.loads(PREIMAGES.read_text(encoding="utf-8"))
    recipe = CONTAINERFILE.read_text(encoding="utf-8")
    assert set(receipts) == {
        "020-sparkcache-vmm-exemption.patch",
        "030-sparkcache-hma-load-failure.patch",
        "040-sparkcache-shared-prefix-lease.patch",
        "041-sparkcache-shared-prefix-attach.patch",
    }
    for name, receipt in receipts.items():
        assert (PATCH_ROOT / name).is_file()
        assert set(receipt["accepted_preimage_sha256"]) == {"source_built_e105"}
        assert set(receipt["accepted_postimage_sha256"]) == {"source_built_e105"}
        assert receipt["accepted_preimage_sha256"]["source_built_e105"] in recipe
        assert receipt["accepted_postimage_sha256"]["source_built_e105"] in recipe
        assert name in recipe
    assert CONTRACT.name in recipe
    assert "patches/vllm-e10536a" in recipe


def test_e105_patch_sequence_terminates_at_attested_contract_postimages() -> None:
    receipts = json.loads(PREIMAGES.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    accepted = {
        record["path"]: record["accepted_sha256"]["source_built_e105"]
        for record in contract["files"]
    }
    scheduler_recovery = receipts["030-sparkcache-hma-load-failure.patch"]
    scheduler_attach = receipts["041-sparkcache-shared-prefix-attach.patch"]
    manager_lease = receipts["040-sparkcache-shared-prefix-lease.patch"]

    assert scheduler_recovery["target_path"] == scheduler_attach["target_path"]
    assert (
        scheduler_recovery["accepted_postimage_sha256"]["source_built_e105"]
        == scheduler_attach["accepted_preimage_sha256"]["source_built_e105"]
    )
    assert (
        scheduler_attach["accepted_postimage_sha256"]["source_built_e105"]
        == accepted[scheduler_attach["target_path"]]
    )
    assert (
        manager_lease["accepted_postimage_sha256"]["source_built_e105"]
        == accepted[manager_lease["target_path"]]
    )


def test_e105_overlay_preserves_verified_or_recompute_and_shared_leases() -> None:
    recovery = (PATCH_ROOT / "030-sparkcache-hma-load-failure.patch").read_text(
        encoding="utf-8"
    )
    lease = (PATCH_ROOT / "040-sparkcache-shared-prefix-lease.patch").read_text(
        encoding="utf-8"
    )
    attach = (PATCH_ROOT / "041-sparkcache-shared-prefix-attach.patch").read_text(
        encoding="utf-8"
    )
    assert "request.num_computed_tokens = 0" in recovery
    assert "publish_shared_prefix_lease" in lease
    assert "mark_shared_prefix_lease_ready" in lease
    assert "attach_shared_prefix_lease" in attach
    assert "num_computed_tokens == 0" in attach


def test_e105_overlay_builds_the_native_cuda_placement_library() -> None:
    recipe = CONTAINERFILE.read_text(encoding="utf-8")
    assert "-DCMAKE_CUDA_ARCHITECTURES=121" in recipe
    assert "--target spark_cache_placement" in recipe
    assert "libspark_cache_placement.so.sha256" in recipe
