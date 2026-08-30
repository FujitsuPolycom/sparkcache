from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
CONTRACT = Path(__file__).with_name(
    "vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json"
)
PATCH_ROOT = ROOT / "patches" / "vllm-glm53-b12x-kda-adaptive-mtp"
PREIMAGES = PATCH_ROOT / "preimages.json"
SOURCE_RECEIPT = PATCH_ROOT / "source-receipt.json"
CONTAINERFILE = ROOT / "deploy/glm53_flash/Containerfile.b12x-kda-adaptive-mtp"
VLLM_COMMIT = "0b67266a0f37d6146a8403fb8482403c62f412d5"
SOURCE_ROLE = "source_built_glm53_b12x_kda_adaptive_mtp"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_glm53_b12x_kda_adaptive_mtp_receipt_pins_lf_source_lineage() -> None:
    receipt = json.loads(SOURCE_RECEIPT.read_text(encoding="utf-8"))
    assert receipt["source"] == {
        "repository": "https://github.com/local-inference-lab/vllm.git",
        "commit": VLLM_COMMIT,
        "tree": "ba9484ccb33aa56e90ff2f447f15ca9b9da97639",
        "line_endings": "LF",
    }
    assert receipt["adaptive_mtp_source_boundary"] == {
        "commit": "e10536aadf02a18fccddda7ec939c33147e8b0b3",
        "tree": "f7864d18865573dd162d3b850b4aa26acf320ab7",
    }
    assert receipt["included_commits_after_da4d7be"][-4:] == [
        "e10536aadf02a18fccddda7ec939c33147e8b0b3",
        *receipt["kda_live_tensor_commits"],
    ]
    assert len(receipt["included_commits_after_da4d7be"]) == 12

    for patch in receipt["patches"]:
        path = PATCH_ROOT / patch["path"]
        assert b"\r\n" not in path.read_bytes()
        assert _sha256(path) == patch["sha256"]

    assert _sha256(CONTRACT) == receipt["contract"]["sha256"]


def test_glm53_b12x_kda_adaptive_mtp_contract_attests_the_complete_sparkcache_vllm_surface() -> None:
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
        set(record["accepted_sha256"]) == {SOURCE_ROLE}
        for record in contract["files"]
    )
    assert all(record["required_symbols"] for record in contract["files"])


def test_glm53_b12x_kda_adaptive_mtp_overlay_has_exact_preimage_and_postimage_receipts() -> None:
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
        assert set(receipt["accepted_preimage_sha256"]) == {SOURCE_ROLE}
        assert set(receipt["accepted_postimage_sha256"]) == {SOURCE_ROLE}
        assert receipt["accepted_preimage_sha256"][SOURCE_ROLE] in recipe
        assert receipt["accepted_postimage_sha256"][SOURCE_ROLE] in recipe
        assert name in recipe
    assert CONTRACT.name in recipe
    assert "patches/vllm-glm53-b12x-kda-adaptive-mtp" in recipe


def test_glm53_b12x_kda_adaptive_mtp_patch_sequence_terminates_at_attested_contract_postimages() -> None:
    receipts = json.loads(PREIMAGES.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    accepted = {
        record["path"]: record["accepted_sha256"][SOURCE_ROLE]
        for record in contract["files"]
    }
    scheduler_recovery = receipts["030-sparkcache-hma-load-failure.patch"]
    scheduler_attach = receipts["041-sparkcache-shared-prefix-attach.patch"]
    manager_lease = receipts["040-sparkcache-shared-prefix-lease.patch"]

    assert scheduler_recovery["target_path"] == scheduler_attach["target_path"]
    assert (
        scheduler_recovery["accepted_postimage_sha256"][SOURCE_ROLE]
        == scheduler_attach["accepted_preimage_sha256"][SOURCE_ROLE]
    )
    assert (
        scheduler_attach["accepted_postimage_sha256"][SOURCE_ROLE]
        == accepted[scheduler_attach["target_path"]]
    )
    assert (
        manager_lease["accepted_postimage_sha256"][SOURCE_ROLE]
        == accepted[manager_lease["target_path"]]
    )


def test_glm53_b12x_kda_adaptive_mtp_overlay_preserves_verified_or_recompute_and_shared_leases() -> None:
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


def test_glm53_b12x_kda_adaptive_mtp_overlay_builds_the_native_cuda_placement_library() -> None:
    recipe = CONTAINERFILE.read_text(encoding="utf-8")
    assert "-DCMAKE_CUDA_ARCHITECTURES=121" in recipe
    assert "--target spark_cache_placement" in recipe
    assert "libspark_cache_placement.so.sha256" in recipe
