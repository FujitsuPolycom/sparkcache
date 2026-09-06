from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PATCH_ROOT = ROOT / "patches/vllm-pr646-815f839"
MANIFEST = PATCH_ROOT / "preimages.json"
CONTRACT = ROOT / (
    "sparkcache/runtime_patches/vllm-kv-block-lease-contract-pr646-815f839.json"
)
COMMIT = "815f839060c2781f6bcc47c0d584358b400ea0ea"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_pr646_runtime_patch_is_commit_bound_and_content_addressed() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    patch = PATCH_ROOT / manifest["patch"]

    assert manifest["schema"] == "sparkcache-vllm-runtime-patch/v1"
    assert manifest["vllm_commit"] == COMMIT
    assert manifest["patch_sha256"] == _sha256(patch)
    assert len(manifest["files"]) == 8
    assert all(
        record["preimage_sha256"] != record["postimage_sha256"]
        for record in manifest["files"].values()
    )


def test_pr646_contract_covers_recurrent_ownership_and_failure_recovery() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    symbols = {
        symbol for record in contract["files"] for symbol in record["required_symbols"]
    }

    assert contract["vllm_commit"] == COMMIT
    assert "KVConnectorBase_V1.get_recurrent_publication_boundaries" in symbols
    assert "KVCacheManager.take_recurrent_boundary_blocks" in symbols
    assert "Scheduler._recurrent_publication_boundary_at" in symbols
    assert "Scheduler._update_requests_with_invalid_blocks" in symbols
    assert "MambaHybridModelState.add_request" in symbols
    assert (
        "SingleTypeKVCacheManager.take_pending_aligned_recurrent_boundaries" in symbols
    )


def test_pr646_patch_preserves_native_geometry_and_adds_no_cache_identity() -> None:
    patch = (PATCH_ROOT / "010-sparkcache-runtime-contract.patch").read_text(
        encoding="utf-8"
    )

    assert "recurrent_publication_boundary" in patch
    assert "supports_recurrent_boundary_blocks" in patch
    assert "SparkContextCacheConnector" in patch
    assert "CacheIdentity" not in patch
    assert "prefix_match_unit" not in patch
    assert "hash_block_size =" not in patch
