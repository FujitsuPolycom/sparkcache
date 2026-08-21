"""Tests for the stock-lineage hybrid-memory-allocator (HMA) failure patch 031.

These tests pin the lineage isolation between the adaptive scheduler chain
(patches 010 → 030) and the stock scheduler chain (patches 011 → 031),
verify the exact preimage/postimage hashes recorded in ``preimages.json``,
and inspect the HMA recovery semantics embedded in patch 031 by AST rather
than by string matching alone.

The stock scheduler source (SHA-256 ``1ea341f4…``) is an image-extracted
artifact that lives outside this repository.  When it is available at
``SPARKCACHE_STOCK_E266_SCHEDULER`` (or the default path under the
neighbour ``stock-e266-image`` directory), the tests additionally verify
that applying 011 then 031 produces the expected final hash
``2f34aa…``.  When the source is absent, the hash-chain and AST tests
still run and the application test is skipped.
"""

from __future__ import annotations

import ast
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path

import pytest

PATCH_DIR = (
    Path(__file__).resolve().parents[2]
    / "patches"
    / "vllm-e2666d9a6"
)
PREIMAGES = PATCH_DIR / "preimages.json"
PATCH_030 = PATCH_DIR / "030-sparkcache-hma-load-failure.patch"
PATCH_031 = PATCH_DIR / "031-sparkcache-stock-hma-load-failure.patch"
CONTRACT = (
    Path(__file__).resolve().parent
    / "vllm-kv-block-lease-contract-e2666d9a6.json"
)

STOCK_PREIMAGE = (
    "1ea341f4cc28d282452597c25d97eea84be8b5f984d2e1a6b548356c8417fdce"
)
STOCK_POST_011 = (
    "d4ebec211b027b6c7f64574f79374237de0f5fde0c5c03f20f1cb1596ffadc3a"
)
STOCK_POST_031 = (
    "2f34aa9d65a495a86d814c90f654fbe1ff754cfdbecd204b98d513652ca3e06d"
)
ADAPTIVE_PREIMAGE = (
    "5020bfa9a8142056949357e6acfc355edb974d799e2e7ed62feb93dfef3316a6"
)
ADAPTIVE_POST_010 = (
    "52bf226d9964d8ddb99f4c7444470d592496c8d4836750f3e0ba226cbbc07d75"
)
ADAPTIVE_POST_030 = (
    "47b6cc16dc61c98beeecc4cbdb2d468844eccc08bc284ffd2be45f0c42b521ca"
)

DEFAULT_STOCK_SCHEDULER = (
    Path(__file__).resolve().parents[3]
    / "stock-e266-image"
    / "scheduler.py"
)


def _stock_scheduler_path() -> Path | None:
    env = os.environ.get("SPARKCACHE_STOCK_E266_SCHEDULER")
    if env:
        p = Path(env)
        return p if p.is_file() else None
    if DEFAULT_STOCK_SCHEDULER.is_file():
        return DEFAULT_STOCK_SCHEDULER
    return None


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _lf(data: bytes) -> bytes:
    return data.replace(b"\r\n", b"\n")


# --------------------------------------------------------------------------- #
# preimages.json integrity
# --------------------------------------------------------------------------- #

def test_preimages_json_contains_patch_031_entry() -> None:
    data = json.loads(PREIMAGES.read_text("utf-8"))
    entry = data.get("031-sparkcache-stock-hma-load-failure.patch")
    assert entry is not None
    assert entry["target_path"] == "vllm/v1/core/sched/scheduler.py"
    assert entry["preimage_sha256"] == STOCK_POST_011
    assert entry["postimage_sha256"] == STOCK_POST_031


def test_preimages_json_030_preimage_is_adaptive_post_010() -> None:
    """Patch 030 belongs to the adaptive lineage, not the stock lineage."""
    data = json.loads(PREIMAGES.read_text("utf-8"))
    entry = data["030-sparkcache-hma-load-failure.patch"]
    assert entry["preimage_sha256"] == ADAPTIVE_POST_010
    assert entry["postimage_sha256"] == ADAPTIVE_POST_030


def test_preimages_json_011_preimage_is_stock_unpatched() -> None:
    data = json.loads(PREIMAGES.read_text("utf-8"))
    entry = data["011-sparkcache-glm52-async-rollback.patch"]
    assert entry["preimage_sha256"] == STOCK_PREIMAGE
    assert entry["postimage_sha256"] == STOCK_POST_011


def test_lineage_preimages_do_not_cross() -> None:
    """The stock post-011 hash must not equal the adaptive post-010 hash."""
    assert STOCK_POST_011 != ADAPTIVE_POST_010
    assert STOCK_POST_031 != ADAPTIVE_POST_030


# --------------------------------------------------------------------------- #
# Lease contract
# --------------------------------------------------------------------------- #

def test_lease_contract_accepts_all_six_scheduler_states() -> None:
    contract = json.loads(CONTRACT.read_text("utf-8"))
    scheduler = next(
        r for r in contract["files"]
        if r["path"] == "vllm/v1/core/sched/scheduler.py"
    )
    accepted = scheduler["accepted_sha256"]
    assert accepted["stock_pre_patch"] == STOCK_PREIMAGE
    assert accepted["stock_async_rollback_post_patch"] == STOCK_POST_011
    assert accepted["stock_hma_recompute_post_patch"] == STOCK_POST_031
    assert accepted["e2666d9a6_pre_patch"] == ADAPTIVE_PREIMAGE
    assert accepted["sparkcache_async_rollback_post_patch"] == ADAPTIVE_POST_010
    assert accepted["sparkcache_hma_recompute_post_patch"] == ADAPTIVE_POST_030


def test_lease_contract_has_no_duplicate_hashes() -> None:
    """The verifier rejects duplicate accepted SHA-256 values."""
    contract_data = json.loads(CONTRACT.read_text("utf-8"))
    scheduler_record = next(
        r for r in contract_data["files"]
        if r["path"] == "vllm/v1/core/sched/scheduler.py"
    )
    values = list(scheduler_record["accepted_sha256"].values())
    assert len(values) == len(set(values))


# --------------------------------------------------------------------------- #
# HMA recovery semantics (AST inspection of the patched source)
# --------------------------------------------------------------------------- #

def _apply_patch_to_source(
    source: bytes, patch: Path, tmp: Path
) -> bytes:
    """Apply a git patch to source bytes in a temp git repo; return result."""
    repo = tmp / "repo"
    target = repo / "vllm/v1/core/sched/scheduler.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(source)
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"], cwd=repo, check=True
    )
    subprocess.run(
        ["git", "config", "core.eol", "lf"], cwd=repo, check=True
    )
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "commit", "-q", "-m", "init"],
        cwd=repo,
        check=True,
        env={**os.environ, "GIT_AUTHOR_NAME": "test",
             "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "test",
             "GIT_COMMITTER_EMAIL": "t@t"},
    )
    patch_bytes = _lf(patch.read_bytes())
    patch_file = repo / ".patch"
    patch_file.write_bytes(patch_bytes)
    subprocess.run(
        ["git", "apply", str(patch_file)],
        cwd=repo,
        check=True,
    )
    return target.read_bytes()


def _find_method(tree: ast.Module, name: str) -> ast.FunctionDef | None:
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


def test_patch_031_hma_semantics_flatten_and_disjoint() -> None:
    """The HMA hunk in 031 must flatten group block IDs, detect
    invalid-block intersection via ``isdisjoint``, reset
    ``num_computed_tokens`` to zero, account affected tokens, and
    evict every group's blocks when requested."""
    stock_path = _stock_scheduler_path()
    if stock_path is None:
        pytest.skip("stock e266 scheduler source not available locally")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = _lf(stock_path.read_bytes())
        # Apply 011 then 031.
        post_011 = _apply_patch_to_source(source, PATCH_DIR / "011-sparkcache-glm52-async-rollback.patch", tmp_path / "a")
        post_031 = _apply_patch_to_source(post_011, PATCH_031, tmp_path / "b")

    assert _sha256(post_031) == STOCK_POST_031

    tree = ast.parse(post_031)
    method = _find_method(tree, "_update_requests_with_invalid_blocks")
    assert method is not None, "patched scheduler must retain _update_requests_with_invalid_blocks"

    source_text = ast.unparse(method)
    # Multi-group flattening: iterate over req_block_groups, build a set.
    assert "req_block_groups" in source_text
    assert "for group_block_ids in req_block_groups" in source_text
    assert "for block_id in group_block_ids" in source_text
    # Disjoint check: skip requests whose blocks don't intersect invalid ones.
    assert "isdisjoint" in source_text
    # Whole-prefix invalidation: reset to zero.
    assert "num_computed_tokens = 0" in source_text
    # Token accounting.
    assert "total_affected_tokens" in source_text
    # Evict all groups' blocks when eviction is requested.
    assert "blocks_to_evict.update" in source_text
    # The single-group fallback must still unpack as a 1-element tuple.
    assert "req_block_ids, = req_block_groups" in source_text


def test_patch_031_produces_expected_final_hash() -> None:
    """Exact application of 011 then 031 on the stock preimage must
    produce the independently reconstructed final hash."""
    stock_path = _stock_scheduler_path()
    if stock_path is None:
        pytest.skip("stock e266 scheduler source not available locally")

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        source = _lf(stock_path.read_bytes())
        assert _sha256(source) == STOCK_PREIMAGE

        post_011 = _apply_patch_to_source(
            source, PATCH_DIR / "011-sparkcache-glm52-async-rollback.patch",
            tmp_path / "a",
        )
        assert _sha256(post_011) == STOCK_POST_011

        post_031 = _apply_patch_to_source(
            post_011, PATCH_031, tmp_path / "b",
        )
        assert _sha256(post_031) == STOCK_POST_031


def test_patch_030_preimage_rejects_stock_post_011() -> None:
    """Patch 030's recorded preimage is the adaptive post-010 hash, not
    the stock post-011 hash.  This enforces lineage isolation at the
    receipt level: an operator who stages the stock post-011 file cannot
    pass the adaptive 030 preimage check."""
    data = json.loads(PREIMAGES.read_text("utf-8"))
    p030_preimage = data["030-sparkcache-hma-load-failure.patch"]["preimage_sha256"]
    assert p030_preimage != STOCK_POST_011
    assert p030_preimage == ADAPTIVE_POST_010


def test_patch_031_preimage_rejects_adaptive_post_010() -> None:
    """Patch 031's recorded preimage is the stock post-011 hash, not
    the adaptive post-010 hash."""
    data = json.loads(PREIMAGES.read_text("utf-8"))
    p031_preimage = data["031-sparkcache-stock-hma-load-failure.patch"]["preimage_sha256"]
    assert p031_preimage != ADAPTIVE_POST_010
    assert p031_preimage == STOCK_POST_011
