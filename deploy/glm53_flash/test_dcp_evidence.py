from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "evidence" / "glm53-dcp" / "tp4-dcp2-dcp4-sparkcache.json"
DOCUMENT = ROOT / "deploy" / "glm53_flash" / "GLM53_DCP2_DCP4_SPARKCACHE_VALIDATION.md"


def test_glm53_dcp_evidence_is_bounded_and_exact() -> None:
    data = json.loads(EVIDENCE.read_text(encoding="utf-8"))

    assert data["schema"] == "sparkcache-glm53-dcp-validation/v1"
    assert data["evidence_status"] == "research-only"
    assert data["qualification"] is False
    assert data["runtime"]["sparkcache_deployable_source_sha256"] == (
        "40de372dda64dd25f493584b2ba3dae81c4350d424d3cf00cfea92452dac170c"
    )
    assert data["runtime"]["sparkcache_behavior_has_uncommitted_changes"] is True
    assert data["serving"] == {
        "max_model_len": 524288,
        "max_num_batched_tokens": 8192,
        "speculative_method": "dflash",
        "speculative_tokens": 7,
        "cp_kv_cache_interleave_size": 4,
        "dcp_ckv_gather": True,
        "publication_schema": "snapshot-v1",
        "manager_page_namespace": "manager-pages-v2",
    }

    dcp2 = data["accepted_runs"]["dcp2"]
    assert dcp2["publication"]["span_tokens"] == 9216
    assert dcp2["python_restore"]["semantic_oracle"] == "exact red"
    assert dcp2["cuda_restore"]["semantic_oracle"] == "exact red"
    assert dcp2["python_restore"]["page_bytes_per_physical_rank"] == 78751393

    dcp4 = data["accepted_runs"]["dcp4"]
    assert dcp4["publication"]["span_tokens"] == 8192
    assert dcp4["cuda_restore"]["semantic_oracle"] == "exact red"
    assert dcp4["cuda_restore"]["page_bytes_per_physical_rank"] == 62953633

    rejected = data["rejected_diagnostics"]
    assert len(rejected) == 1
    assert "rejected" in rejected[0]["semantic_result"]
    assert "large-context restore" in data["not_established"]
    assert "concurrent restore" in data["not_established"]
    assert "page-delta or tail-only publication under DCP" in data["not_established"]


def test_glm53_dcp_document_routes_to_machine_evidence() -> None:
    text = DOCUMENT.read_text(encoding="utf-8")

    assert "**Status: research-only evidence.**" in text
    assert "tp4-dcp2-dcp4-sparkcache.json" in text
    assert "does not qualify" in text
    assert "Exact `red`" in text
    assert "MambaSpec" in text
