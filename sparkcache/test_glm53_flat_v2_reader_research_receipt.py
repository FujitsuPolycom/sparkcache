from __future__ import annotations

import json
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    REPOSITORY_ROOT
    / "evidence"
    / "glm53-flash-dflash7-bf16"
    / "flat-v2-four-reader-semantic-rejection-eabe7fd.json"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_four_reader_receipt_cannot_be_interpreted_as_qualification() -> None:
    receipt = _receipt()

    assert receipt["schema"] == "sparkcache-glm53-flat-v2-reader-research/v1"
    assert receipt["status"] == "research-only"
    assert receipt["conclusion"] == "semantic-rejection"
    assert receipt["semantic_restore"]["passed"] is False
    assert receipt["admission"] == {
        "deployable": False,
        "qualified": False,
        "reason": (
            "A structurally verified persistent restore failed the exact semantic "
            "oracle."
        ),
    }


def test_four_reader_receipt_binds_structure_and_semantic_controls() -> None:
    receipt = _receipt()
    runtime = receipt["runtime"]
    stored = receipt["stored_context"]
    structural = receipt["structural_restore"]
    semantic = receipt["semantic_restore"]
    control = receipt["recomputation_control"]

    assert runtime["image_id"] == (
        "sha256:df4e09a32cdbf1c0e69cc7c4c9e95d890d6c7a1e3eaac84f969912a16fd27dd3"
    )
    assert runtime["sparkcache_commit"] == (
        "eabe7fd0c878db7384ef87fe80a1e96b9bedcf67"
    )
    assert stored == {
        "context_digest": (
            "b4161571df103395e2abae10372a90f35468561ec6c42bf4a7b7f0d0dfda5873"
        ),
        "prompt_sha256": (
            "965acd85cb28f804ab59cdc160688b04efaee14341e0bd27b647673e652ab812"
        ),
        "tokens": 131072,
        "encoded_bytes_per_rank": 813068464,
        "objects_per_rank": 13,
    }
    assert structural["ranks"] == [0, 1, 2, 3]
    assert structural["all_ranks_verified"] is True
    assert structural["cache_service_ms"] == {
        "minimum": 1231.7,
        "maximum": 1331.2,
    }
    assert semantic["expected"] == control["expected"] == "red"
    assert semantic["observed"] == "spark"
    assert control["observed"] == "red"
    assert control["passed"] is True
    assert control["elapsed_seconds"] == 55.14106


def test_dedicated_record_labels_four_reader_candidate_research_only() -> None:
    record = (
        REPOSITORY_ROOT
        / "GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md"
    ).read_text(encoding="utf-8")
    prose = " ".join(record.split())

    assert "Rejected four-reader flat-v2 artifact" in prose
    assert "rejected for deployment" in prose
    assert "does not replace the qualified single-reader artifact" in prose
    assert RECEIPT.name in record
