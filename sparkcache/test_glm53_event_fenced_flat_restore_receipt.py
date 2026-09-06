from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.validate_event_fenced_flat_restore_receipt import validate_receipt


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RECEIPT = (
    REPOSITORY_ROOT
    / "evidence"
    / "glm53-flash-dflash7-bf16"
    / "event-fenced-flat-restore-861a965.json"
)


def _receipt() -> dict[str, object]:
    return json.loads(RECEIPT.read_text(encoding="utf-8"))


def test_event_fenced_flat_restore_receipt_validates_without_gpu() -> None:
    assert validate_receipt(_receipt()) == []


def test_event_fenced_flat_restore_receipt_cannot_claim_qualification() -> None:
    receipt = _receipt()
    receipt["status"] = "qualified"
    receipt["admission"]["qualified"] = True

    errors = validate_receipt(receipt)

    assert "status must be research-only" in errors
    assert "receipt cannot claim qualification" in errors


def test_event_fenced_flat_restore_receipt_detects_timing_drift() -> None:
    receipt = copy.deepcopy(_receipt())
    record = receipt["observations"][0]["rank_records"][0]
    record["payload"]["service_ms"] = 1.0

    errors = validate_receipt(receipt)

    assert any("canonical payload SHA-256 changed" in error for error in errors)
    assert any("timing disagree" in error for error in errors)


def test_event_fenced_flat_restore_receipt_detects_artifact_drift() -> None:
    receipt = copy.deepcopy(_receipt())
    receipt["captured_artifacts"]["image_receipt"]["sha256"] = "0" * 64

    errors = validate_receipt(receipt)

    assert "image_receipt identity changed" in errors


def test_event_fenced_flat_restore_receipt_discloses_log_retention_limit() -> None:
    receipt = _receipt()

    assert receipt["raw_log_retention"] == {
        "rank_log_files_retained": False,
        "structured_payloads_retained": True,
        "boundary_artifact_hashes_retained": True,
        "limitation": (
            "The receipt retains the exact structured timing payloads and "
            "capture-boundary hashes. It does not claim byte identity for "
            "complete container log windows because those windows were not "
            "retained as files."
        ),
    }
