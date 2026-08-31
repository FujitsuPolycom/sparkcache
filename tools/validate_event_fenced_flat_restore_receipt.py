#!/usr/bin/env python3
"""Validate the retained event-fenced GLM-5.3 flat-restore research receipt."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RECEIPT = (
    REPOSITORY_ROOT
    / "evidence"
    / "glm53-flash-dflash7-bf16"
    / "event-fenced-flat-restore-861a965.json"
)
SHA256 = re.compile(r"^[0-9a-f]{64}$")
PHASES = {
    "manifest_lookup",
    "prior_cuda_work",
    "restore_read",
    "reassembly_decode",
    "h2d_submit",
    "cuda_sync",
}
EXPECTED_OBSERVATIONS = {
    "stored-prefix-replay": {
        "captured_after_utc": "2026-08-30T20:53:12.5280514Z",
        "request_id": "cmpl-9726270f6854a535-0-9c879921",
        "context_digest": (
            "b4161571df103395e2abae10372a90f35468561ec6c42bf4a7b7f0d0dfda5873"
        ),
        "prompt_sha256": (
            "965acd85cb28f804ab59cdc160688b04efaee14341e0bd27b647673e652ab812"
        ),
        "request_receipt_sha256": (
            "1a76f8daec8e020b98099bdf326dbd9fec386304ea1314254a08c5f0e5b249b5"
        ),
        "response_sha256": (
            "2c68d02422a6c4bdb42bd10221940894e746342bef6a56695fdbcb549074a355"
        ),
        "elapsed_seconds": 1.613022,
    },
    "distinct-stored-prefix-replay": {
        "captured_after_utc": "2026-08-30T20:53:52.9347373Z",
        "request_id": "cmpl-a0fbe1e38fe03ada-0-a3fc86ee",
        "context_digest": (
            "6d03abbdf6e6463f2da029f13b9db43e04bda238082dafa83a413da715443ab0"
        ),
        "prompt_sha256": (
            "4bb683a895caaaacb783294e65cb9c4b59c808c1e7b563a48edb6cd52b302dfe"
        ),
        "request_receipt_sha256": (
            "c9e21368b11658082535a3ca25cbca1b4a93981684688a4274545168baa59b08"
        ),
        "response_sha256": (
            "2c68d02422a6c4bdb42bd10221940894e746342bef6a56695fdbcb549074a355"
        ),
        "elapsed_seconds": 1.590435,
    },
}
EXPECTED_ARTIFACTS = {
    "image_receipt": (
        9238,
        "ac0959abee37bc7eab93de03d3fd0298fa8c901653b07292de81d146039cde70",
    ),
    "archive_receipt": (
        925,
        "9a7d243d349fa073234021ab2cd326fd8f2a42add2d8a16b2e2ae840af13b348",
    ),
    "fanout_receipt": (
        3093,
        "88579a0d9ca8c11447ccff2a3fc019ef29e5e099609a050726acaab17a69cdde",
    ),
    "runtime_profile": (
        16503,
        "2ecca7a80390280bd3f2b43bb8cb883b5c496cb3f6b4c79cbad00efb86ef1007",
    ),
    "all_rank_image_verification": (
        619,
        "b1eeb1273a4b96daf2da2a5b17be1574b17d2d4344415cc77b49b9b4b77b86ae",
    ),
    "all_rank_preflight": (
        73011,
        "6fb487261eabcf088830d08afb1ef6d722be9f16fce4ae1bad68f4736dc5b22e",
    ),
    "all_rank_start_receipt": (
        1096,
        "a50784a2ac54eec0bee164c1676b285bb65465e1249fe22847455e75f7d5c099",
    ),
    "semantic_canary_receipt": (
        424,
        "df6b0c6892f38154b5a4a693654bee1bceb556564e14a8a7ce43ad48033002b2",
    ),
    "stored_prefix_request_receipt": (
        394,
        "1a76f8daec8e020b98099bdf326dbd9fec386304ea1314254a08c5f0e5b249b5",
    ),
    "distinct_prefix_request_receipt": (
        394,
        "c9e21368b11658082535a3ca25cbca1b4a93981684688a4274545168baa59b08",
    ),
    "stored_prefix_log_boundary": (
        30,
        "77ed985ed5a25647e808e1cc5d02d70148f5ada082a4f620259fdb8a03f78ba0",
    ),
    "distinct_prefix_log_boundary": (
        30,
        "27c15206d46e7f2423cc18be0d78f0811eb20ee773d7fdc3f4dc921b9e594ec5",
    ),
}


def canonical_payload_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate_receipt(document: dict[str, Any]) -> list[str]:
    """Return every contract violation without requiring CUDA or a live service."""

    errors: list[str] = []
    _require(
        document.get("schema")
        == "sparkcache-glm53-event-fenced-flat-restore-research/v1",
        "schema must identify the event-fenced flat-restore research receipt",
        errors,
    )
    _require(
        document.get("status") == "research-only",
        "status must be research-only",
        errors,
    )
    _require(
        document.get("conclusion") == "two-exact-oracles-passed",
        "conclusion must describe the two retained oracle passes",
        errors,
    )

    runtime = document.get("runtime", {})
    _require(
        runtime.get("image_id")
        == "sha256:f2723a71b49509294072f5886b4fe081ac1f87dd1f931cc3cb8f538bc3eb037d",
        "runtime image identity changed",
        errors,
    )
    _require(
        runtime.get("sparkcache_commit") == "861a9651e043709340644b2c7512cf82fc86c701",
        "evaluated SparkCache commit changed",
        errors,
    )
    _require(
        runtime.get("sparkcache_tree") == "85353fe6a7d7540bcb5d605945fb6fc792f9fdce",
        "evaluated SparkCache tree changed",
        errors,
    )
    _require(
        runtime.get("sparkcache_source_sha256")
        == "b2f7277c3c0ee13e07e70aed45fdbc005f57ef4b4d8475af1dd84ef7f176eb88",
        "evaluated SparkCache source SHA-256 changed",
        errors,
    )
    _require(
        runtime.get("vllm_contract_sha256")
        == "8adbdfa3fd4b06b213c3aab45255a0b039f1c9940a4b1fad0efd004d263227c9",
        "vLLM contract changed",
        errors,
    )
    _require(
        runtime.get("cuda_placement_library_sha256")
        == "d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c",
        "CUDA placement library changed",
        errors,
    )
    _require(runtime.get("topology") == "TP4/DCP1", "topology changed", errors)
    _require(
        runtime.get("publication_schema") == "snapshot-v1",
        "publication schema changed",
        errors,
    )
    _require(
        runtime.get("cache_namespace_impact") == "none",
        "namespace impact changed",
        errors,
    )
    _require(
        runtime.get("maximum_authenticated_readers") == 4,
        "reader bound changed",
        errors,
    )

    ordering = document.get("prior_cuda_work_ordering", {})
    _require(
        ordering.get("status") == "implemented", "CUDA ordering status changed", errors
    )
    _require(
        ordering.get("commit") == runtime.get("sparkcache_commit"),
        "CUDA ordering commit must match the evaluated source",
        errors,
    )
    _require(
        ordering.get("record_failure_behavior") == "recompute",
        "CUDA event-record failure must require recomputation",
        errors,
    )
    _require(
        len(ordering.get("gpu_free_tests", [])) == 3,
        "three ordering tests must be named",
        errors,
    )

    artifacts = document.get("captured_artifacts", {})
    _require(
        set(artifacts) == set(EXPECTED_ARTIFACTS),
        "captured artifact roles changed",
        errors,
    )
    for role, artifact in artifacts.items():
        digest = artifact.get("sha256", "") if isinstance(artifact, dict) else ""
        size = artifact.get("bytes", 0) if isinstance(artifact, dict) else 0
        _require(
            bool(SHA256.fullmatch(digest)), f"{role} has an invalid SHA-256", errors
        )
        _require(
            isinstance(size, int) and size > 0,
            f"{role} has an invalid byte count",
            errors,
        )
        expected_artifact = EXPECTED_ARTIFACTS.get(role)
        if expected_artifact is not None:
            _require(
                (size, digest) == expected_artifact,
                f"{role} identity changed",
                errors,
            )

    observations = document.get("observations", [])
    _require(
        len(observations) == 2, "exactly two observations must be retained", errors
    )
    observed_roles = {
        item.get("role") for item in observations if isinstance(item, dict)
    }
    _require(
        observed_roles == set(EXPECTED_OBSERVATIONS),
        "observation roles changed",
        errors,
    )

    for observation in observations:
        if not isinstance(observation, dict):
            errors.append("observation must be an object")
            continue
        role = observation.get("role")
        expected = EXPECTED_OBSERVATIONS.get(role)
        if expected is None:
            continue
        for field, value in expected.items():
            _require(observation.get(field) == value, f"{role} {field} changed", errors)
        _require(
            observation.get("prompt_tokens") == 131073,
            f"{role} prompt length changed",
            errors,
        )
        _require(
            observation.get("restored_tokens") == 131072,
            f"{role} restore span changed",
            errors,
        )
        _require(
            observation.get("http_status") == 200, f"{role} HTTP status changed", errors
        )
        _require(
            observation.get("completion_tokens") == 1,
            f"{role} completion size changed",
            errors,
        )
        _require(
            observation.get("expected_oracle") == "red",
            f"{role} expected oracle changed",
            errors,
        )
        _require(
            observation.get("observed_oracle") == "red",
            f"{role} observed oracle changed",
            errors,
        )
        _require(
            observation.get("oracle_match") is True,
            f"{role} oracle did not pass",
            errors,
        )
        _require(
            bool(SHA256.fullmatch(str(observation.get("response_sha256", "")))),
            f"{role} response SHA-256 is invalid",
            errors,
        )

        rank_records = observation.get("rank_records", [])
        _require(
            len(rank_records) == 4, f"{role} must contain four rank records", errors
        )
        _require(
            [item.get("rank") for item in rank_records if isinstance(item, dict)]
            == [0, 1, 2, 3],
            f"{role} rank order changed",
            errors,
        )
        for rank_record in rank_records:
            if not isinstance(rank_record, dict):
                errors.append(f"{role} rank record must be an object")
                continue
            rank = rank_record.get("rank")
            payload = rank_record.get("payload", {})
            _require(
                isinstance(payload, dict),
                f"{role} rank {rank} payload is invalid",
                errors,
            )
            if not isinstance(payload, dict):
                continue
            _require(
                rank_record.get("canonical_payload_sha256")
                == canonical_payload_sha256(payload),
                f"{role} rank {rank} canonical payload SHA-256 changed",
                errors,
            )
            _require(
                payload.get("schema") == "sparkcache-restore-timing/v1",
                f"{role} rank {rank} timing schema changed",
                errors,
            )
            _require(
                payload.get("request_id") == expected["request_id"],
                f"{role} rank {rank} request changed",
                errors,
            )
            _require(
                payload.get("digest") == expected["context_digest"][:12],
                f"{role} rank {rank} digest changed",
                errors,
            )
            _require(
                payload.get("span_tokens") == 131072,
                f"{role} rank {rank} span changed",
                errors,
            )
            _require(
                payload.get("storage_mode") == "block_pages_v1",
                f"{role} rank {rank} storage mode changed",
                errors,
            )
            _require(
                payload.get("outcome") == "verified",
                f"{role} rank {rank} outcome changed",
                errors,
            )
            _require(
                payload.get("chunk_count") == 13,
                f"{role} rank {rank} object count changed",
                errors,
            )
            _require(
                payload.get("page_bytes") == 813068464,
                f"{role} rank {rank} byte count changed",
                errors,
            )
            phase_ms = payload.get("phase_ms", {})
            _require(
                set(phase_ms) == PHASES, f"{role} rank {rank} phase set changed", errors
            )
            _require(
                all(
                    isinstance(value, (int, float)) and value >= 0
                    for value in phase_ms.values()
                ),
                f"{role} rank {rank} phase duration is invalid",
                errors,
            )
            _require(
                0 < phase_ms.get("prior_cuda_work", 0) < 1,
                f"{role} rank {rank} did not retain the CUDA prerequisite wait",
                errors,
            )
            queue = payload.get("queue_wait_ms", -1)
            service = payload.get("service_ms", -1)
            end_to_end = payload.get("end_to_end_ms", -1)
            _require(
                all(
                    isinstance(value, (int, float)) and value >= 0
                    for value in (queue, service, end_to_end)
                ),
                f"{role} rank {rank} aggregate timing is invalid",
                errors,
            )
            _require(
                abs((queue + service) - end_to_end) <= 0.002,
                f"{role} rank {rank} queue and service timing disagree with end-to-end timing",
                errors,
            )

    if len(observations) == 2:
        _require(
            observations[0].get("prompt_sha256")
            != observations[1].get("prompt_sha256"),
            "the two prompt identities must be distinct",
            errors,
        )
        _require(
            observations[0].get("context_digest")
            != observations[1].get("context_digest"),
            "the two stored-prefix identities must be distinct",
            errors,
        )

    log_retention = document.get("raw_log_retention", {})
    _require(
        log_retention.get("rank_log_files_retained") is False,
        "receipt must not imply that complete rank log files were retained",
        errors,
    )
    _require(
        log_retention.get("structured_payloads_retained") is True,
        "structured timing payloads must remain retained",
        errors,
    )

    admission = document.get("admission", {})
    _require(
        admission.get("deployable") is False,
        "receipt cannot authorize deployment",
        errors,
    )
    _require(
        admission.get("qualified") is False,
        "receipt cannot claim qualification",
        errors,
    )
    not_established = set(document.get("scope", {}).get("not_established", []))
    _require(
        {
            "Tail-only page-delta semantic correctness",
            "Segment-level shared restore correctness",
            "Concurrent restore correctness",
            "A generally qualified or deployable GLM-5.3 runtime",
        }.issubset(not_established),
        "receipt must retain its semantic and concurrency limitations",
        errors,
    )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", nargs="?", type=Path, default=DEFAULT_RECEIPT)
    args = parser.parse_args()
    document = json.loads(args.receipt.read_text(encoding="utf-8"))
    errors = validate_receipt(document)
    result = {
        "schema": "sparkcache-research-receipt-validation/v1",
        "status": "verified" if not errors else "rejected",
        "capability_status": "research-only",
        "receipt": str(args.receipt),
        "errors": errors,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
