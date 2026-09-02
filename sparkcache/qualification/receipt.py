"""Receipt schema and validation for page-tail qualification evidence.

One receipt documents one complete qualification cohort: the cache-identity
namespace, the byte-exact fixture identity, publication and restore
measurements, restart restoration, corruption rejection, and the logical
versus unique publication byte accounting. Validation is GPU-free and shared
by the local harness, its regression tests, and the live runner under
``deploy/`` so no producer can understate a failed check.
"""

from __future__ import annotations

import json
import re
from typing import Any


RECEIPT_SCHEMA = "sparkcache-page-tail-qualification/v1"

_RUNNERS = frozenset({"local", "live"})
_SHA256 = re.compile(r"[0-9a-f]{64}")
_DELTA_SCHEMAS = frozenset(
    {
        "sparkcache-page-delta-manifest/v2",
        "sparkcache-page-snapshot-manifest/v2",
    }
)


def canonical_payload_sha256(receipt: dict[str, Any]) -> str:
    """SHA-256 of the canonical JSON encoding of one receipt."""

    import hashlib

    encoded = json.dumps(
        receipt, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _digest(value: Any) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _nonnegative_ms(value: Any) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value >= 0.0
    )


def validate_receipt(receipt: Any) -> list[str]:
    """Return every contract violation in one receipt.

    Validation never re-derives storage bytes; it enforces the schema,
    integrity invariants, and measurement presence that every producer must
    satisfy before its cohort is accepted as qualification evidence.
    """

    errors: list[str] = []

    def require(condition: Any, message: str) -> None:
        if not condition:
            errors.append(message)

    if not isinstance(receipt, dict):
        return ["receipt is not an object"]
    require(receipt.get("schema") == RECEIPT_SCHEMA, "schema differs")
    require(receipt.get("runner") in _RUNNERS, "runner is unknown")

    identity = receipt.get("identity")
    require(isinstance(identity, dict), "identity is not an object")
    if isinstance(identity, dict):
        require(
            identity.get("publication_schema") == "page-tail-cow-v1",
            "identity publication schema differs",
        )
        require(
            identity.get("record_schema") == ["target_ckv", "logical_positions"],
            "identity record schema differs",
        )
        require(identity.get("tp_degree") == 4, "identity tp_degree differs")
        require(identity.get("dcp_degree") == 4, "identity dcp_degree differs")
        require(
            isinstance(identity.get("dcp_shard_rank"), int)
            and 0 <= identity["dcp_shard_rank"] < 4,
            "dcp_shard_rank is out of range",
        )
        require(_digest(identity.get("storage_key")), "storage_key is not a digest")
        require(
            identity.get("chunk_tokens") == 256, "identity chunk_tokens differs"
        )

    fixture = receipt.get("fixture")
    require(isinstance(fixture, dict), "fixture is not an object")
    if isinstance(fixture, dict):
        require(
            _digest(fixture.get("layout_sha256")), "fixture layout is not a digest"
        )
        require(
            fixture.get("tokens_per_page") == 256,
            "fixture tokens_per_page differs",
        )
        require(
            isinstance(fixture.get("identity_salt"), str)
            and fixture["identity_salt"] != "",
            "fixture identity_salt is absent",
        )

    base = receipt.get("base")
    require(isinstance(base, dict), "base is not an object")
    if isinstance(base, dict):
        require(_digest(base.get("context_digest")), "base digest is not a digest")
        require(
            _digest(base.get("manifest_digest")), "base manifest is not a digest"
        )
        require(_nonnegative_ms(base.get("commit_ms")), "base commit_ms is invalid")
        require(
            isinstance(base.get("snapshot_bytes"), int) and base["snapshot_bytes"] > 0,
            "base snapshot_bytes is invalid",
        )
        require(
            isinstance(base.get("committed_tokens"), int)
            and base["committed_tokens"] > 0,
            "base committed_tokens is invalid",
        )

    deltas = receipt.get("deltas")
    require(isinstance(deltas, list) and deltas, "deltas are absent")
    if isinstance(deltas, list) and deltas:
        prior_tokens = base["committed_tokens"] if isinstance(base, dict) else 0
        prior_digest = base["context_digest"] if isinstance(base, dict) else ""
        for index, delta in enumerate(deltas):
            require(isinstance(delta, dict), f"delta {index} is not an object")
            if not isinstance(delta, dict):
                continue
            require(
                delta.get("base_context_digest") == prior_digest,
                f"delta {index} base digest does not chain",
            )
            require(
                delta.get("base_committed_tokens") == prior_tokens,
                f"delta {index} base boundary does not chain",
            )
            require(
                isinstance(delta.get("committed_tokens"), int)
                and delta["committed_tokens"] > prior_tokens,
                f"delta {index} does not extend its base",
            )
            require(
                delta.get("manifest_schema") in _DELTA_SCHEMAS,
                f"delta {index} manifest schema is unsupported",
            )
            require(
                _digest(delta.get("context_digest")),
                f"delta {index} context digest is not a digest",
            )
            require(
                _digest(delta.get("result_snapshot_sha256")),
                f"delta {index} result snapshot digest is not a digest",
            )
            require(
                _nonnegative_ms(delta.get("commit_ms")),
                f"delta {index} commit_ms is invalid",
            )
            require(
                _nonnegative_ms(delta.get("restore_ms")),
                f"delta {index} restore_ms is invalid",
            )
            if delta.get("manifest_schema") == "sparkcache-page-delta-manifest/v2":
                require(
                    isinstance(delta.get("delta_objects"), int)
                    and delta["delta_objects"] > 0,
                    f"delta {index} has no delta objects",
                )
                require(
                    isinstance(delta.get("delta_encoded_bytes"), int)
                    and delta["delta_encoded_bytes"] > 0,
                    f"delta {index} has no delta bytes",
                )
                require(
                    isinstance(delta.get("base_snapshot_bytes"), int)
                    and delta["base_snapshot_bytes"] > 0,
                    f"delta {index} has no base snapshot bytes",
                )
            prior_tokens = delta.get(
                "committed_tokens", prior_tokens + 1
            )
            prior_digest = delta.get("context_digest", "")

    result = receipt.get("result")
    require(isinstance(result, dict), "result is not an object")
    if isinstance(result, dict) and isinstance(deltas, list) and deltas:
        require(
            result.get("context_digest") == deltas[-1].get("context_digest"),
            "result digest does not match the last delta",
        )
        require(
            result.get("committed_tokens") == deltas[-1].get("committed_tokens"),
            "result boundary does not match the last delta",
        )
        require(
            _digest(result.get("expected_snapshot_sha256")),
            "result expected snapshot digest is invalid",
        )
        require(
            _nonnegative_ms(result.get("restart_restore_ms")),
            "restart_restore_ms is invalid",
        )

    accounting = receipt.get("unique_publication_bytes")
    require(
        isinstance(accounting, int) and accounting >= 0,
        "unique_publication_bytes is invalid",
    )
    logical = receipt.get("logical_snapshot_bytes")
    require(
        isinstance(logical, int) and logical >= 0,
        "logical_snapshot_bytes is invalid",
    )

    corruption = receipt.get("corruption")
    require(isinstance(corruption, dict), "corruption probe is absent")
    if isinstance(corruption, dict):
        require(
            corruption.get("rejected") is True,
            "corruption probe did not reject",
        )
        require(
            corruption.get("lookup_reason") == "corrupt",
            "corruption probe did not report a corrupt miss",
        )

    validation = receipt.get("validation")
    require(isinstance(validation, dict), "validation is not an object")
    if isinstance(validation, dict):
        require(
            validation.get("byte_exact_restores") is True,
            "byte_exact_restores must be true for an accepted cohort",
        )
        require(
            validation.get("complete_snapshot_equal") is True,
            "complete_snapshot_equal must be true for an accepted cohort",
        )
        require(
            validation.get("corruption_rejected") is True,
            "corruption_rejected must be true for an accepted cohort",
        )

    live = receipt.get("live") if receipt.get("runner") == "live" else {}
    if receipt.get("runner") == "live":
        require(isinstance(live, dict), "live receipt lacks the live section")
        if isinstance(live, dict):
            for field in ("base_bytes", "delta_bytes", "commit_ms_by_rank"):
                require(field in live, f"live receipt lacks {field}")

    return errors
