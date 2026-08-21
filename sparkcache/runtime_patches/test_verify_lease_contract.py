from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sparkcache.runtime_patches.verify_lease_contract import (
    ContractError,
    verify_contract,
)

AGGREGATOR_RELATIVE_PATH = (
    "vllm/distributed/kv_transfer/kv_connector/utils.py"
)
AGGREGATOR_PINNED_SHA256 = (
    "df46c6d533affdfd1970219c32266186f7dbddf50fb94ab76a67a1878be90db5"
)
SCHEDULER_RELATIVE_PATH = "vllm/v1/core/sched/scheduler.py"
SCHEDULER_UPSTREAM_SHA256 = (
    "6108b941a539fd3ee046d8e4b38c79d4a275cb18ececb5efba5361af4ded28d5"
)


def write_contract(tmp_path: Path, relative: str, payload: bytes) -> tuple[Path, Path]:
    root = tmp_path / "vllm"
    source = root / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    contract = tmp_path / "contract.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "sparkring-vllm-kv-block-lease-contract/v1",
                "vllm_commit": "test",
                "files": [
                    {
                        "path": relative,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                        "required_symbols": [],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return root, contract


def test_exact_source_is_accepted(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"exact source")
    assert verify_contract(root, contract) == [(root / "vllm/base.py").resolve()]


def test_mutated_source_fails_closed(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"exact source")
    (root / "vllm/base.py").write_bytes(b"changed")
    with pytest.raises(ContractError, match="mismatch"):
        verify_contract(root, contract)


def test_declared_class_member_is_required(tmp_path: Path) -> None:
    payload = b"class Lease:\n    def release(self):\n        pass\n"
    root, contract = write_contract(tmp_path, "vllm/base.py", payload)
    document = json.loads(contract.read_text("utf-8"))
    document["files"][0]["required_symbols"] = ["Lease.release"]
    contract.write_text(json.dumps(document), encoding="utf-8")

    assert verify_contract(root, contract) == [(root / "vllm/base.py").resolve()]


def test_missing_required_class_member_fails_closed(tmp_path: Path) -> None:
    payload = b"class Lease:\n    pass\n"
    root, contract = write_contract(tmp_path, "vllm/base.py", payload)
    document = json.loads(contract.read_text("utf-8"))
    document["files"][0]["required_symbols"] = ["Lease.release"]
    contract.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContractError, match="required symbol absent"):
        verify_contract(root, contract)


def test_invalid_required_symbol_name_fails_closed(tmp_path: Path) -> None:
    payload = b"class Lease:\n    pass\n"
    root, contract = write_contract(tmp_path, "vllm/base.py", payload)
    document = json.loads(contract.read_text("utf-8"))
    document["files"][0]["required_symbols"] = ["not-qualified"]
    contract.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ContractError, match="invalid required symbol"):
        verify_contract(root, contract)


def test_model_serving_contract_pins_exact_kv_output_aggregator_source() -> None:
    contract_path = Path(__file__).with_name(
        "vllm-kv-block-lease-contract.json"
    )
    payload = json.loads(contract_path.read_text("utf-8"))
    records = [
        record
        for record in payload["files"]
        if record["path"] == AGGREGATOR_RELATIVE_PATH
    ]

    assert records == [
        {
            "path": AGGREGATOR_RELATIVE_PATH,
            "sha256": AGGREGATOR_PINNED_SHA256,
            "required_symbols": [
                "KVOutputAggregator.from_connector",
                "KVOutputAggregator.aggregate",
            ],
        }
    ]


def test_model_serving_contract_accepts_official_upstream_scheduler() -> None:
    contract_path = Path(__file__).with_name(
        "vllm-kv-block-lease-contract.json"
    )
    payload = json.loads(contract_path.read_text("utf-8"))
    record = next(
        record
        for record in payload["files"]
        if record["path"] == SCHEDULER_RELATIVE_PATH
    )

    assert record["accepted_sha256"]["official_upstream"] == (
        SCHEDULER_UPSTREAM_SHA256
    )


def test_kv_output_aggregator_exact_source_is_accepted(tmp_path: Path) -> None:
    root, contract = write_contract(
        tmp_path,
        AGGREGATOR_RELATIVE_PATH,
        b"exact pinned aggregator semantics",
    )

    assert verify_contract(root, contract) == [
        (root / AGGREGATOR_RELATIVE_PATH).resolve()
    ]


def test_kv_output_aggregator_mismatch_fails_closed(tmp_path: Path) -> None:
    root, contract = write_contract(
        tmp_path,
        AGGREGATOR_RELATIVE_PATH,
        b"exact pinned aggregator semantics",
    )
    (root / AGGREGATOR_RELATIVE_PATH).write_bytes(b"changed quorum semantics")

    with pytest.raises(
        ContractError,
        match=r"lease-contract mismatch .*kv_connector[\\/]utils\.py",
    ):
        verify_contract(root, contract)


def test_kv_output_aggregator_missing_file_fails_closed(
    tmp_path: Path,
) -> None:
    root, contract = write_contract(
        tmp_path,
        AGGREGATOR_RELATIVE_PATH,
        b"exact pinned aggregator semantics",
    )
    (root / AGGREGATOR_RELATIVE_PATH).unlink()

    with pytest.raises(
        ContractError,
        match=r"cannot hash .*kv_connector[\\/]utils\.py",
    ):
        verify_contract(root, contract)


def test_named_exact_source_state_is_accepted(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"post overlay")
    payload = json.loads(contract.read_text("utf-8"))
    payload["files"][0].pop("sha256")
    payload["files"][0]["accepted_sha256"] = {
        "base": hashlib.sha256(b"base").hexdigest(),
        "post_overlay": hashlib.sha256(b"post overlay").hexdigest(),
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")
    assert verify_contract(root, contract) == [(root / "vllm/base.py").resolve()]


def test_unknown_named_source_state_fails_closed(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"unknown")
    payload = json.loads(contract.read_text("utf-8"))
    payload["files"][0].pop("sha256")
    payload["files"][0]["accepted_sha256"] = {
        "base": hashlib.sha256(b"base").hexdigest(),
        "post_overlay": hashlib.sha256(b"post overlay").hexdigest(),
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="expected one of"):
        verify_contract(root, contract)


def test_record_cannot_mix_single_and_named_source_states(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"source")
    payload = json.loads(contract.read_text("utf-8"))
    payload["files"][0]["accepted_sha256"] = {
        "same": hashlib.sha256(b"source").hexdigest()
    }
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="exactly one"):
        verify_contract(root, contract)


def test_contract_path_may_not_escape_root(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"source")
    payload = json.loads(contract.read_text("utf-8"))
    payload["files"][0]["path"] = "../outside.py"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="unsafe"):
        verify_contract(root, contract)


def test_unsupported_schema_fails_closed(tmp_path: Path) -> None:
    root, contract = write_contract(tmp_path, "vllm/base.py", b"source")
    payload = json.loads(contract.read_text("utf-8"))
    payload["schema"] = "future"
    contract.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ContractError, match="unsupported"):
        verify_contract(root, contract)
