"""D21: legacy or incomplete contracts cannot enable connector-owned reads."""

import hashlib
import json
from collections import defaultdict

import pytest

from sparkcache.runtime_patches.generic_connector_contract import (
    CONNECTOR_API,
    REQUIRED_SEMANTICS,
    REQUIRED_SYMBOLS,
    verify_connector_job_contract,
)
from sparkcache.runtime_patches.verify_lease_contract import ContractError


@pytest.fixture
def contract_tree(tmp_path):
    files = []
    for path, symbols in REQUIRED_SYMBOLS.items():
        classes = defaultdict(list)
        for symbol in symbols:
            name, member = symbol.split(".")
            classes[name].append(member)
        source = (
            "\n".join(
                f"class {name}:\n"
                + "\n".join(f"    {member} = None" for member in members)
                for name, members in classes.items()
            )
            + "\n"
        )
        target = tmp_path / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.encode())
        files.append(
            {
                "path": path,
                "sha256": hashlib.sha256(source.encode()).hexdigest(),
                "required_symbols": list(symbols),
            }
        )
    record = {
        "schema": "sparkring-vllm-kv-block-lease-contract/v1",
        "connector_api": CONNECTOR_API,
        "required_semantics": list(REQUIRED_SEMANTICS),
        "files": files,
    }
    path = tmp_path / "contract.json"
    path.write_text(json.dumps(record))
    return tmp_path, path, record


def test_d21_matching_legacy_hashes_do_not_qualify_job_leases(contract_tree):
    root, path, record = contract_tree
    record.pop("connector_api")
    path.write_text(json.dumps(record))
    with pytest.raises(ContractError, match="does not qualify"):
        verify_connector_job_contract(root, path)


def test_d21_post_mtp_semantics_are_required_even_with_matching_hashes(contract_tree):
    root, path, record = contract_tree
    record["required_semantics"].remove(
        "worker-post-forward-follows-target-and-mtp-state-writes"
    )
    path.write_text(json.dumps(record))
    with pytest.raises(ContractError, match="ownership semantics"):
        verify_connector_job_contract(root, path)


def test_d21_missing_pool_lifetime_source_cannot_be_omitted(contract_tree):
    root, path, record = contract_tree
    record["files"] = [
        r for r in record["files"] if r["path"] != "vllm/v1/core/block_pool.py"
    ]
    path.write_text(json.dumps(record))
    with pytest.raises(ContractError, match="required capabilities"):
        verify_connector_job_contract(root, path)


def test_d21_source_change_cannot_reuse_reviewed_capability_labels(contract_tree):
    root, path, _record = contract_tree
    assert len(verify_connector_job_contract(root, path)) == 10
    source = root / "vllm/v1/worker/gpu/model_runner.py"
    source.write_text(
        source.read_text() + "# Different ordering requires source review.\n"
    )
    with pytest.raises(ContractError, match="mismatch"):
        verify_connector_job_contract(root, path)
