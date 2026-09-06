"""Execute V2 request initialization from the shipped patch without GPUs.

The upstream fixture is the complete source file at vLLM commit
815f839060c2781f6bcc47c0d584358b400ea0ea. Applying the shipped patch before
extracting the method checks the delivered code, without importing vLLM or
fetching source. The source override supports testing a local vLLM edit before
regenerating the shipped patch.
"""

from __future__ import annotations

import ast
import hashlib
import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
PATCH = ROOT / "patches/vllm-pr646-815f839/010-sparkcache-runtime-contract.patch"
RELATIVE_SOURCE = "vllm/v1/worker/gpu/model_states/mamba_hybrid.py"
PREIMAGE = Path(__file__).with_name("fixtures") / "pr646_815f839_mamba_hybrid.py.txt"
PREIMAGE_SHA256 = "b34cb130e233f4d322390acf7fec01a3a368090c7f79da1fa7770c622a2f0dde"


class _Cell:
    def __init__(self, value):
        self.value = value

    def fill_(self, value):
        self.value = value


class _DefaultModelState:
    def add_request(self, req_index, new_req_data):
        self.parent_calls.append((req_index, new_req_data))


@pytest.fixture(scope="module")
def model_state_type(tmp_path_factory):
    preimage = PREIMAGE.read_bytes().replace(b"\r\n", b"\n")
    assert hashlib.sha256(preimage).hexdigest() == PREIMAGE_SHA256
    source_override = os.environ.get("SPARKCACHE_PR646_VLLM_TEST_SOURCE")
    if source_override:
        source_path = Path(source_override) / RELATIVE_SOURCE
    else:
        worktree = tmp_path_factory.mktemp("pr646-mamba-state")
        source_path = worktree / RELATIVE_SOURCE
        source_path.parent.mkdir(parents=True)
        source_path.write_bytes(preimage)
        result = subprocess.run(
            ["git", "apply", f"--include={RELATIVE_SOURCE}", str(PATCH)],
            cwd=worktree,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, result.stderr

    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    model_class = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "MambaHybridModelState"
    )
    model_class.body = [
        node
        for node in model_class.body
        if isinstance(node, ast.FunctionDef) and node.name == "add_request"
    ]
    assert len(model_class.body) == 1
    scope = {"DefaultModelState": _DefaultModelState}
    exec(
        compile(
            ast.Module(body=[model_class], type_ignores=[]), str(source_path), "exec"
        ),
        scope,
    )
    return scope["MambaHybridModelState"]


@pytest.mark.parametrize("capacity,slot", [(1, 0), (2, 1), (4, 2)])
@pytest.mark.parametrize("computed_tokens", [0, 8192, 523520])
@pytest.mark.parametrize(
    "attention_block_size,recurrent_block_size", [(2048, 256), (256, 256), (2048, 2048)]
)
def test_restored_state_uses_recurrent_block_geometry(
    model_state_type,
    capacity,
    slot,
    computed_tokens,
    attention_block_size,
    recurrent_block_size,
):
    model = model_state_type()
    model.parent_calls = []
    model.cache_config = SimpleNamespace(
        block_size=attention_block_size,
        mamba_block_size=recurrent_block_size,
    )
    model._align_mode = True
    model.num_accepted_tokens_gpu = [_Cell(7) for _ in range(capacity)]
    model._mamba_state_idx_gpu = [_Cell(-999) for _ in range(capacity)]
    request = SimpleNamespace(num_computed_tokens=computed_tokens)

    model.add_request(slot, request)

    expected_column = (computed_tokens - 1) // recurrent_block_size
    actual_column = model._mamba_state_idx_gpu[slot].value
    assert actual_column == expected_column
    assert model.parent_calls == [(slot, request)]
    assert model.num_accepted_tokens_gpu[slot].value == 1
    for untouched in set(range(capacity)) - {slot}:
        assert model._mamba_state_idx_gpu[untouched].value == -999
        assert model.num_accepted_tokens_gpu[untouched].value == 7

    if computed_tokens:
        # Align-mode external allocation retains only the last recurrent
        # checkpoint; preceding columns contain the null physical block.
        block_table = [0] * expected_column + [71]
        assert block_table[actual_column] == 71
    else:
        # The pre-copy kernel recognizes -1 as a fresh request and skips
        # copying any initial state into its first running block.
        assert actual_column == -1


def test_non_align_request_does_not_require_recurrent_geometry(model_state_type):
    model = model_state_type()
    model.parent_calls = []
    model.cache_config = SimpleNamespace(block_size=2048, mamba_block_size=None)
    model._align_mode = False
    model.num_accepted_tokens_gpu = [_Cell(7)]
    request = SimpleNamespace(num_computed_tokens=8192)

    model.add_request(0, request)

    assert model.parent_calls == [(0, request)]
    assert model.num_accepted_tokens_gpu[0].value == 1
