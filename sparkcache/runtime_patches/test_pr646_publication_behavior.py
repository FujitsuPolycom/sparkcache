"""Execute publication methods from the shipped patch without importing vLLM."""

from __future__ import annotations

import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest


PATCH = Path(__file__).resolve().parents[2] / (
    "patches/vllm-pr646-815f839/010-sparkcache-runtime-contract.patch"
)


def added_method(name: str):
    lines = PATCH.read_text(encoding="utf-8").splitlines()
    start = next(
        i for i, line in enumerate(lines) if line.startswith(f"+    def {name}(")
    )
    selected = []
    for line in lines[start:]:
        if not line.startswith("+") or line.startswith("+++"):
            break
        if selected and line.startswith("+    def "):
            break
        selected.append(line[1:])
    source = "from __future__ import annotations\n" + textwrap.dedent(
        "\n".join(selected)
    )
    scope = {"get_group_id": lambda block_hash: block_hash[0]}
    exec(compile(source, str(PATCH), "exec"), scope)
    return scope[name]


def recurrent_manager(*, tokens: int = 4096, block_hash=(2, "prefix")):
    block = SimpleNamespace(
        block_id=71,
        is_null=False,
        block_hash=block_hash,
        block_hash_num_tokens=tokens,
    )
    manager = SimpleNamespace(
        block_size=256,
        kv_cache_group_id=2,
        recurrent_publication_boundary=tokens,
        block_pool=SimpleNamespace(hash_block_size=256),
        req_to_blocks={"request": [None] * 15 + [block]},
        _pending_aligned_recurrent_boundaries=[],
    )
    return manager, block


def test_aligned_boundary_requires_exact_group_and_token_proof():
    queue = added_method("_queue_aligned_recurrent_boundary")
    request = SimpleNamespace(request_id="request", num_prompt_tokens=4353)
    manager, block = recurrent_manager()
    queue(manager, request, 4096)
    assert manager._pending_aligned_recurrent_boundaries == [
        ("request", 2, block, 4096)
    ]


@pytest.mark.parametrize(
    "fault", ["wrong_group", "wrong_tokens", "null", "unhashed", "short"]
)
def test_unproven_recurrent_pages_are_never_published(fault):
    queue = added_method("_queue_aligned_recurrent_boundary")
    manager, block = recurrent_manager()
    if fault == "wrong_group":
        block.block_hash = (3, "prefix")
    elif fault == "wrong_tokens":
        block.block_hash_num_tokens = 4352
    elif fault == "null":
        block.is_null = True
    elif fault == "unhashed":
        block.block_hash = None
    else:
        manager.req_to_blocks["request"] = []
    queue(manager, SimpleNamespace(request_id="request", num_prompt_tokens=4353), 4096)
    assert manager._pending_aligned_recurrent_boundaries == []


def test_boundary_pins_are_idempotent_and_require_live_hash_registration():
    pin = added_method("_pin_recurrent_boundary")
    registered = {(2, "prefix")}
    touched = []
    manager = SimpleNamespace(
        _partial_tail_pins={},
        block_pool=SimpleNamespace(
            cached_block_hash_to_block=SimpleNamespace(
                contain=lambda digest, block_id: digest in registered and block_id == 71
            ),
            touch=lambda blocks: touched.extend(blocks),
        ),
    )
    block = SimpleNamespace(block_id=71, is_null=False)
    assert pin(manager, "request", block, 4096, (2, "prefix")) is block
    assert pin(manager, "request", block, 4096, (2, "prefix")) is block
    assert touched == [block]
    registered.clear()
    with pytest.raises(AssertionError):
        pin(manager, "request", block, 4096, (2, "prefix"))
    assert touched == [block]


def test_publication_boundary_validation_keeps_256_token_units_at_512k():
    boundaries = added_method("_recurrent_publication_boundaries")
    targets = [522240, 4096, 522240]
    scheduler = SimpleNamespace(
        hash_block_size=256,
        connector=SimpleNamespace(
            get_recurrent_publication_boundaries=lambda request: targets
        ),
    )
    request = SimpleNamespace(num_prompt_tokens=524000)
    assert boundaries(scheduler, request) == (4096, 522240)
    for invalid in (0, 522241, 524288, True):
        targets[:] = [invalid]
        with pytest.raises(ValueError):
            boundaries(scheduler, request)


def test_no_connector_proposes_no_extra_boundary():
    boundaries = added_method("_recurrent_publication_boundaries")
    assert boundaries(SimpleNamespace(connector=None), object()) == ()
