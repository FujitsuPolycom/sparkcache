from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).parents[2]
PATCH = ROOT / "patches/vllm-da4d7be/030-sparkcache-hma-load-failure.patch"

# Exact da4d7be method body around the defective single-group assumption. The
# test applies the shipped source patch, then executes the resulting method
# with lightweight scheduler and request objects. No vLLM or GPU import is
# required.
PREIMAGE_SOURCE = b'''from __future__ import annotations

class Scheduler:
    def _update_requests_with_invalid_blocks(
        self,
        requests: Iterable[Request],
        invalid_block_ids: set[int],
        num_scheduled_tokens: dict[str, int],
        evict_blocks: bool = True,
    ) -> tuple[set[str], int, set[int]]:
        affected_req_ids: set[str] = set()
        total_affected_tokens = 0
        blocks_to_evict: set[int] = set()
        marked_invalid_block_ids: set[int] = set()
        for request in requests:
            is_affected = False
            marked_invalid_block = False
            req_id = request.request_id
            # TODO (davidb): add support for hybrid memory allocator
            (req_block_ids,) = self.kv_cache_manager.get_block_ids(req_id)
            # We iterate only over blocks that may contain externally computed
            # tokens
            req_num_computed_tokens = (
                request.num_computed_tokens - num_scheduled_tokens.get(req_id, 0)
            )

            req_num_computed_blocks = (
                req_num_computed_tokens + self.block_size - 1
            ) // self.block_size
            for idx, block_id in zip(range(req_num_computed_blocks), req_block_ids):
                if block_id not in invalid_block_ids:
                    continue

                is_affected = True

                if block_id in marked_invalid_block_ids:
                    continue

                marked_invalid_block_ids.add(block_id)

                if marked_invalid_block:
                    continue

                marked_invalid_block = True
                request.num_computed_tokens = idx * self.block_size
                num_affected_tokens = (
                    req_num_computed_tokens - request.num_computed_tokens
                )
                total_affected_tokens += num_affected_tokens

                if evict_blocks:
                    blocks_to_evict.update(req_block_ids[idx:])

            if is_affected:
                if not marked_invalid_block:
                    total_affected_tokens += (
                        request.num_computed_tokens - req_num_computed_tokens
                    )
                    request.num_computed_tokens = req_num_computed_tokens

                affected_req_ids.add(request.request_id)

        return affected_req_ids, total_affected_tokens, blocks_to_evict
'''


def _patched_scheduler_class() -> type:
    with tempfile.TemporaryDirectory() as temporary:
        repository = Path(temporary)
        target = repository / "vllm/v1/core/sched/scheduler.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(PREIMAGE_SOURCE)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "core.autocrlf", "false"],
            cwd=repository,
            check=True,
        )
        subprocess.run(["git", "add", "-A"], cwd=repository, check=True)
        subprocess.run(
            ["git", "commit", "-q", "-m", "preimage"],
            cwd=repository,
            check=True,
            env={
                **os.environ,
                "GIT_AUTHOR_NAME": "test",
                "GIT_AUTHOR_EMAIL": "test@example.invalid",
                "GIT_COMMITTER_NAME": "test",
                "GIT_COMMITTER_EMAIL": "test@example.invalid",
            },
        )
        subprocess.run(
            ["git", "apply", str(PATCH)],
            cwd=repository,
            check=True,
        )
        source = target.read_text(encoding="utf-8")

    namespace: dict[str, object] = {}
    exec(compile(source, str(target), "exec"), namespace)
    scheduler = namespace["Scheduler"]
    assert isinstance(scheduler, type)
    return scheduler


class _BlockManager:
    def __init__(self, groups: tuple[list[int], ...]) -> None:
        self.groups = groups

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        assert request_id == "request"
        return self.groups


def _run(
    groups: tuple[list[int], ...],
    invalid: set[int],
    *,
    evict_blocks: bool = True,
) -> tuple[SimpleNamespace, tuple[set[str], int, set[int]]]:
    scheduler = _patched_scheduler_class()()
    scheduler.block_size = 256
    scheduler.kv_cache_manager = _BlockManager(groups)
    request = SimpleNamespace(request_id="request", num_computed_tokens=1024)
    result = scheduler._update_requests_with_invalid_blocks(
        [request], invalid, {}, evict_blocks
    )
    return request, result


def test_hybrid_failure_recomputes_the_whole_external_prefix() -> None:
    request, result = _run(([1, 2], [3, 4]), {3})
    assert request.num_computed_tokens == 0
    assert result == ({"request"}, 1024, {1, 2, 3, 4})


def test_disjoint_hybrid_failure_does_not_modify_the_request() -> None:
    request, result = _run(([1, 2], [3, 4]), {9})
    assert request.num_computed_tokens == 1024
    assert result == (set(), 0, set())


def test_hybrid_async_failure_does_not_evict_unpublished_blocks() -> None:
    request, result = _run(([1, 2], [3, 4]), {3}, evict_blocks=False)
    assert request.num_computed_tokens == 0
    assert result == ({"request"}, 1024, set())


def test_single_group_partial_prefix_recovery_is_preserved() -> None:
    request, result = _run(([1, 2, 3, 4],), {2})
    assert request.num_computed_tokens == 256
    assert result == ({"request"}, 768, {2, 3, 4})
