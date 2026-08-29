from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
PATCH = ROOT / "patches/vllm-da4d7be/040-sparkcache-shared-prefix-lease.patch"
SCHEDULER_PATCH = ROOT / "patches/vllm-da4d7be/041-sparkcache-shared-prefix-attach.patch"

# Exact da4d7be contexts touched by patch 040, joined into a GPU-free fixture.
# The unchanged coordinator seam below models the public source contract:
# allocate_new_computed_blocks adopts existing per-group objects and increments
# their ordinary references; free releases only the named request's references.
PREIMAGE_SOURCE = b'''# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal, overload

KVCacheBlock = object

class _Logger:
    def exception(self, *args):
        pass

logger = _Logger()

@dataclass
class KVCacheBlocks:
    blocks: tuple[Sequence[KVCacheBlock], ...]

    def new_empty(self) -> "KVCacheBlocks":
        """
        Creates a new KVCacheBlocks instance with no blocks.
        """
        return KVCacheBlocks(tuple(() for _ in range(len(self.blocks))))


class KVCacheManager:
    def __init__(
        self,
        coordinator,
        num_groups: int,
    ):
        self.coordinator = coordinator
        self.block_pool = coordinator.block_pool
        self.num_kv_cache_groups = num_groups
        # Pre-constructed KVCacheBlocks with no blocks, callers should use this
        # via create_kv_cache_blocks instead of creating new ones to avoid GC
        # overhead.
        #
        # We use nested tuples to ensure the empty KVCacheBlocks is immutable.
        self.empty_kv_cache_blocks = KVCacheBlocks(
            tuple(() for _ in range(self.num_kv_cache_groups))
        )

        # Off-table cow blocks handed to a KV connector for partial-tail
        # offload; pinned until the request's blocks are freed.
        self._partial_tail_pins: dict[str, list[KVCacheBlock]] = {}

    def allocate_slots(self):
        if True:
            num_blocks_to_allocate = self.coordinator.get_num_blocks_to_allocate(
                apply_admission_cap=True,
            )
            required_blocks = num_blocks_to_allocate + watermark_blocks
            if required_blocks > self.block_pool.get_num_free_blocks():
                return None

        # Keep `reserved_blocks` free for other in-flight sequences, and an
        # additional watermark of headroom for waiting/preempted admissions.
        available_blocks = self.block_pool.get_num_free_blocks() - reserved_blocks
        required_blocks = num_blocks_to_allocate + watermark_blocks
        if required_blocks > available_blocks:
            # Cannot allocate new blocks
            return None

    def create_kv_cache_blocks(self, blocks):
        return KVCacheBlocks(blocks)

    def get_blocks(self, request_id: str) -> KVCacheBlocks:
        """Get the blocks of a request."""
        return self.create_kv_cache_blocks(self.coordinator.get_blocks(request_id))

    def get_block_ids(self, request_id: str) -> tuple[list[int], ...]:
        """Get the block ids of a request."""
        return self.get_blocks(request_id).get_block_ids()
'''

@dataclass(eq=False)
class _Block:
    name: str
    ref_cnt: int = 1
    is_null: bool = False
    block_id: int = 0


class _BlockPool:
    def __init__(self):
        self.next_id = 0
        self.free_blocks = 100

    def get_new_blocks(self, count: int) -> list[_Block]:
        blocks = [
            _Block(
                f"hot-{self.next_id + index}",
                block_id=self.next_id + index,
            )
            for index in range(count)
        ]
        self.next_id += count
        self.free_blocks -= count
        return blocks

    def get_num_free_blocks(self) -> int:
        return self.free_blocks


class _Manager:
    def __init__(self, coordinator: "_Coordinator", group_index: int):
        self.coordinator = coordinator
        self.group_index = group_index
        self.block_size = 300
        self._partial_hit_reqs: dict[str, tuple[int, _Block]] = {}
        self.new_block_ids: list[int] = []
        self.pending_copies: list[tuple[_Block, _Block]] = []

    @property
    def req_to_blocks(self) -> dict[str, list[_Block]]:
        return {
            request_id: groups[self.group_index]
            for request_id, groups in self.coordinator.tables.items()
        }

    def _apply_cow(
        self, request_id: str, block_idx: int, source: _Block, destination: _Block
    ) -> None:
        groups = self.coordinator.tables[request_id]
        assert groups[self.group_index][block_idx] is source
        groups[self.group_index][block_idx] = destination
        destination.ref_cnt += 1
        self.pending_copies.append((source, destination))


class _Coordinator:
    def __init__(self, source_groups: tuple[list[_Block], ...], null: _Block):
        self.tables: dict[str, tuple[list[_Block], ...]] = {
            "leader": tuple(list(group) for group in source_groups)
        }
        self.null = null
        self.block_pool = _BlockPool()
        self.authoritative_partial_slot: int | None = None
        self.single_type_managers = (
            _Manager(self, 0),
            _Manager(self, 1),
        )

    def get_blocks(self, request_id: str) -> tuple[list[_Block], ...]:
        return self.tables.get(request_id, ([], []))

    def allocate_new_computed_blocks(
        self,
        request_id: str,
        new_computed_blocks: tuple[list[_Block], ...],
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> None:
        assert num_local_computed_tokens == 1024
        assert num_external_computed_tokens == 0
        assert request_id not in self.tables
        # Group 0 is dense full attention. Group 1 models an align-mode Mamba
        # table whose historical slots become null and whose boundary state is
        # the only physical block needed by an attached request.
        dense = list(new_computed_blocks[0])
        partial_slot = (
            self.authoritative_partial_slot
            if request_id.startswith("\x00sparkcache-shared-prefix:")
            and self.authoritative_partial_slot is not None
            else 3
        )
        mamba = [self.null] * 4
        mamba[partial_slot] = new_computed_blocks[1][partial_slot]
        for block in dense + [mamba[partial_slot]]:
            block.ref_cnt += 1
        self.tables[request_id] = (dense, mamba)
        if not request_id.startswith("\x00sparkcache-shared-prefix:"):
            self.single_type_managers[0]._partial_hit_reqs[request_id] = (
                3,
                dense[-1],
            )
        # Model the second group's 2,304-token page at a 1,024-token partial
        # boundary. Patch 040 must replace this source page with a dedicated
        # hot destination before the lease can become READY.
        self.single_type_managers[1]._partial_hit_reqs[request_id] = (
            partial_slot,
            mamba[partial_slot],
        )

    def free(self, request_id: str) -> None:
        groups = self.tables.pop(request_id, ([], []))
        for block in groups[0] + groups[1]:
            if not block.is_null:
                block.ref_cnt -= 1


def _patched_manager_class() -> type:
    with tempfile.TemporaryDirectory() as temporary:
        repository = Path(temporary)
        target = repository / "vllm/v1/core/kv_cache_manager.py"
        target.parent.mkdir(parents=True)
        target.write_bytes(PREIMAGE_SOURCE)
        subprocess.run(["git", "init", "-q"], cwd=repository, check=True)
        subprocess.run(
            ["git", "config", "core.autocrlf", "false"], cwd=repository, check=True
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
        subprocess.run(["git", "apply", str(PATCH)], cwd=repository, check=True)
        source = target.read_text(encoding="utf-8")

    namespace: dict[str, object] = {}
    exec(compile(source, str(target), "exec"), namespace)
    manager = namespace["KVCacheManager"]
    assert isinstance(manager, type)
    return manager


def _manager():
    null = _Block("null", ref_cnt=0, is_null=True)
    dense = [_Block(f"fa-{index}") for index in range(4)]
    mamba = [_Block(f"mamba-{index}") for index in range(4)]
    coordinator = _Coordinator((dense, mamba), null)
    manager = _patched_manager_class()(coordinator, 2)
    return manager, coordinator, dense, mamba


def _fence_pending_copies(coordinator: _Coordinator) -> tuple[_Block, _Block]:
    hot_blocks = []
    for manager in coordinator.single_type_managers:
        source, hot = manager.pending_copies.pop()
        source.ref_cnt -= 1
        hot.ref_cnt -= 1
        hot_blocks.append(hot)
    return hot_blocks[0], hot_blocks[1]


def test_lease_and_follower_own_independent_standard_references() -> None:
    manager, coordinator, dense, mamba = _manager()

    assert manager.publish_shared_prefix_lease(
        "digest", "leader", 1024, 15.0, now=10.0
    )
    assert [block.ref_cnt for block in dense] == [2, 2, 2, 2]
    assert [block.ref_cnt for block in mamba] == [1, 1, 1, 2]
    assert manager.attach_shared_prefix_lease("digest", "too-early", now=10.5) == 0

    dense_hot, hot = _fence_pending_copies(coordinator)
    assert dense_hot is not dense[-1]
    assert hot is not mamba[-1]
    # Worker copy completion releases only the two copy-retention refs.
    assert manager.mark_shared_prefix_lease_ready("digest", now=10.6)

    assert manager.attach_shared_prefix_lease("digest", "follower", now=11.0) == 1024
    assert [block.ref_cnt for block in dense] == [3, 3, 3, 1]
    assert [block.ref_cnt for block in mamba] == [1, 1, 1, 1]
    assert [block.is_null for block in coordinator.get_blocks("follower")[1]] == [
        True,
        True,
        True,
        False,
    ]
    assert coordinator.single_type_managers[0]._partial_hit_reqs["follower"] == (
        3,
        dense_hot,
    )
    assert coordinator.single_type_managers[1]._partial_hit_reqs["follower"] == (
        3,
        hot,
    )

    manager.discard_shared_prefix_lease("digest")
    assert [block.ref_cnt for block in dense] == [2, 2, 2, 1]
    assert hot.ref_cnt == 1
    coordinator.free("leader")
    assert [block.ref_cnt for block in dense] == [1, 1, 1, 0]
    assert dense_hot.ref_cnt == 1
    assert mamba[-1].ref_cnt == 0
    assert hot.ref_cnt == 1


def test_expiry_releases_only_the_lease_pin_not_attached_request_refs() -> None:
    manager, coordinator, dense, mamba = _manager()
    assert manager.publish_shared_prefix_lease(
        "digest", "leader", 1024, 2.0, now=10.0
    )
    dense_hot, hot = _fence_pending_copies(coordinator)
    assert manager.mark_shared_prefix_lease_ready("digest", now=10.6)
    assert manager.attach_shared_prefix_lease("digest", "follower", now=11.0) == 1024

    manager.expire_shared_prefix_leases(now=12.001)

    assert manager.attach_shared_prefix_lease("digest", "late", now=12.001) == 0
    assert [block.ref_cnt for block in dense] == [2, 2, 2, 1]
    assert dense_hot.ref_cnt == 1
    assert mamba[-1].ref_cnt == 1
    assert hot.ref_cnt == 1
    assert coordinator.get_blocks("follower")[0] == dense[:3] + [dense_hot]


def test_lease_capacity_is_hard_bounded_to_two_entries() -> None:
    manager, coordinator, _, _ = _manager()
    for key, now in (("one", 1.0), ("two", 2.0), ("three", 3.0)):
        assert manager.publish_shared_prefix_lease(
            key, "leader", 1024, 15.0, now=now
        )

    assert set(manager._shared_prefix_leases) == {"two", "three"}
    assert not any(coordinator.get_blocks("\x00sparkcache-shared-prefix:one"))
    with pytest.raises(ValueError, match="at most two"):
        manager.publish_shared_prefix_lease(
            "four", "leader", 1024, 15.0, max_entries=3, now=4.0
        )


def test_null_partial_page_refuses_lease_and_pressure_drops_only_pin() -> None:
    manager, coordinator, _, mamba = _manager()
    mamba[-1].is_null = True
    assert not manager.publish_shared_prefix_lease(
        "invalid", "leader", 1024, 15.0, now=10.0
    )
    assert "invalid" not in manager._shared_prefix_leases

    manager, coordinator, _, _ = _manager()
    assert manager.publish_shared_prefix_lease(
        "pressure", "leader", 1024, 15.0, now=10.0
    )
    _fence_pending_copies(coordinator)
    assert manager.mark_shared_prefix_lease_ready("pressure", now=10.5)
    assert manager.attach_shared_prefix_lease(
        "pressure", "follower", now=10.6
    ) == 1024
    manager.block_pool.free_blocks = 0

    manager.evict_shared_prefix_leases_until_free(1, now=10.7)

    assert "pressure" not in manager._shared_prefix_leases
    assert any(coordinator.get_blocks("follower"))


def test_authoritative_checkpoint_slot_can_differ_from_arithmetic_boundary() -> None:
    manager, coordinator, _, mamba = _manager()
    coordinator.authoritative_partial_slot = 2

    assert manager.publish_shared_prefix_lease(
        "checkpoint", "leader", 1024, 15.0, now=10.0
    )

    mamba_source, mamba_hot = (
        coordinator.single_type_managers[1].pending_copies.pop()
    )
    assert mamba_source is mamba[2]
    assert mamba_hot is not mamba_source
    lease = manager._shared_prefix_leases["checkpoint"]
    lease_table = coordinator.get_blocks(lease.request_id)[1]
    assert lease_table[2] is mamba_hot


def test_scheduler_patch_orders_publish_and_attach_after_verification() -> None:
    source = SCHEDULER_PATCH.read_text(encoding="utf-8")

    assert source.index("expire_shared_prefix_leases") > source.index(
        "new_step_starts"
    )
    assert source.index("get_shared_prefix_lease_candidate") < source.index(
        "# Get already-cached tokens."
    )
    cache_at = source.index("self.kv_cache_manager.cache_blocks")
    publish_at = source.index("get_shared_prefix_lease_to_publish")
    full_hit_at = source.index("# on a full prompt hit")
    assert cache_at < publish_at < full_hit_at
