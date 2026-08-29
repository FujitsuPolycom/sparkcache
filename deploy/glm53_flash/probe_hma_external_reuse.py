#!/usr/bin/env python3
"""CPU probe for vLLM hybrid-prefix publication after an external load."""

from __future__ import annotations

from math import lcm

import torch

from vllm.sampling_params import SamplingParams
from vllm.utils.hashing import sha256
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request


def request(request_id: str, tokens: list[int], hash_block_size: int) -> Request:
    params = SamplingParams(max_tokens=1)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=tokens,
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(hash_block_size, sha256),
    )


def main() -> None:
    init_none_hash(sha256)
    hash_block_size = 2
    full_block_size = 2
    mamba_block_size = 8
    config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        prefix_cache_retention_interval=16,
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["full"],
                FullAttentionSpec(
                    block_size=full_block_size,
                    num_kv_heads=1,
                    head_size=1,
                    dtype=torch.float32,
                ),
            ),
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=mamba_block_size,
                    shapes=(1, 1),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    manager = KVCacheManager(
        config,
        max_model_len=8192,
        scheduler_block_size=lcm(full_block_size, mamba_block_size),
        hash_block_size=hash_block_size,
        enable_caching=True,
        use_eagle=True,
        num_prefill_lookahead=7,
    )
    tokens = list(range(42))
    leader = request("leader", tokens, hash_block_size)
    loaded = manager.allocate_slots(
        leader,
        num_new_tokens=10,
        num_external_computed_tokens=32,
        delay_cache_blocks=True,
    )
    assert loaded is not None
    leader.num_computed_tokens = 32
    manager.cache_blocks(leader, 32)
    continued = manager.allocate_slots(leader, num_new_tokens=10)
    assert continued is not None
    leader.num_computed_tokens = 42
    manager.remove_skipped_blocks(
        "leader", processed_computed_tokens=42, num_prompt_tokens=42
    )
    manager.free(leader)
    manager.new_step_starts()

    follower = request("follower", tokens, hash_block_size)
    blocks, hit, boundary = manager.get_computed_blocks(follower)
    print(
        {
            "hit_tokens": hit,
            "shared_prefix_boundary": boundary,
            "block_ids": blocks.get_block_ids(),
        }
    )
    assert hit > 0, "externally published hybrid prefix was not reusable"


if __name__ == "__main__":
    main()
