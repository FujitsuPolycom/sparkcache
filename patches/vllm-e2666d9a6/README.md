# SparkCache patches for vLLM build g/e2666d9a6

Connector-support patches for vLLM source build
`0.1.dev1+ge2666d9a6.d20260810`. The fixed-MTP4 GLM-5.2 serving-image lineage
is identified by SparkRing as `R7`. Whole-file pre- and postimage SHA-256 values
are recorded in [`preimages.json`](preimages.json); application requires an
exact preimage and rejects fuzz or unrecorded output.
DeepSeek cache groups use hybrid-memory-allocator (HMA) pages.

| Patch | Target | Effect |
|---|---|---|
| `010-sparkcache-async-rollback.patch` | `vllm/v1/core/sched/scheduler.py` | On the `kv_load_failure_policy=recompute` recovery path, moves in-flight speculative output placeholders of failed requests into `async_tokens_to_discard` so stale frames cannot drive the placeholder count negative. The mechanism it uses (`num_output_placeholders`, `async_tokens_to_discard`, and the force-preemption discard path) exists upstream in this build; the KV-load-failure hunk itself does not. |
| `011-sparkcache-glm52-async-rollback.patch` | `vllm/v1/core/sched/scheduler.py` | Applies the same rollback fix to the distinct stock scheduler preimage in the accepted GLM-5.2 R7 image. Its exact hunk location and whole-file hashes differ from the DeepSeek/adaptive scheduler used by patch 010. |
| `020-sparkcache-vmm-exemption.patch` | `vllm/config/vllm.py` | Exempts `SparkContextCacheConnector` from `VllmConfig._verify_kv_transfer_compat`'s rejection of KV connectors under `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`. The connector registers no GPU memory, so VMM remaps cannot invalidate a registration. Unnecessary when the deployment does not set `expandable_segments:True` or runs `enable_cumem_allocator`. |
| `030-sparkcache-hma-load-failure.patch` | `vllm/v1/core/sched/scheduler.py` | Makes `kv_load_failure_policy=recompute` conservative and HMA-aware: a failed page in any cache group invalidates the complete external prefix and reschedules from token zero instead of unpacking the request's block tables as a one-group tuple. Applies to the adaptive scheduler lineage (patch 010 preimage). |
| `031-sparkcache-stock-hma-load-failure.patch` | `vllm/v1/core/sched/scheduler.py` | Applies the same HMA-aware load-failure semantics to the stock scheduler lineage (patch 011 preimage). For multiple KV-cache groups, flattens all request group block IDs, detects any invalid-block intersection, invalidates the whole external prefix, resets `num_computed_tokens` to zero, accounts affected tokens, and evicts every group's blocks when requested. |

The matching KV-block-lease ownership contract for this build is
`sparkcache/runtime_patches/vllm-kv-block-lease-contract-e2666d9a6.json`.
Two scheduler lineages share the same vLLM commit but diverge at the
whole-file hash level. The **adaptive lineage** starts from the DeepSeek
development preimage, applies patch 010 (async rollback), then optionally
patch 030 (HMA load failure). The **stock lineage** starts from the
GLM-5.2 R7 image preimage, applies patch 011 (async rollback), then
optionally patch 031 (HMA load failure). Each patch is pinned by exact
whole-file preimage and postimage SHA-256; the contract verifier accepts
only the recorded states and rejects any other hash.

Deployment notes for this build, established by source analysis and
recorded here because they gate any live use:

- `kv_load_failure_policy` defaults to `fail` in this build; the connector
  requires `recompute` and refuses to start otherwise.
- DeepSeek-V4 models with `compress_ratios` use the
  `deepseek-v4-fp8-hma` profile. It persists opaque pages from all five HMA
  block tables, including compressor state, under a topology-hashed cache
  identity. This profile is qualified for TP2/DCP1 and TP4/DCP1; native
  restore and streaming snapshots remain unsupported for block-page storage.
- With speculative decoding configured, this build defers connector
  finalization until after the draft model runs. The synchronous store
  path is unaffected; the streaming-snapshot producer-stream capture has
  not been revalidated under that ordering and streaming must stay
  disabled on this build.
