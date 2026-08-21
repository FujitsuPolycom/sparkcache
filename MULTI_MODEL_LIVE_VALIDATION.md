# Multi-model SparkCache live validation

Status: **qualified** for the exact configurations recorded below.

## Scope

On 2026-08-21, SparkCache source tree
`33fbe426045a64b4c46a957c39ebad7cfc85db35be0925ef77a017bf3e53adec`
from commit `82b35b3c345501bd3275029d7656facebe55ef23` was validated on two-
and four-Spark DeepSeek-V4-Flash-0731 services and on the four-Spark GLM-5.2
EXL3 3.5-bpw fixed-MTP4 service. The GLM deployment recipe's durable
identifier is `R7`.

The source implements immutable connector configuration in
`spark_context_cache_config.py` and explicit removal of inherited Docker image
environment values in `deploy/deployment_contract/container.py`. The GPU-free
repository gate passed 647 tests with 5 skipped, plus Ruff and Python
compilation.

## DeepSeek-V4-Flash-0731 TP2/DCP1

| Attribute | Value |
| --- | --- |
| Hardware | Two NVIDIA DGX Sparks |
| Image | `sha256:d316c96d7e1b77ea5459e778c869289c59f7d8a55f3ce274a2082ba48c1be9f6` |
| Model limit | 131,072 tokens |
| Checkpoint identity | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |
| Cache root | `/cache/sparkcache-dsv4` |
| Capacity policy | 200 GiB high, 180 GiB low, TTL disabled |

The connector source identified above discovered 19 manifests. A deterministic
73,774-token prompt committed a 73,728-token entry on both ranks. After a
coordinated engine restart, both ranks restored the entry asynchronously in
570.5 ms and 549.4 ms. The response was exactly `SPARKCACHE_OK:9540`; the
post-restore canary was exactly `SPARKCACHE_CANARY_OK`.

A fresh non-cached semantic probe correctly returned:

```json
{"action":"recompute","product":1517,"sum":9540}
```

The qualification-time Docker environments also confirmed that inherited
`SPARKRING_MODEL_CONFIG_SHA256`, `SPARKRING_MODEL_REPOSITORY`, and
`SPARKRING_MODEL_REVISION` values were explicitly empty.

## DeepSeek-V4-Flash-0731 TP4/DCP1

| Attribute | Value |
| --- | --- |
| Hardware | Four directly cabled NVIDIA DGX Sparks |
| Image | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Model limit | 524,288 tokens |
| Checkpoint identity | `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f` |
| Cache root | `/cache/sparkcache-deepseek0731-tp4-dcp1` |
| Capacity policy | 200 GiB high, 180 GiB low, TTL disabled |

A 294,958-token deterministic prompt snapshotted 294,912 reusable tokens
on all four ranks in 1.51-1.88 seconds. Durable background commits completed
in 10.32-11.81 seconds. After a coordinated restart, the qualified connector
restored the entry in:

| Physical rank | Restore time |
| ---: | ---: |
| 0 | 1,999.4 ms |
| 1 | 1,882.3 ms |
| 2 | 2,093.7 ms |
| 3 | 2,303.1 ms |

The external-cache response was exactly `SPARKCACHE_OK:9540`, the canary was
exact, and a fresh arithmetic/checksum semantic probe passed. All four
qualification-time environments showed the inherited GLM model variables
explicitly empty.

## GLM-5.2 EXL3 3.5-bpw fixed-MTP4 TP4/DCP4 (`R7`)

| Attribute | Value |
| --- | --- |
| Hardware | Four directly cabled NVIDIA DGX Sparks |
| Image | `sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513` |
| Model revision | `9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Checkpoint identity | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Parallelism | TP4/DCP4, all-gather/reduce-scatter backend (`ag_rs`), interleave one |
| Speculation | Fixed MTP4, greedy draft sampling |
| KV representation | Dynamic `nvfp4_ds_mla`, FP8 RoPE |
| KV allocation | 9,250,000,000 bytes/rank |
| Model limit | 262,144 tokens |
| Graphs | `FULL_AND_PIECEWISE`, CUDA graph capture sizes of 1 through 40 query rows (`Q1` through `Q40`) |
| Cache root | `/cache/sparkcache-glm52-r7` |

Startup registered 101 persistent layers: 79 target-KV layers and 22 sparse-
indexer layers. It discovered 10 compatible manifests before the store gate
and 11 after restart.

The qualification's initial GLM start failed closed because receipts for the
40-query-row exact graph (`Q40`) already existed in the shared JIT roots.
Those four files were moved to hash-preserving backup names. The restarted
profile generated and attested fresh receipts and captured every graph size
from 1 through 40 query rows (`Q1` through `Q40`).

A 225,555-token deterministic prompt committed 225,536 reusable tokens on
every rank. After a coordinated restart and fresh 40-query-row exact-graph
receipt generation,
SparkCache restored the entry in:

| DCP rank | Restore time |
| ---: | ---: |
| 0 | 4,358.8 ms |
| 1 | 3,180.1 ms |
| 2 | 3,386.2 ms |
| 3 | 3,694.6 ms |

The scheduler reported an external hit for all 225,536 reusable tokens.
The model returned exactly `SPARKCACHE_OK:9540`. A short request that published
all-rank cache inventory to the scheduler (the quorum prime) returned `2`, and
the post-restore canary returned `SPARKCACHE_CANARY_OK`. A fresh semantic probe correctly
computed both arithmetic results and selected `recompute` for a checksum
mismatch. GLM's default high-reasoning mode required a 1,024-token completion
budget for that short structured probe; 256 tokens ended before assistant
content was emitted.

## Conclusion and limits

The connector-configuration and Docker environment-removal implementations are
qualified for these exact DeepSeek and GLM deployments. The evidence proves
durable store, engine-restart discovery, all-rank external restore, exact
semantic output, continued generation, bounded capacity reporting, and clean
handling of inherited image environment state.

This record does not qualify DeepSeek opaque pages exposed through vLLM's
hybrid-memory-allocator (HMA)
API at DCP2/DCP4, another checkpoint or vLLM source contract, streaming
snapshots, native restore, or a GLM image rebuilt from different inputs.
