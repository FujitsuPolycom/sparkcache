# GLM-5.3 native direct-restore performance validation

Date: 2026-08-29

Status: **experimental performance result, not yet a concurrency
qualification**. This record covers the native direct-restore implementation,
one 128K request, the earlier 16K concurrency comparison, continued generation,
and GPU-free regression checks. C16 shared-prefix, mixed-trunk, unrelated-cold,
and decode-interference measurements remain required.

## Exact implementation and runtime

| Attribute | Value |
|---|---|
| SparkCache branch | `codex/hybrid-restore-phase-timing` |
| Timing implementation | `175f940` |
| Native page placement | `71f367b` |
| Multi-slab restore and exact-prefix discovery | `8e7f5fc` |
| Direct pipelined slab restore | `94c4493` |
| Full authenticated span-table bound | `9dbf73c` |
| SparkCache source-tree SHA-256 | `368cc18dbccc262a1f2a1f1eef5aced90690031abd1f2fedf3d192e60a67012b` |
| Qualified parent/runtime image | `sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290` |
| Serving topology | GLM-5.3 Flash, TP4/DCP1, one rank on each of `spark-r0` through `spark-r3` |
| Restore workers | Two host restore workers and two native placement lanes per rank |
| Native arenas | Two 256 MiB mapped-host arenas per rank |
| Restore safety | Restore only authenticated, identity-compatible extents; otherwise recompute |

The direct path reads each immutable `.spcc` object into alternating mapped-host
arenas with `pread`, computes its whole-file SHA-256 in place, validates its
extent table, and submits authenticated spans directly to the page-placement
kernel. It avoids Python `ContextChunk` reconstruction and the previous
813 MiB join/copy operation.

The span-table limit is 4,096, matching the validated native ABI maximum. The
first 128K attempt exposed the earlier 64-span adapter limit; increasing this
adapter bound does not change cache identity, digest values, chunk geometry, or
the on-disk format.

## 128K direct-restore result

Each rank restored 813,068,464 bytes through four native slabs.

| Rank | Restore service | Read and hash | Native submit | CUDA finish |
|---:|---:|---:|---:|---:|
| 0 | 141.9 ms | 77.9 ms | 25.4 ms | 3.4 ms |
| 1 | 131.3 ms | 84.6 ms | 13.4 ms | 3.4 ms |
| 2 | 139.0 ms | 92.7 ms | 14.1 ms | 3.4 ms |
| 3 | 250.1 ms | 140.7 ms | 20.1 ms | 3.4 ms |

The slowest rank completed cache service in 250.1 ms, below the 500 ms C1
target. End-to-end client latency was 0.907 seconds; that includes scheduler
work, live-token execution, and DFlash generation in addition to cache restore.
A fresh post-restore semantic canary reported `semantic_match: true`, and the
HTTP health endpoint remained 200.

Before the direct `pread` pipeline, the same 128K entry required 1.29--1.46
seconds of cache service per rank. Its dominant costs were 395--417 ms for
read/verification, 200--243 ms for Python reconstruction, 373--438 ms for the
arena copy/submission path, and 142--160 ms for CUDA completion. The direct
path removes the reconstruction and large intermediate copy from the request
critical path.

## Earlier 16K concurrency comparison

The original Python/Torch placement path developed 1.54--1.57 second submit
spikes under concurrent restore, and one restore worker amplified queue delay
to about 2.7 seconds. Eight simultaneous clients completed in 9.45--10.64
seconds.

Native placement submitted in 6--15 ms. With two restore workers and lanes,
the observed maximum queue delay fell to about 0.93 seconds and eight clients
completed in approximately 1.2--2.1 seconds. This is strong evidence that the
Python/Torch page-placement section caused the earlier serialized stall, but
it is not a completed C16 qualification.

## 128K concurrency baseline

After rolling the scheduler recovery correction in `b00c6d4`, a synchronized
fixture reproduced persisted digest `53d5e0f5fe6b...`. Every request restored
131,072 tokens and 813,068,464 bytes per rank. The baseline predates
single-flight cohort sharing, so logs record one complete restore per request:
2 for C2, 8 for C8, and 16 for C16 on every rank.

| Cohort | Cache state | Client min | Client p50 | Client p95/max |
|---:|---|---:|---:|---:|
| C2 | first read after restart | 1.282 s | 1.282 s | 1.377 s |
| C8 | host/filesystem warm | 0.947 s | 2.017 s | 3.129 s |
| C16 | host/filesystem warm | 1.063 s | 3.363 s | 5.335 s |

All 26 requests succeeded. No load failure, placement failure, engine death,
container exit, or semantic-canary failure was observed. C2 per-rank cache
service ranged from 283.9 to 564.7 ms, with queue wait below 0.4 ms. This
baseline demonstrates that native placement removed the earlier Python submit
stall, while also quantifying the remaining duplication: C16 still reads,
verifies, and places the same snapshot sixteen times per rank.

Committed receipts:

- `evidence/glm53-flash-dflash7-bf16/native-128k-c2-b00c6d4.json`;
- `evidence/glm53-flash-dflash7-bf16/native-128k-c8-hot-b00c6d4.json`;
- `evidence/glm53-flash-dflash7-bf16/native-128k-c16-hot-b00c6d4.json`;
- `evidence/glm53-flash-dflash7-bf16/post-b00c6d4-semantic.json`.

### Initial single-flight result and retention diagnosis

The first single-flight tracer bullet proved restore coalescing but initially
left fifteen requests deferred because no runnable request remained to wake the
scheduler. After adding a verified-leader completion wake edge, C16 completed,
but each follower started another external restore: p50 was 4.740 seconds and
the maximum was 8.252 seconds. The run was correct but slower than the baseline
and is not a qualification result.

vLLM metrics showed zero local-prefix hit tokens and 100% external-prefix hits.
A CPU differential probe initially implicated the GLM hybrid-cache retention
policy: a minimal Mamba+EAGLE fixture with retention zero lost its common hit,
while a positive aligned interval retained it. That result did not generalize
to the full GLM topology. A live run with
`--prefix-cache-retention-interval 18432` still recorded zero local-prefix hit
tokens and sixteen external restores. The retention setting is therefore not
part of the profile contract.

The remaining boundary is explicit HMA block ownership: at least one live
cache group is not rediscoverable through the common hash lookup after the
leader advances. The next implementation attaches followers directly to the
verified leader's rank-local block table through vLLM's existing block-pool
reference accounting. It must occur only after all-worker restore completion
and before leader reclamation.

The first explicit hot-lease run (`d30cdea`) also rejected capture safely.
GLM's Mamba manager carries internal checkpoint blocks, so its authoritative
partial-tail tuple can name a different valid table slot than the simple
`span // block_size` page. The initial assertion treated that valid topology
as disagreement. All sixteen requests fell back to verified external restore
and completed (p50 4.786 seconds, maximum 7.893 seconds); no lease became
visible. The corrected implementation validates and follows vLLM's
authoritative tuple when present, while retaining physical-page arithmetic as
the fallback when no tuple exists. Optional lease rejections now emit one-line
warnings rather than exception tracebacks.

Diagnostic receipts:

- `evidence/glm53-flash-dflash7-bf16/singleflight-128k-c16-830a117.json`;
- `evidence/glm53-flash-dflash7-bf16/post-singleflight-semantic-830a117.json`;
- `evidence/glm53-flash-dflash7-bf16/singleflight-retention18432-128k-c16.json`;
- `evidence/glm53-flash-dflash7-bf16/post-retention18432-semantic.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-d30cdea-128k-c16.json`;
- `evidence/glm53-flash-dflash7-bf16/post-hotlease-d30cdea-semantic.json`.

## Repository validation

The exact source tree above passed:

- `python -m pytest sparkcache -q`: 634 passed, 4 skipped;
- `python -m pytest deploy -q`: 92 passed, 1 skipped;
- `python -m ruff check sparkcache deploy`: all checks passed.

## Required concurrency qualification

The next live matrix records request latency, per-rank phase timing, restore
queue depth, GPU-memory stability, preemptions, health, and continued unrelated
decode progress for:

1. C2, C8, and C16 requests sharing the same 128K prefix;
2. C16 requests sharing several large trunks but having different tails;
3. C16 hot host-resident prefixes;
4. C16 unrelated cold 128K prefixes.

Targets are below 500 ms for one shared 128K restore, 500--1,000 ms for mostly
shared trunks, below 500 ms for hot host prefixes, and stable 2--5 second
service for unrelated cold prefixes without OOMs or scheduler stalls.

## Recovery correction

The rejected first 128K attempt also exposed a pinned-vLLM recovery defect:
`_update_requests_with_invalid_blocks` unpacks the result of
`get_block_ids(req_id)` as though every request has one KV-cache group. GLM-5.3
has multiple groups, so that recovery path raises `ValueError: too many values
to unpack`.

The da4d7be image recipe now applies an exact-preimage scheduler correction.
When any HMA group contains an invalid restored block, it discards the complete
external prefix for that request and recomputes it; partially verified hybrid
state is never published. The expanded eight-file vLLM source contract pins
both this scheduler behavior and `KVCacheManager.get_block_ids()`. GPU-free
tests execute the patched method against multi-group, disjoint, asynchronous,
and preserved single-group recovery cases. A rebuilt thin image and an injected
live restore rejection are still required before production qualification.
