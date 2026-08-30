# SparkCache

SparkCache is a persistent, rank-local NVMe context cache for vLLM. It lets a
request restore the longest verified prompt prefix available on every required
tensor-parallel rank, then compute only the uncached tail. This turns repeated
long-context prefill into local storage work without introducing a cache
network or moving one rank's model state through another rank.

Each worker stores only the state owned by its physical rank. A restore becomes
usable only after identity, topology, all-rank availability, object integrity,
and coordinated completion agree. If that proof is incomplete, vLLM computes
the request normally. Optional cache work never becomes a serving dependency.

The [interactive prefix-reuse explorer](docs/sparkcache-prefix-explainer.html)
shows longest-prefix selection, row descriptor segments, copy-on-write tails,
64 MiB page objects, and the boundary between implemented sharing mechanisms
and qualified serving behavior.

## Qualified outcomes

Qualification belongs to the exact artifact, model, topology, vLLM source, and
bounded workload named by its evidence record.

| Qualified deployment | Bound demonstrated | Recorded outcome | Evidence |
|---|---:|---|---|
| `sparkcache==0.1.0a1`, DeepSeek-V4 TP2/DCP1 and TP4/DCP1 | 73,728 restored tokens | 413.9–517.0 ms cache service per rank | [release-wheel validation](MULTI_MODEL_LIVE_VALIDATION.md) |
| `sparkcache==0.1.0a2`, GLM-5.2 TP4/DCP4 | 225,536 restored tokens | 3.17–4.17 s cache service per rank | [package validation](GLM52_A2_LIVE_VALIDATION.md) |
| GLM-5.3 DFlash7 TP4/DCP1 source runtime at `da4d7be6` | `snapshot-v1`, 131,072 tokens, C1 | 131–250 ms cold SparkCache CUDA restore per rank; continued generation reached the recorded marker | [CUDA restore validation](GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md) |
| GLM-5.3 DFlash7 TP4/DCP1 local image `35b58a7…`, SparkCache `a1511d2` | `snapshot-v1`, `sparkcache-page-snapshot-manifest/v2`, 131,072 tokens, C1 | 13 authenticated objects; sequential object read/hash took 1.35–1.50 s, all-rank SparkCache CUDA restore took 1.55–1.70 s, and the exact codeword matched before and after restart | [CUDA restore validation](GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md) |

C1, C8, and C16 mean one, eight, and sixteen concurrent requests. At the tested
20 GiB KV-cache setting, C2×128K is an observed capacity candidate, not a
qualified SparkCache workload. C6×128K admitted one request at a time and
serialized completion over 61–313 seconds. C8×64K and C16×32K are planned and
unqualified. Sixteen independent 128K requests are unsupported at that capacity
unless they share a GPU-resident trunk or the deployment provides more KV
capacity.

## Capability status

Status words are deliberate:

- **implemented**: present in this source tree with GPU-free behavioral tests;
- **qualified**: exercised by the exact live artifact and bounded case linked;
- **research-only**: experimental work that does not define serving support;
- **unsupported**: no compatible serving contract or qualification exists.

| Capability | Status | Scope |
|---|---|---|
| Content-addressed persistent snapshots | **implemented** | Immutable objects, manifest-last publication, verified reads, capacity maintenance |
| Longest stored-boundary discovery | **implemented** | One incremental token-digest pass; longest all-rank candidate wins |
| Sparse row-prefix aliases | **implemented** | Authenticated `per_token_rows` descriptor segments; no live serving qualification |
| Tail-only row publication | **implemented** | Opt-in `tail-cow-v1`; GPU-free tested, no live serving qualification |
| Opaque HMA page snapshots | **qualified** | Exact DeepSeek-V4 and GLM deployments linked above |
| Tail-only opaque-page deltas | **research-only** | Implemented and GPU-free tested; C2 restored-delta responses failed exact semantics while recomputation succeeded |
| 64 MiB flat page macro objects | **qualified** | Exact C1 evidence remains SparkCache `a1511d2` / image `35b58a7…` with sequential reads; PR43 final head `ad8df66` implements bounded pipelined prefetch and is GPU-free tested, not live qualified |
| SparkCache CUDA restore and placement | **qualified** | Exact GLM-5.3 TP4/DCP1 source artifacts in the linked records |
| Shared exact-prefix GPU blocks | **implemented** | Bounded vLLM lease path; earlier timing receipts are diagnostic, not exact-output semantic qualification |
| Different-root shared row segments | **implemented** | Authenticated `per_token_rows` descriptor-prefix sharing; GPU-free tested, not live qualified |
| Shared opaque-page base reads | **research-only** | A 16-member host-base coalescer is implemented and GPU-free tested at SparkCache `a1511d2`; C2 multi-root restored-delta semantics failed; [review evidence](https://github.com/FujitsuPolycom/sparkcache/pull/42) |
| Streaming snapshots | **research-only** | GLM-5.2 DCP4 inventory; disabled for opaque page profiles |
| Buddy replication | **research-only** | Protocol and receiver state exist; no network carrier is included |
| Heat and SSD write-control model | **research-only** | Independent offline design in [PR #36](https://github.com/FujitsuPolycom/sparkcache/pull/36); excluded from serving and package imports |

SparkCache is alpha research software. APIs, cache formats, patches, and
supported profiles may change.

## How prefix reuse works

The logical cache boundary is 256 tokens.

1. The scheduler hashes eligible boundaries in one incremental pass.
2. Each worker advertises compatible exact roots and permitted row aliases from
   its active worker-process generation.
3. The scheduler selects the longest digest advertised by every expected
   physical rank.
4. Workers authenticate their rank-local root and objects, then place state in
   request-private blocks.
5. Coordinated completion accepts the whole external prefix on every rank or
   causes the request to recompute it everywhere.
6. Publication writes immutable payload objects before an atomic, fsynced
   manifest makes the root discoverable.

Cache identity binds checkpoint digests, model layout, TP/DCP degrees, physical
rank, logical chunk geometry, record schema, draft policy, and page-reuse
policy. Persistent files contain no CUDA pointers, allocator block tables,
physical slot coordinates, or transport sequence numbers.

## Storage and publication

### Row-oriented state

`per_token_rows` stores independently reusable rank-owned tensor rows.
Authenticated descriptor segments contain at most 16 chunk descriptors.
Sparse aliases normally select 4,096-token boundaries plus the source boundary
and retain at most 64 aliases without copying payload bytes.

The opt-in `tail-cow-v1` identity publishes immutable replacement and extension
chunks after a verified base. A partial final chunk is replaced rather than
mutated. Row aliases and row tails are **implemented** and GPU-free tested; live
model-serving qualification is absent.

### Opaque page state

`block_pages_v1` stores complete hybrid-memory-allocator page snapshots. Flat
v2 roots use `sparkcache-page-snapshot-manifest/v2` and content-addressed
objects of at most 64 MiB. An 813,068,464-byte flat snapshot therefore needs 13
payload objects rather than 512 logical-chunk files. Version 1 flat manifests
remain readable. Physical grouping does not change the 256-token identity or
admission geometry.

The qualified `35b58a7…` v2 artifact reads and hashes its 13 macro objects
sequentially. That phase measured 1.35–1.50 seconds within the 1.55–1.70-second
all-rank restore. The result is correct but slower than the identified legacy
512-object path, which used bounded parallel reads within each arena slab.

PR43 final head `ad8df66e8ff1b6680689612690fedcdd75eff175` implements bounded
flat-v2 pipelined prefetch. It authenticates the first object before parsing the
header, then reads and authenticates at most two remaining objects concurrently
into request-private host buffers before waiting for a placement arena. A batch
retains at most 256 MiB beyond the two arenas, allowing storage work to overlap
placement of the preceding batch. Only after the complete batch succeeds are
objects digested, copied into mapped arenas, and submitted in manifest order.
`spark_cache_cuda_restore_io_workers` may reduce the path to one worker; values
above two remain capped at two. Diagnostics separate read/hash, arena wait,
host copy, submission-call, and completion time. This pipeline is implemented
and GPU-free tested, not live qualified. Existing legacy roots remain readable;
the connector does not offer an operator setting that publishes additional
legacy roots.

Conversation extensions use `sparkcache-hybrid-page-delta/v1` page semantics
and `sparkcache-page-delta-manifest/v2` roots. The delta binds the exact base,
layout, block counts, and recurrent or sliding boundary. It reuses only
byte-identical pages, including the correct replacement of a boundary that
falls inside a physical page. Delta and flat payload objects are at most
64 MiB; version 1 delta roots remain readable. At most two deltas form one
graph before compaction publishes a fresh flat root.

Tail-only opaque-page publication is implemented and GPU-free tested, but its
serving status is **research-only**. In the exact DFlash7 C2 live case,
reconstructed page-delta roots completed restore yet failed the exact codeword
check; the recomputation control returned the expected codewords. Flat
`snapshot-v1` publication remains the qualified path. Arbitrary earlier
opaque-page aliases are **unsupported** because truncating an encoded snapshot
does not create a valid earlier context.

### Shared GPU prefixes and row segments

Concurrent requests for one persistent digest coalesce around one restore.
After all workers succeed, patched vLLM may retain the verified multi-group
block table as a bounded shared-prefix lease. Each follower owns ordinary block
references and computes a request-private GPU tail. A partial page is copied to
a dedicated immutable block before attachment.

The implementation permits at most 16 waiting followers, keeps at most two
prefixes eligible for 15 seconds, and releases lease references under allocation
pressure. Those limits describe the implementation, not a semantic concurrency
qualification. Earlier C16 receipts establish timing and completion only; they
did not use the exact-output codeword oracle.

For `per_token_rows`, different selected roots can also name one identical
authenticated descriptor prefix. Every rank must prove the same descriptor
sequence; a mismatch causes the affected request to recompute. Different-root
row-segment sharing is **implemented** and GPU-free tested, but no live model
artifact qualifies it.

### Shared opaque-page base reads

Concurrent `block_pages_v1` page-delta restores with independently
authenticated result roots may share one rank-local read of an identical
immutable base. Each request still authenticates its private delta,
reconstructs its own result snapshot, and performs request-private placement.
The coordinator admits at most two simultaneous flights, 16 cumulative
participants per flight, 1 GiB declared bytes per flight, and 2 GiB of peak
reservations. Followers do not occupy loader lanes while the base is pending;
an unrelated restore can use another configured lane.

The base buffer is released after every registered member acquires or abandons
it. This is bounded request-cohort I/O sharing, not a retained host-memory tier,
and it never shares mutable recurrent pages. The 16-member coordinator is
implemented and GPU-free tested at SparkCache `a1511d2`. Its serving status is
**research-only** because the exact C2 multi-root restored-delta case failed
semantic comparison while the same requests succeeded through recomputation.

## Installation and qualified entry points

Install the package artifact that matches the intended evidence:

```bash
# DeepSeek-V4 at TP2/DCP1 or TP4/DCP1
python -m pip install sparkcache==0.1.0a1

# GLM-5.2 EXL3 3.5-bpw at TP4/DCP4
python -m pip install sparkcache==0.1.0a2
```

Use the [DeepSeek-V4 guide](deploy/deepseek_v4/README.md) or
[GLM-5.2 guide](deploy/glm52_35bpw/README.md). The DeepSeek profiles included
in `0.1.0a2` are **implemented**, but DeepSeek package qualification remains
bound to `0.1.0a1`.

The public GLM-5.3 image is qualified for its recorded 8,192-token Python/Torch
page restore:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943
```

The [GLM-5.3 source guide](deploy/glm53_flash/README.md) describes
SparkCache CUDA restore and adaptive-MTP runtimes. Those source-bound artifacts
do not have a published OCI digest. PyPI `0.1.0a3` is **implemented** and
GPU-free package tested; it has no live serving qualification.

Configure vLLM with connector module
`sparkcache.spark_context_cache_connector`. Source deployments add the
directory containing `sparkcache/` to `PYTHONPATH`.

## Operations

`spark_cache_max_bytes` is the high watermark for filesystem-allocated bytes
under one rank-local root. Maintenance evicts least-recently-used roots down to
`spark_cache_low_watermark_bytes`. `spark_cache_ttl_seconds` expires roots by
recency; zero disables TTL. Capacity accounting counts shared objects once and
preserves every object referenced by a surviving root.

Each asynchronous restore emits one `sparkcache-restore-timing/v1` JSON record
covering queue wait, manifest lookup, read and verification, reconstruction,
device submission, and CUDA synchronization. Timing is diagnostic and cannot
make an entry eligible.

Released opaque-page base-read cohorts emit
`sparkcache-page-base-restore-flight/v1` summaries with authenticated base
identity, participants, physical and avoided reads, bytes, duration,
cancellations, outcome, worker generation, and storage mode. Prompt content is
not logged.

Repeated restores do not rewrite payload objects. Publication writes immutable
objects for row tails, page deltas, and periodic flat compactions.
Operators should monitor NVMe Data Units Written against publication volume.
Hourly or daily write-budget enforcement and cache-specific physical-write
amplification are **unsupported** in the serving runtime.

## Exact vLLM source contracts

SparkCache verifies accepted file hashes and required symbols before using a
patched vLLM tree. A different source is **unsupported** until its ownership,
copying, cleanup, and recovery behavior are derived and tested.

| vLLM source | Status | Contract scope |
|---|---|---|
| `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3` | **implemented** | Exact inputs in `patches/vllm/`; no standalone public runtime builder |
| vLLM build `e2666d9a6` | **qualified** | DeepSeek-V4 and GLM-5.2 builders in `patches/vllm-e2666d9a6/` |
| `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355` | **qualified** | GLM-5.3 full-snapshot HMA recovery and SparkCache CUDA restore; shared-prefix attachment is implemented but lacks exact-output concurrency qualification |
| `local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3` | **implemented** | Adaptive-MTP integration and ten-file lease contract; no four-rank qualification |
| `local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5` | **implemented** | Exact 31-file Python overlay over the `da4d7be6` compiled extensions; full-snapshot C1 is qualified on the named local artifacts, while tail-delta and multi-root concurrency remain research-only; a source-built `0b67266a` wheel is **unsupported** |

`libspark_cache_placement` is the optional C++/CUDA component. Its page ABI
uses mapped host arenas, authenticated extents, and a CUDA scatter kernel.
Loading it requires an explicit absolute path and SHA-256. SparkCache CUDA
restore authenticates complete objects before placement and verifies a flat
snapshot's complete digest before the parked request may resume.

## Qualification boundaries

- Full `snapshot-v1` opaque-page restore is **qualified** at 131,072 tokens and
  C1 for the exact GLM-5.3 DFlash7 TP4/DCP1 artifacts named in the validation
  record.
- Flat `sparkcache-page-snapshot-manifest/v2` and header-inclusive source-byte
  accounting originated at SparkCache `229d7d6`. Qualification belongs to the
  complete runtime source `a1511d26a1fe2b17b24561bc52e376bf7f54b06a`, tree
  `4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb`, and local image
  `sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e`:
  13 objects, sequential read/hash in 1.35–1.50 seconds, all-rank SparkCache
  CUDA restore in 1.55–1.70 seconds, and the exact codeword before and after
  restart. No published OCI digest is qualified.
- Bounded flat-v2 pipelined prefetch at PR43 final head `ad8df66` is
  **implemented** and GPU-free tested, not live qualified. It does not transfer
  the `35b58a7…` semantic or timing result to the PR43 artifact.
- Tail-only page deltas and opaque-page base-read cohorts are **research-only**.
  The implementation and GPU-free tests remain, but the exact C2 restored-delta
  case failed semantics while recomputation succeeded.
- The heat/write-control work in PR #36 is **research-only** and independent of
  this runtime stack. It reports hypothetical admission and write pressure; it
  does not enforce serving budgets or affect restore eligibility.
- At 20 GiB of KV cache, C2×128K is only an observed safe capacity candidate.
  C6×128K admitted one request at a time and completed serially in 61–313
  seconds. C8×64K and C16×32K are planned and unqualified. Sixteen independent
  128K requests are unsupported without GPU trunk sharing or additional KV
  capacity. Multi-root cached concurrency is not qualified at any of those
  points.
- Cross-topology and heterogeneous-TP reuse are **unsupported**. Identity is
  bound to topology and physical rank; there is no canonical cross-shard
  format.
- DeepSeek-V4 opaque pages at DCP2 or DCP4 are **unsupported** because page
  ownership and rolling-state sharding are undefined for those layouts.
- Qwen recurrent-state persistence and network cache backends are
  **unsupported**.

## Repository map and validation

| Path | Responsibility |
|---|---|
| `sparkcache/spark_context_cache_connector.py` | scheduler admission, worker I/O, restore coordination, and vLLM callbacks |
| `sparkcache/persistent_context_cache/cache_manifest.py` | manifests, aliases, immutable objects, lookup, invalidation, capacity, and garbage collection |
| `sparkcache/page_base_read_flights.py` | bounded request-cohort sharing for authenticated opaque-page base reads |
| `sparkcache/spark_context_cache_cuda_hybrid_restore.py` | SparkCache CUDA object reads, slab planning, and page placement |
| `sparkcache/runtime_patches/` | exact-hash vLLM source contracts and GPU-free patch tests |
| `sparkcache/native/` | C++/CUDA ABI, parser, reference implementation, kernel, and probes |
| `deploy/` | exact deployment builders, launchers, and verification procedures |
| `evidence/` | immutable request, runtime, recovery, and concurrency receipts |

Run the GPU-free checks with:

```bash
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m ruff check sparkcache deploy
```

CUDA execution requires a CUDA 13 build from
[`sparkcache/native/CMakeLists.txt`](sparkcache/native/CMakeLists.txt).

## License and support

SparkCache is licensed under Apache-2.0. See [`LICENSE`](LICENSE). Report
defects and compatibility requests through the
[issue tracker](https://github.com/FujitsuPolycom/sparkcache/issues), including
the package or source revision, vLLM contract, model profile, topology, and
relevant receipt paths.
