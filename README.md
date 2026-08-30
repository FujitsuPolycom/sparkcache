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
64 MiB page objects, and bounded shared-prefix attachment.

## Qualified outcomes

Qualification belongs to the exact artifact, model, topology, vLLM source, and
bounded workload named by its evidence record.

| Qualified deployment | Bound demonstrated | Recorded outcome | Evidence |
|---|---:|---|---|
| `sparkcache==0.1.0a1`, DeepSeek-V4 TP2/DCP1 and TP4/DCP1 | 73,728 restored tokens | 413.9–517.0 ms cache service per rank | [release-wheel validation](MULTI_MODEL_LIVE_VALIDATION.md) |
| `sparkcache==0.1.0a2`, GLM-5.2 TP4/DCP4 | 225,536 restored tokens | 3.17–4.17 s cache service per rank | [package validation](GLM52_A2_LIVE_VALIDATION.md) |
| GLM-5.3 TP4/DCP1 source runtime at `da4d7be6` | 131,072 tokens, C1 | 131–250 ms cold SparkCache CUDA restore per rank; 104–165 ms host-warm | [CUDA restore validation](GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md) |
| Same GLM-5.3 source runtime | 131,072-token prefix, C16 | one 813 MB restore per rank instead of 16; standard-chat client p50 3.363 s → 2.980 s | [CUDA restore and concurrency validation](GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md) |
| Exact local GLM-5.3 page-tail/CUDA image | 256K restore and C16 shared exact prefix | 128K→256K page tail used 13 authenticated delta objects; 16 distinct request tails shared one restored 128K `block_pages_v1` prefix | [ed60 validation](https://github.com/FujitsuPolycom/sparkring/pull/147) |

C1, C8, and C16 mean one, eight, and sixteen concurrent requests. Client
latency and cache-service time are different measurements; the evidence records
keep them separate.

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
| Tail-only opaque-page deltas | **qualified** | Byte-correct `sparkcache-page-delta-manifest/v2` publication and restore on local image `ed60…`; latency and write-volume bounds are not established |
| 64 MiB flat page macro objects | **implemented** | `sparkcache-page-snapshot-manifest/v2` at SparkCache `90946fd6`; GPU-free tested, not live qualified; [review evidence](https://github.com/FujitsuPolycom/sparkcache/pull/40) |
| SparkCache CUDA restore and placement | **qualified** | Exact GLM-5.3 TP4/DCP1 source artifacts in the linked records |
| Shared exact-prefix GPU blocks | **qualified** | Up to 16 waiting followers in the recorded GLM-5.3 runtime |
| Different-root shared row segments | **implemented** | Authenticated `per_token_rows` descriptor-prefix sharing; GPU-free tested, not live qualified |
| Streaming snapshots | **research-only** | GLM-5.2 DCP4 inventory; disabled for opaque page profiles |
| Buddy replication | **research-only** | Protocol and receiver state exist; no network carrier is included |

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

Conversation extensions use `sparkcache-hybrid-page-delta/v1` page semantics
and `sparkcache-page-delta-manifest/v2` roots. The delta binds the exact base,
layout, block counts, and recurrent or sliding boundary. It reuses only
byte-identical pages, including the correct replacement of a boundary that
falls inside a physical page. Delta and flat payload objects are at most
64 MiB; version 1 delta roots remain readable. At most two deltas form one
graph before compaction publishes a fresh flat root.

Tail-only opaque-page publication is **qualified** for byte-correct behavior on
the exact local `ed60…` image using `sparkcache-page-delta-manifest/v2`; the
[evidence record](https://github.com/FujitsuPolycom/sparkring/pull/147) binds
its sources and limits. Flat macro publication is **implemented** and
GPU-free tested but not live qualified. Arbitrary earlier opaque-page aliases
are **unsupported** because truncating an encoded snapshot does not create a
valid earlier context.

### Shared GPU prefixes and row segments

Concurrent requests for one persistent digest coalesce around one restore.
After all workers succeed, patched vLLM may retain the verified multi-group
block table as a bounded shared-prefix lease. Each follower owns ordinary block
references and computes a request-private GPU tail. A partial page is copied to
a dedicated immutable block before attachment.

The recorded runtime permits at most 16 waiting followers, keeps at most two
prefixes eligible for 15 seconds, and releases lease references under allocation
pressure. It is **qualified** through C16 for the exact local `ed60…` image:
16 distinct request tails shared one restored 128K `block_pages_v1` prefix.

For `per_token_rows`, different selected roots can also name one identical
authenticated descriptor prefix. Every rank must prove the same descriptor
sequence; a mismatch causes the affected request to recompute. Different-root
row-segment sharing is **implemented** and GPU-free tested, but no live model
artifact qualifies it.

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
| `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355` | **qualified** | GLM-5.3 HMA recovery, SparkCache CUDA restore, and shared-prefix attachment |
| `local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3` | **implemented** | Adaptive-MTP integration and ten-file lease contract; no four-rank qualification |
| `local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5` | **qualified** | Exact 31-file Python overlay over the `da4d7be6` compiled extensions on local image `ed60…`; a source-built `0b67266a` wheel is **unsupported**; [evidence](https://github.com/FujitsuPolycom/sparkring/pull/147) |

`libspark_cache_placement` is the optional C++/CUDA component. Its page ABI
uses mapped host arenas, authenticated extents, and a CUDA scatter kernel.
Loading it requires an explicit absolute path and SHA-256. SparkCache CUDA
restore authenticates complete objects before placement and verifies a flat
snapshot's complete digest before the parked request may resume.

## Qualification boundaries

- The exact local GLM-5.3 page-tail/CUDA artifact is image ID
  `sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8`,
  built from SparkCache `65b6642` and SparkRing
  `d93cb3d98305041081cf572521602625185112ae`; its
  [evidence record](https://github.com/FujitsuPolycom/sparkring/pull/147)
  is the qualification authority.
  It does not qualify a published OCI digest, response quality, general
  restore latency, or write endurance.
- Flat `sparkcache-page-snapshot-manifest/v2` objects at SparkCache `90946fd6`
  are **implemented** and GPU-free tested, not live qualified. Their 13-object
  813,068,464-byte geometry is a
  format result, not a latency claim.
- More than 16 waiting followers, C24/C32 cohorts, unrelated-cold C16 behavior,
  and decode interference are outside the qualified bounds. Existing
  measurements for the latter two are **research-only**.
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
