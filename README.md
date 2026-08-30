# SparkCache

> [!WARNING]
> **Evaluation use only.** Serving support is limited to the exact artifacts,
> models, topologies, and source contracts marked **qualified** below. APIs,
> cache formats, deployment patches, and supported profiles may change.

SparkCache is a persistent, rank-local NVMe context cache for vLLM's
KV-Connector-V1 interface. It lets subsequent requests reuse the longest
verified stored prefix instead of repeating the corresponding prefill. Each
tensor-parallel worker stores and restores only the model state owned by its
physical rank.

Normal restore reads each rank's local filesystem; SparkCache does not send KV
payload over a cache network. vLLM collectives may still use Ethernet, a
switched fabric, or a switchless ring. State that cannot pass identity,
compatibility, all-rank availability, and payload-integrity checks is rejected,
and vLLM computes the request normally.

Deployment labels use TP for tensor-parallel degree and DCP for
decode-context-parallel degree.

## Implemented capabilities

| Capability | Status | Scope |
|---|---|---|
| Content-addressed persistent snapshots | **implemented** | Immutable chunks, manifest-last publication, rank-local capacity control, and verified restore |
| Longest stored exact-boundary discovery | **implemented** | One incremental token-digest pass; longest all-rank candidate wins |
| Sparse row-prefix aliases | **implemented** | Authenticated metadata over `per_token_rows`; no live serving qualification |
| Tail-only row publication | **implemented** | Opt-in `tail-cow-v1` storage for `per_token_rows`; no live serving qualification |
| Tail-only opaque-page publication | **implemented** | Authenticated `block_pages_v1` deltas; reconstruction and placement have no live serving qualification |
| Different-root row-segment sharing | **implemented** | Coalesces authenticated `per_token_rows` trunks; no live serving qualification |
| Opaque hybrid-memory-allocator page snapshots | **qualified** | Listed DeepSeek-V4 and GLM deployments only |
| SparkCache C++/CUDA page restore | **qualified** | Flat GLM-5.3 TP4/DCP1 snapshots under the recorded source deployment |
| Concurrent exact-prefix GPU reuse | **qualified** | Up to 16 waiting followers, two retained prefixes, 15-second retention; GLM-5.3 through 16 concurrent requests |
| Streaming snapshots | **research-only** | GLM-5.2 DCP4 inventory; disabled for opaque page profiles |
| Buddy replication | **research-only** | Protocol and receiver state exist; no network carrier is included |

## Verified persistent-restore model

SparkCache uses a verified-or-recompute rule. Restored blocks reach inference
only after identity, compatibility, all-rank availability, and payload-integrity
checks succeed.

If any physical rank cannot prove its shard, vLLM discards the complete external
prefix on every rank and computes the request normally. Partially verified
hybrid state is never published.

The persistent transaction has six steps:

1. The scheduler hashes eligible 256-token boundaries in one incremental pass.
2. Every physical rank advertises structurally compatible reusable entries for
   its worker-process generation.
3. The scheduler selects the longest digest advertised by every expected rank.
4. Workers read their local objects, verify encoded bytes, and place state into
   private request blocks.
5. Coordinated worker completion accepts the whole external prefix or rejects
   it on every rank.
6. A completed prefill publishes immutable chunks before an atomic, fsynced
   manifest makes the entry discoverable.

Cache identity binds checkpoint digests, model layout, TP/DCP degrees, physical
rank, chunk geometry, record schema, draft policy, and page-reuse policy.
Incompatible entries produce a cache miss.

Persistent files contain no CUDA pointers, allocator block tables, physical
slot coordinates, or transport sequence numbers.

## Artifact and qualification scope

PyPI version `0.1.0a3` has GPU-free package validation but no live serving
qualification. Live qualification is bound to an exact artifact, model,
topology, and vLLM source contract.

The public GLM-5.3 OCI artifact is qualified for its recorded 8,192-token
Python/Torch page restore. A separate source deployment qualifies SparkCache
C++/CUDA page restore at 131,072 tokens and bounded exact-prefix GPU reuse. See
[the public image record](deploy/glm53_flash/IMAGE_ANNOUNCEMENT.md),
[the GLM-5.3 validation](GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md), and
[the C++/CUDA restore record](GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md).

The public image omits model checkpoints. The C++/CUDA restore and shared-prefix
qualification belongs to a source-bound runtime without a published OCI
digest. The adaptive-MTP runtime described in the GLM-5.3 deployment guide is
implemented but requires its own four-rank qualification record.

## Quickstart for qualified deployments

Install the package artifact that matches the intended qualification evidence:

```bash
# GLM-5.2 EXL3 3.5-bpw at TP4/DCP4
python -m pip install sparkcache==0.1.0a2

# DeepSeek-V4 at TP2/DCP1 or TP4/DCP1
python -m pip install sparkcache==0.1.0a1
```

Use [the GLM-5.2 deployment guide](deploy/glm52_35bpw/README.md) or
[the DeepSeek-V4 deployment guide](deploy/deepseek_v4/README.md). Each builder
verifies its accepted serving source and emits a source-bound overlay receipt.

The DeepSeek profiles in `0.1.0a2` are **implemented** but **unqualified** for
that package artifact. DeepSeek qualification remains bound to `0.1.0a1`.

The GLM-5.3 source deployment uses this repository and
[its deployment guide](deploy/glm53_flash/README.md):

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git
cd sparkcache
python -m pip install '.[connector]'
```

The qualified 8,192-token public image is available by immutable digest:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943
```

The matching SparkRing procedure is pinned at
[`FujitsuPolycom/sparkring@6e9e3ace`](https://github.com/FujitsuPolycom/sparkring/blob/6e9e3acef62886a71531310673463972944b2b84/docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md).

The wheel contains the Python package and exact-hash vLLM source contracts. It
omits `patches/`, `deploy/`, and optional C++/CUDA sources, so a wheel alone
does not construct a supported vLLM serving runtime.

Configure vLLM with connector module path
`sparkcache.spark_context_cache_connector`. A source deployment adds the
directory containing `sparkcache/` to `PYTHONPATH`.

## Qualified models and measured results

Qualification applies only to the named artifact, model, topology, and source
contract. Detailed commands, runtime identities, and receipts remain in the
linked validation records.

The `0.1.0a1` serving receipts used `--max-num-batched-tokens 4096`. A value of
8192 is known to run, but performance and capacity at that value are outside
the package qualification.

| Artifact and deployment | Restored tokens | Cache service per rank | Evidence |
|---|---:|---:|---|
| `0.1.0a1`, DeepSeek-V4-Flash-0731 TP2/DCP1 | 73,728 | 459.8–517.0 ms | [release-wheel validation](MULTI_MODEL_LIVE_VALIDATION.md) |
| `0.1.0a1`, DeepSeek-V4-Flash-0731 TP4/DCP1 | 73,728 | 413.9–494.6 ms | [release-wheel validation](MULTI_MODEL_LIVE_VALIDATION.md) |
| `0.1.0a1`, GLM-5.2 EXL3 3.5-bpw TP4/DCP4 | 225,536 | 3.39–3.95 s | [release-wheel validation](MULTI_MODEL_LIVE_VALIDATION.md) |
| `0.1.0a2`, GLM-5.2 EXL3 3.5-bpw TP4/DCP4 | 225,536 | 3.17–4.17 s | [package validation](GLM52_A2_LIVE_VALIDATION.md) |
| Public GLM-5.3 OCI image, TP4/DCP1 | 8,192 | 156.8–171.8 ms | [public image record](deploy/glm53_flash/IMAGE_ANNOUNCEMENT.md) |
| GLM-5.3 source deployment, TP4/DCP1 | 8,192 | 147.2–194.0 ms | [GLM-5.3 validation](GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md) |

The native GLM-5.3 work changes practical restore cost across prefix sizes and
concurrency. C1, C8, and C16 mean one, eight, and sixteen concurrent requests.

Client timing includes scheduler and model work. Cache-service timing isolates
rank-local restore work. The pretokenized result is not compared with chat API
timing.

| Prefix and concurrency | Comparison | Recorded result |
|---|---|---|
| 8,192 tokens, C1 | qualified Python page restore | 147.2–194.0 ms cache service per rank |
| 16,384 tokens, C8 | Python/Torch page restore vs SparkCache C++/CUDA page restore | 9.45–10.64 s vs 1.2–2.1 s client latency |
| 131,072 tokens, C1 | reconstruction pipeline vs cold direct mapped-arena restore | 1.29–1.46 s vs 131–250 ms cache service per rank; a host-warm restore reached 104–165 ms |
| 131,072-token shared prefix, C16 | independent restores vs shared verified GPU blocks | rank-local work fell from 16 × 813 MB to 1 × 813 MB; standard-chat client p50 fell from 3.363 s to 2.980 s |
| 131,072-token shared prefix, pretokenized C16 | standalone measurement | 2.698 s client p50 and 2.701 s maximum |

The 16-request shared-prefix qualification used vLLM `--max-num-seqs 32`.
Every request succeeded and each rank performed one external restore. The
historical post-run canary accepted any content ending in
`SPARKCACHE_GLM53_OK`; its receipt proves continued generation and the marker
suffix, not exact visible output.

Two- and eight-request identical-prefix cohorts and a 16-request shared-trunk
cohort with distinct tails also passed under the recorded runtime.

## Prefix reuse

The logical cache boundary is 256 tokens. The scheduler computes wire-compatible
digests at each eligible boundary and selects the longest reusable digest
advertised by every physical rank.

An exact manifest represents one complete aligned snapshot. A grown conversation
can reuse an earlier exact snapshot when every rank still advertises that exact
boundary.

For row-oriented storage, identified by `per_token_rows`, SparkCache also
publishes authenticated sparse aliases over already durable chunk payloads.
Exact manifests take precedence over aliases with the same digest.

Default alias publication selects 4,096-token descriptor boundaries and the
source boundary, retaining at most 64 aliases. Alias graphs participate in TTL,
LRU, capacity accounting, invalidation, and orphan collection.

Opaque hybrid page storage, identified by `block_pages_v1`, encodes a complete
boundary snapshot and partitions its bytes across chunk files. Those byte
partitions are not independently usable token ranges.

Concurrent requests for one persistent digest are coalesced around one restore.
After every worker finishes, patched vLLM retains the verified multi-group block
table as a bounded shared-prefix lease.

A lease permits at most 16 waiting followers and remains eligible for 15
seconds. At most two prefixes remain retained; allocation pressure releases
lease references before denying ordinary serving allocations.

Each partial physical page is copied into a dedicated immutable block before a
lease becomes attachable. Followers use ordinary vLLM block references and
copy-on-write handling for their private tails.

## Operations

`spark_cache_max_bytes` is the high watermark for filesystem-allocated bytes
under one rank-local root. Maintenance evicts least-recently-used roots down to
`spark_cache_low_watermark_bytes`.

`spark_cache_ttl_seconds` expires roots by recency; zero disables TTL. Capacity
accounting counts shared chunks and alias segments once and preserves every
object referenced by a surviving root.

Each asynchronous restore emits one compact `sparkcache-restore-timing/v1` JSON
record. It separates queue wait, manifest lookup, read and verification,
reconstruction, device submission, and CUDA synchronization.

Timing collection is diagnostic. Missing timing data cannot make cached state
eligible. A restore either supplies verified state or causes request
recomputation.

Repeated restores do not rewrite KV payload files. A successful restore may
update manifest recency metadata at most once per minute.

Publishing a reusable context writes immutable payload objects and a manifest.
The default `snapshot-v1` schema writes a complete grown snapshot. The opt-in
`tail-cow-v1` schema writes immutable row tails or authenticated opaque-page
deltas and periodically compacts bounded delta graphs. SparkCache must not be
described as having negligible SSD wear under either schema.

Operators should monitor the NVMe Data Units Written counter and compare its
daily change with workload publication volume. Device endurance depends on
context size, unique publication rate, retention, and storage amplification.

SparkCache does not expose hourly write budgets, daily write budgets, or a
physical-write-amplification estimate. Those controls are **unsupported**.

## vLLM source contracts and native components

SparkCache verifies whole-file hashes and required symbols before accepting a
patched vLLM source tree. A different hash is unsupported until its ownership
and recovery behavior are derived and tested.

| vLLM source contract | Status | Scope |
|---|---|---|
| `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3` with `patches/vllm/` | **implemented** | Exact patch inputs are published; no standalone public runtime builder is provided |
| vLLM build `e2666d9a6` with `patches/vllm-e2666d9a6/` | **qualified** | DeepSeek-V4 and GLM-5.2 builders verify source, patch, and postimage hashes |
| `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355` with `patches/vllm-da4d7be/` | **qualified** | GLM-5.3 HMA recovery, SparkCache C++/CUDA restore, and bounded shared-prefix attachment |
| `local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3` with `patches/vllm-e10536a/` | **implemented** | Adaptive-MTP integration and ten-file lease contract; no four-rank qualification |
| `local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5` with `patches/vllm-glm53-b12x-kda-adaptive-mtp/` | **implemented** | Adaptive MTP, live-tensor B12X KDA, and eleven-file runtime contract; no four-rank qualification |

The GLM-5.3 contract at
[`vllm-kv-block-lease-contract-da4d7be.json`](sparkcache/runtime_patches/vllm-kv-block-lease-contract-da4d7be.json)
pins ten vLLM files and the symbols used for ownership, copying, and recovery.

`libspark_cache_placement` provides the optional C++/CUDA placement path. Its
page ABI uses mapped host arenas, authenticated copy spans, and a CUDA scatter
kernel for opaque hybrid pages.

Native loading requires an explicit library path and SHA-256. CUDA 13 builds
run a GPU-free byte-exact reference test and a CUDA hybrid-page probe before
model-serving qualification.

SparkCache's C++/CUDA restore path reads `.spcc` objects into alternating
mapped arenas, hashes complete files in place, validates authenticated extents,
and overlaps read work with CUDA submission into vLLM-owned cache pages.

## Repository map and development validation

| Path | Responsibility |
|---|---|
| `sparkcache/spark_context_cache_connector.py` | scheduler admission, worker I/O, all-rank availability, restore coalescing, shared-prefix coordination, and vLLM callbacks |
| `sparkcache/persistent_context_cache/cache_manifest.py` | exact manifests, row-prefix aliases, immutable chunks, lookup, invalidation, capacity, and garbage collection |
| `sparkcache/spark_context_cache_native_hybrid_restore.py` | authenticated direct reads, slab planning, and mapped-arena page placement |
| `sparkcache/spark_context_cache_restore_timing.py` | machine-readable asynchronous restore timing |
| `sparkcache/runtime_patches/` | exact-hash vLLM source contracts and GPU-free patch execution tests |
| `sparkcache/native/` | C++/CUDA ABI, parser, reference implementation, kernel, and probes |
| `deploy/deepseek_v4/` | DeepSeek-V4 build, launch, capacity, corruption, and semantic procedures |
| `deploy/glm52_35bpw/` | GLM-5.2 TP4/DCP4 inspection, build, launch, and semantic procedures |
| `deploy/glm53_flash/` | GLM-5.3 image, connector, benchmark, publication, and source-verification tools |
| `evidence/glm53-flash-dflash7-bf16/` | immutable GLM-5.3 request, runtime, recovery, and concurrency receipts |

Run the GPU-free repository checks with:

```bash
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m ruff check sparkcache deploy
```

CUDA execution requires a CUDA 13 build from
[`sparkcache/native/CMakeLists.txt`](sparkcache/native/CMakeLists.txt).

## Limitations and research work

Tail-only publication for `per_token_rows` is **implemented** with GPU-free
regression coverage and no live model-serving qualification.
The opt-in `tail-cow-v1` publication schema writes only immutable replacement
and extension chunks after an all-rank reusable boundary. It uses a distinct
cache namespace. Default `snapshot-v1` deployments retain their existing wire
identity and full-snapshot publication behavior.

Tail-only publication for `block_pages_v1` is also **implemented** with
GPU-free regression coverage and no live model-serving qualification. The
page-semantic `sparkcache-hybrid-page-delta/v1` codec binds
the exact base snapshot and recurrent/sliding boundary and reuses only
byte-identical opaque pages. Restore reconstructs and verifies the complete
snapshot before Python or native page placement. Arbitrary earlier-prefix
aliases cannot be derived from opaque page snapshots.

Opaque HMA snapshots cannot be shortened by truncating chunk lists. SparkCache
therefore uses the page-semantic format and distinct namespace described above.
At most two page deltas may form one graph; the following publication compacts
the context into a fresh flat snapshot. Live GLM latency and write-volume
qualification for this path remains outstanding.

Sparse row-prefix aliases are **implemented** but have no live model-serving
qualification. Their behavior is covered by GPU-free publication, discovery,
restore, capacity, and corruption regressions.

The GLM shared-prefix runtime is qualified through 16 concurrent requests under
`--max-num-seqs 32`. Cohorts of 24 or 32 requests and more than 16 waiting
followers are **unsupported** by qualification evidence.

The unrelated-cold 16-request matrix and decode-interference measurement are
**unqualified**. The shared-prefix measurements do not establish those workload
bounds.

Cross-topology or heterogeneous-TP reuse is **unsupported**. Persistent identity
is bound to topology and physical rank; no canonical cross-shard format exists.

DeepSeek-V4 opaque HMA pages at DCP2 or DCP4 are **unsupported** because page
ownership and rolling-state sharding are undefined for those layouts.

Streaming snapshots and buddy replication remain **research-only**. Qwen
recurrent-state persistence and network cache backends are **unsupported**.

## License and support

SparkCache is licensed under Apache-2.0. See [`LICENSE`](LICENSE) for the full
terms.

Report defects and compatibility requests through the
[SparkCache issue tracker](https://github.com/FujitsuPolycom/sparkcache/issues).

Include the package or source revision, vLLM source contract, model profile,
topology, and relevant receipt paths.
