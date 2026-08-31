# Research-only and unsupported work

This document records work that is not part of SparkCache's implemented
interface. Status labels have these meanings:

- **research-only** — a design or evidence exists, but deployment support is
  not qualified; accepted design directions without implementation also use
  this status and state their prerequisites;
- **unsupported** — no safe interface exists; configuration fails closed;
  designs incompatible with SparkCache's invariants are explicitly marked
  unsupported by design.

Open correctness and scaling defects are listed in `DEFECTS.md` and take
priority over feature work.

## Research-only design work

### Tail-publication performance qualification

**Status: implemented; general qualification remains research work.** Longest
exact-boundary search, authenticated row-prefix aliases, immutable row tails,
and authenticated block-page deltas are implemented. Tail publication is
opt-in through
`spark_cache_publication_schema=tail-cow-v1`, which selects distinct row and
page cache identities while leaving the default snapshot identity unchanged.

GPU-free coverage proves copy-on-write extension, bounded page-delta
compaction, recurrent/sliding boundary geometry, corruption removal,
reference-aware maintenance, and verified reconstruction. The exact PR535
GLM-5.3 TP4 record covers a 98,304-to-131,072-token delta restart and eight
concurrent persistent 16,384-token restores sharing one base read per rank.
Production qualification still requires repeated conversation extensions,
publication-byte and SSD-write accounting, corruption recovery, and broader
concurrency and geometry coverage.

### Per-entry retention controls

**Status: research-only.** The implemented TTL and LRU policy applies to an entire
cache root. Per-entry metadata would allow a caller to set a TTL and pin a
known session against pressure eviction. Pins must have an expiry or explicit
release operation so abandoned sessions cannot permanently defeat capacity
bounds.

### Trunk-aware eviction

**Status: research-only.** Prefix aliases and cross-root row-trunk sharing are
implemented. Manifest-recency LRU still treats every root independently.
Frequency and reference-value metadata could allow eviction to prefer
unshared suffixes and preserve chunks referenced by many conversation trunks.
Such metadata must remain outside authenticated restore state and cannot make
an entry eligible for restoration.

### Restore prefetch

**Status: research-only.** A named-digest warm-up operation would start read and
verification before vLLM schedules the request. It must reuse the bounded
asynchronous-load machinery without claiming an external hit until all ranks
confirm completion.

### SparkCache CUDA restore expansion

**Status: implemented; expansion qualification remains research work.**
SparkCache CUDA multi-group page restore is source-runtime-qualified for the
recorded GLM-5.3 TP4/DCP1 full-snapshot profile. Direct page-delta placement,
authenticated shared-base reads, and bounded eight-lane restore have exact TP4
evidence in `GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md`. MTP with
SparkCache, C16 page-delta restore, soak behavior, live fault injection, and
other model or topology contracts remain unqualified.

DeepSeek-V4 opaque HMA pages retain their verified Python restore path. CUDA
support for that profile must describe all five page groups, preserve each
group's semantic reuse window, and prove byte identity before changing its
qualified deployment contract.

## Research-only

### Streaming snapshot generalization

The streaming planner, lease registry, gather ring, journal, and progress
runtime are implemented and GPU-free tested. The registered tensor inventory
and translator are specific to GLM-5.2 DCP4. Qualification for another model
requires an explicit registered-layer inventory, byte-exact translation, CUDA
ownership tests, and interference measurements.

The model-serving qualification requirement sets cache-active time-to-first-token
and decode throughput within 2% of the cache-off profile under the same workload.

### Buddy-replication carrier

The replication package implements transaction framing, credit limits,
idempotency, stale-generation rejection, expiry, and reconnect state. It has no
network adapter. Possible carriers are:

- an asyncio TCP adapter for minimal dependencies and transparent failures;
- NIXL for UCX/RDMA, POSIX, and storage backends where its dependency cost is
  acceptable.

The supported use case is repair of a missing or corrupt rank-local shard from
a replica. Normal restores remain rank-local.

### Lossless chunk compression

Measure zstd ratio and CPU cost on real target-KV, sparse-indexer, and draft
records before defining a format. Any accepted format must authenticate the
complete encoded representation and preserve byte-exact decoded records.

## Unsupported

### DeepSeek-V4 HMA pages at DCP2/DCP4

Opaque page ownership and DSpark rolling-state sharding are undefined for DCP
degrees above one. `deploy/deepseek_v4/DCP_SUPPORT.md` states the required
wire-format, ownership, and qualification work. Configuration fails closed.

### Qwen recurrent-state persistence

No model profile, record schema, state ownership contract, or live evidence
exists. Supporting Qwen requires identifying every recurrent state tensor and
proving its lifecycle under prefix reuse and restart.

## Unsupported by design

- **Lossy KV compression or quantized archival tiers.** Approximate state
  violates byte-exact restore and continued-generation equivalence.
- **Cross-position blended reuse.** Repairing approximate KV from another
  position is a different cache contract and cannot be represented as a
  verified SparkCache hit.
- **A distributed filesystem as the normal cache store.** Rank-local DCP
  placement already supplies parallel local reads; a distributed filesystem
  adds inference-fabric traffic and cluster operational weight. Replication is
  scoped to repair.
- **Process-level GPU checkpointing.** Driver/runtime-specific process images
  are not portable logical context records and cannot be shared by prompt
  identity.
- **Erasure coding at small ring sizes.** Parity and reconstruction complexity
  does not provide a measured benefit over one bounded buddy replica.
- **Prefill/decode disaggregation.** The switchless deployment does not have
  spare interconnect bandwidth for continuous KV movement between serving
  pools.
