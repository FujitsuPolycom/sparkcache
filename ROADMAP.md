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

### Longest-stored-prefix restore

**Status: research-only.** Exact full-span digests do not match a conversation after
new turns extend it. A prefix-aware interface requires:

- chained per-chunk token digests so each 256-token prefix is available while
  hashing the full prompt;
- alias manifests that reference existing immutable chunks without copying
  payloads;
- descending-span admission bounded by the configured maximum context;
- tail-only publication after the longest verified prefix; and
- identity changes that make pre-alias manifests miss safely.

The implemented bounded quorum-delta protocol must carry alias additions and
withdrawals without making each scheduler report proportional to the complete
stored-prefix inventory.

### Per-entry retention controls

**Status: research-only.** The implemented TTL and LRU policy applies to an entire
cache root. Per-entry metadata would allow a caller to set a TTL and pin a
known session against pressure eviction. Pins must have an expiry or explicit
release operation so abandoned sessions cannot permanently defeat capacity
bounds.

### Trunk-aware eviction

**Status: research-only.** Prefix aliases are a prerequisite. Manifest-recency LRU treats every
entry independently. Alias reference counts would allow eviction to prefer
unshared suffixes and preserve chunks referenced by many conversation trunks.

### Restore prefetch

**Status: research-only.** A named-digest warm-up operation would start read and
verification before vLLM schedules the request. It must reuse the bounded
asynchronous-load machinery without claiming an external hit until all ranks
confirm completion.

### Hybrid-memory-allocator native restore

**Status: research-only.** DeepSeek-V4 opaque hybrid-memory-allocator (HMA)
pages use the verified Python
restore path. A native path must describe all five page groups, preserve each
group's semantic reuse window, and prove byte identity before changing the
qualified profile.

## Research-only

### Streaming snapshot generalization

The streaming planner, lease registry, gather ring, journal, and progress
runtime are implemented and GPU-free tested. The registered tensor inventory
and translator are specific to GLM-5.2 DCP4. Qualification for another model
requires an explicit registered-layer inventory, byte-exact translation, CUDA
ownership tests, and interference measurements.

The model-serving qualification gate requires cache-active time-to-first-token and decode
throughput within 2% of the cache-off profile under the same workload.

### Buddy-replication carrier

The replication package implements transaction framing, credit limits,
idempotency, stale-generation rejection, expiry, and reconnect state. It has no
network adapter. Candidate carriers are:

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
