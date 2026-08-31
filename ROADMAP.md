# Research-only and unsupported work

This document records work outside SparkCache's implemented, supported
interface. Deployment-specific qualification gaps belong in the corresponding
profile under [`deploy/`](deploy/).

- **research-only** means a design, component, or bounded experiment exists,
  but no supported deployment contract covers it.
- **unsupported** means no safe interface exists or configuration is rejected.

Open correctness and scaling defects are listed in [`DEFECTS.md`](DEFECTS.md)
and take priority over feature work.

## Qualification work for implemented features

### Copy-on-write publication

**Status: implemented; broader qualification is research-only.** Row tails and
physical-page deltas use distinct cache namespaces and authenticated immutable
graphs.

Remaining qualification work includes repeated conversation extensions,
publication-byte and SSD-write accounting, corruption recovery, wider
concurrency, and additional page geometries.

### SparkCache CUDA restore

**Status: implemented; profile expansion is research-only.** Direct placement
requires a checksum-attested library and a deployment profile that describes
every destination group and semantic reuse boundary.

Each additional profile must prove byte identity, continued-generation
equivalence, cancellation behavior, bounded memory, and requester isolation.

### Shared persistent bases and GPU prefixes

**Status: implemented; broader concurrency is research-only.** Sharing is
bounded by authenticated descriptor graphs, follower limits, retained-prefix
limits, and lease expiry.

Further qualification must measure mixed prompt lengths, allocation pressure,
cancellation, leader failure, and sustained high concurrency.

## Research-only features

### Per-entry retention controls

The implemented TTL and LRU policy applies to a cache root. Per-entry metadata
could allow callers to expire or temporarily pin a session. Pins require an
expiry or explicit release so abandoned sessions cannot defeat capacity bounds.

### Trunk-aware eviction

Manifest-recency LRU treats roots independently. Frequency and shared-byte
metadata could preserve heavily reused trunks while evicting private suffixes.
This metadata must not participate in restore eligibility.

### Restore prefetch

A named-digest warm-up could begin reading and verification before scheduling.
It must use bounded asynchronous loading and must not claim a hit until every
expected rank confirms the verified result.

### Streaming snapshot generalization

The planner, lease registry, gather ring, journal, and progress runtime are
implemented and GPU-free tested. Each cache layout still needs an explicit
tensor inventory, byte-exact translator, source-ownership proof, and live
interference measurements.

Streaming remains default-off. A profile-specific gate must reject any runtime
or layout that does not match its attested contract.

### Buddy-replication carrier

The replication package implements framing, credits, idempotency, stale-
generation rejection, expiry, and reconnect state. It has no network adapter.

Candidate carriers include a minimal TCP adapter and an optional RDMA-capable
adapter. Replication remains repair-only; normal restores stay rank-local.

### Heat-aware admission and SSD controls

Potential controls include hit counters, recomputation avoided, shared-trunk
value, exclusive-byte cost, admission sketches, publication budgets, logical
write accounting, and device endurance telemetry.

These signals may influence publication and eviction. They must never make an
unverified entry eligible for restoration.

### Lossless compression

Any stored compression format must authenticate the complete encoded form and
decode to byte-identical records. Adoption requires measured space savings,
CPU cost, and restore-latency impact on representative profile artifacts.

## Unsupported without a profile contract

- Recurrent-state persistence for an unregistered model layout.
- Opaque-page reuse under a parallel geometry whose ownership is undefined.
- SparkCache CUDA placement for an unregistered destination layout.
- Streaming publication for an unregistered tensor inventory.

Support requires an explicit profile, cache namespace, state-ownership proof,
GPU-free regression coverage, and exact live evidence.

## Unsupported by design

- **Lossy KV compression or quantized archival tiers.** Approximate state
  violates byte-exact restore and continued-generation equivalence.
- **Cross-position blended reuse.** Approximate repair is not a verified
  SparkCache hit.
- **A distributed filesystem as the normal cache store.** Ordinary restore is
  rank-local; replication is limited to repair.
- **Process-level GPU checkpointing.** Driver-specific process images are not
  portable logical context records.
- **Cross-topology cache identity.** Different physical shard layouts cannot
  alias rank-local entries without a separately designed canonical format.
- **Serving-path dependence on optional cache work.** Cache pressure,
  publication, replication, or telemetry must not delay unrelated inference.
