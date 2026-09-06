# SparkCache research map

This page collects ideas and unsupported designs. Implemented behavior belongs
in the package documentation. Model-specific test gaps belong with the matching
profile under [`deploy/`](deploy/).

- **research-only** means code or a bounded experiment exists, but no supported
  deployment profile uses it.
- **unsupported** means SparkCache has no safe interface for the behavior.

Open correctness defects are listed in [`DEFECTS.md`](DEFECTS.md).

## Implemented features that need wider live testing

| Feature | Implemented behavior | Useful live tests |
|---|---|---|
| Copy-on-write publication | Row tails and physical-page deltas use immutable authenticated graphs and separate namespaces. | Repeated conversation growth, published-byte accounting, corruption recovery, and more page geometries. |
| SparkCache CUDA restore | An attested library places authenticated objects into request-owned GPU blocks. | More model layouts, cancellation, allocation pressure, and bounded-memory checks. |
| Shared bases and GPU prefixes | Bounded requests can share authenticated stored bases and verified GPU blocks. | Mixed prompt lengths, leader cancellation, allocation pressure, and sustained concurrency. |

These features remain **implemented**. The open work expands the deployments
and workloads for which live evidence exists.

## Research-only ideas

### Per-entry retention

Per-entry expiry or temporary session pins could complement root-wide TTL and
LRU. Every pin would need an expiry or explicit release.

### Trunk-aware eviction

Frequency and shared-byte metadata could preserve heavily reused prompt trunks
while evicting private suffixes. This metadata must never decide restore
eligibility.

### Restore prefetch

A digest-based warm-up could begin bounded reading and verification before
scheduling. A request would still count as a hit only after every rank confirms
the verified result.

### Streaming publication

The planner, lease registry, gather ring, journal, and progress runtime are
implemented and GPU-free tested. Each storage layout still needs a tensor
inventory, byte-exact translator, ownership proof, and live interference test.

Streaming remains off unless a deployment profile explicitly enables a
matching runtime, library, and layout.

### Buddy-replication carrier

The replication package implements framing, credits, retries, expiry, and
reconnect state. It does not include a network adapter.

Possible carriers include TCP and RDMA-capable transports. Replication is for
repair; ordinary restore remains rank-local.

### Heat-aware storage controls

Possible signals include hit count, recomputation avoided, shared bytes,
exclusive bytes, publication budgets, logical write volume, and device
endurance data.

These signals may guide publication and eviction. They must not make an
unverified entry eligible for restore.

### Lossless compression

A compressed object must authenticate its encoded form and decode to the exact
original bytes. Adoption needs useful space savings without unacceptable CPU
or restore cost.

## Unsupported without a profile

- Recurrent state for an unregistered storage layout.
- Opaque pages whose parallel ownership is undefined.
- CUDA placement for an unregistered destination layout.
- Streaming publication for an unregistered tensor inventory.

Supporting one of these requires an explicit profile, a distinct cache
namespace when storage meaning changes, GPU-free tests, and a matching live
test record.

## Unsupported by design

- Lossy compression or quantized archival state.
- Approximate or blended cross-position reuse.
- Network storage as the ordinary rank-local cache path.
- Driver-specific process checkpoints as portable context records.
- Reusing one cache identity across different physical shard layouts.
- Making inference wait for publication, replication, eviction, or telemetry.
