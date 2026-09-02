# SparkCache package

The `sparkcache` package implements persistent, rank-local context storage for
vLLM's KV-Connector-V1 interface. The scheduler chooses reusable prefixes;
each worker reads and writes only its physical rank's state.

## Request flow

1. The scheduler hashes each eligible prompt boundary once.
2. Workers report structurally valid entries and their process generation.
3. The scheduler chooses the longest entry present on every expected rank.
4. Each worker authenticates its local manifest and payload objects.
5. Workers place state only after every required check succeeds.
6. Any rejection becomes a normal cache miss and prompt recomputation.
7. A completed prefill publishes immutable objects before its manifest.

Row-oriented storage may also publish authenticated aliases that point to an
earlier exact manifest. A broken alias does not affect the exact manifest.

### Startup inventory

Each worker sends at most 512 discovered manifest digests through vLLM's
one-time connector handshake before API readiness. The scheduler exposes an
entry only when every physical rank reports the same digest.

Larger inventories continue through bounded delta and checkpoint reports
after the engine starts. Entries outside the startup subset recompute until
their all-rank reports arrive.

## Configure the connector

Pass SparkCache through vLLM's `--kv-transfer-config`. Omitting the connector
configuration leaves SparkCache unloaded.

```json
{
  "kv_connector": "SparkContextCacheConnector",
  "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "spark_cache_root": "/cache/sparkcache/deployment-name",
    "spark_cache_model_profile": "profile-name",
    "spark_cache_target_checkpoint_sha256": "<64 lowercase hex characters>",
    "spark_cache_draft_policy": "colocated_target",
    "spark_cache_access_mode": "read-write",
    "spark_cache_shared_prefix_lease_ttl_seconds": 15,
    "spark_cache_max_bytes": 214748364800,
    "spark_cache_low_watermark_bytes": 193273528320,
    "spark_cache_ttl_seconds": 0,
    "spark_cache_clear_once": ""
  }
}
```

The checkpoint digest identifies immutable checkpoint contents, not a mutable
path or tag. A separately loaded drafter uses policy `separate` and supplies
its own checkpoint digest.

The profile is part of cache identity. An unknown profile or unsupported
parallel geometry stops connector startup before any stored entry is used.

### Restore and publication controls

`spark_cache_access_mode` selects how the connector uses persistent storage:

| Mode | Restore stored prefixes | Publish completed prefixes |
|---|---:|---:|
| `read-write` | Yes | Yes |
| `restore-only` | Yes | No |
| `store-only` | No | Yes |
| `disabled` | No | No |

`read-write` is the default and preserves the behavior of deployments that do
not set a mode.

`restore-only` is useful when serving prompts with uncertain reuse. Existing
entries remain available, but completed requests do not capture or publish
model state. An unavailable or rejected entry is computed normally.

The independent `spark_cache_store` and `spark_cache_restore` booleans remain
supported. Each explicitly supplied boolean overrides its side of the selected
mode.

The equivalent environment variables are
`SPARK_CONTEXT_CACHE_ACCESS_MODE`, `SPARK_CONTEXT_CACHE_STORE`, and
`SPARK_CONTEXT_CACHE_RESTORE`.

Access controls do not participate in cache identity. Switching between
`read-write` and `restore-only` can reuse compatible stored entries without a
namespace change.

## Publication options

`spark_cache_publication_schema` defaults to `snapshot-v1`, which publishes a
complete aligned snapshot.

| Storage layout | `tail-cow-v1` behavior |
|---|---|
| Row-oriented | Store immutable row tails and authenticated descriptor chains. |
| Opaque manager pages | Store changed physical pages and an authenticated base graph. |

Each schema has its own cache namespace. Entries written by one schema cannot
be mistaken for entries written by another.

## Storage and integrity

`ManifestStore` writes and synchronizes immutable objects before it exposes an
atomic manifest. Startup checks identity, geometry, descriptors, alias chains,
and referenced object sizes.

Restore reads and hashes the selected payloads before releasing state.
`sweep_integrity()` performs an explicit full-payload diagnostic.

Persistent data contains no CUDA pointers, allocator block tables, physical
slot coordinates, or transport sequence numbers.

## Prefix reuse

- **Exact prefixes:** choose the longest aligned digest reported by every
  expected rank.
- **Sparse aliases:** authenticate lightweight row-boundary references to
  already stored objects.
- **Copy-on-write tails:** replace a partial terminal object and append only
  changed rows or pages.
- **Shared restores:** let bounded followers attach to one verified GPU prefix
  through ordinary vLLM block references.

Verified shared GPU prefixes remain retained for
`spark_cache_shared_prefix_lease_ttl_seconds`. The default is 15 seconds; the
maximum is 300 seconds. Longer retention can serve later queued requests
without another persistent restore.

The two-prefix limit and vLLM's memory-pressure eviction remain active
regardless of the configured duration. The equivalent environment variable is
`SPARK_CONTEXT_CACHE_SHARED_PREFIX_LEASE_TTL_SECONDS`.

Page graphs admit at most two deltas. Another extension compacts the graph into
a complete snapshot.

## SparkCache CUDA restore

CUDA restore is optional. It requires an absolute
`libspark_cache_placement` path, its SHA-256, and a compatible arena size.

SparkCache authenticates objects, checks logical positions, places bytes into
request-owned GPU blocks, and resumes the request only after CUDA completion.
Any error discards those private blocks and recomputes the prompt.

See [`native/README.md`](native/README.md) for the ABI and memory-ordering
rules. Deployment profiles record the model layouts tested with this path.

## SparkCache CUDA publication

Asynchronous manager-page capture is **implemented** and disabled by default.

It records producer readiness on the model-runner stream, gathers complete
request-owned pages on a low-priority CUDA stream, and hands a claimed mapped
ring view to the durable writer.

Ring saturation skips optional publication without waiting. Preemption
synchronizes only the affected capture before its source pages can be reused.

Enablement requires an attested `libspark_cache_snapshot` library, bounded
slot sizes, and the exact vLLM ownership contract described in
[`native/MANAGER_PAGE_CAPTURE_CONTRACT.md`](native/MANAGER_PAGE_CAPTURE_CONTRACT.md).

This path supports complete page snapshots. Page-tail publication remains a
separate implementation path.

## Capacity and cleanup

`spark_cache_max_bytes` is the high watermark for one cache root. Crossing it
evicts least-recently-used manifests down to
`spark_cache_low_watermark_bytes`, which defaults to 90% of the high watermark.

`spark_cache_ttl_seconds` expires manifests by recency; zero disables TTL.
Maintenance preserves shared objects referenced by surviving manifests.

To clear one cache root once, set `spark_cache_clear_once` to a deliberate
token:

```json
"spark_cache_clear_once": "storage-layout-reset-2026-08-31"
```

SparkCache removes only directories it owns, then writes a completion marker.
Reusing the token does nothing. A different token requests another clear.

The root must be absolute, narrow, and free of symlinked components. A lock or
filesystem failure disables caching for that connector while serving continues.

## Diagnostics

Each asynchronous restore emits compact INFO lines with restored tokens,
latency, effective token rate, bytes, and phase timings. A
`sparkcache-restore-timing/v1` JSON record with the complete phase breakdown is
available at DEBUG.

Asynchronous publication emits a `sparkcache: capture` INFO line when the
progress thread observes GPU-to-host copy completion. It reports rank, digest,
tokens, observed elapsed time, effective token rate, and copied bytes.

The separate commit log reports durable-storage time. The two records separate
capture interference from background storage work.

Each completed store also emits one compact `sparkcache: publish` line. Exact
process-local totals are available from
`ManifestStore.publication_telemetry_snapshot()` using schema
`sparkcache-publication-telemetry/v1`.

| Counter | Meaning |
|---|---|
| `logical_payload_bytes` | Encoded state represented by committed roots. A row tail or page delta counts only its extension. |
| `reused_base_bytes` | Encoded base payload referenced without staging it again. |
| `unique_object_bytes` | Complete immutable files newly linked or repaired, including metadata roots. |
| `committed_unique_object_bytes` | Newly retained immutable bytes reachable from committed roots. |
| `uncommitted_unique_object_bytes` | Immutable bytes left unreachable after an aborted or failed attempt. |
| `staged_write_bytes` | Payload bytes submitted to temporary-file writes, including later deduplication. |
| `deduplicated_bytes` | Identical immutable bytes already present at their content-addressed paths. |

The counters describe host-side operations. They do not report filesystem
allocation, NVMe Data Units Written, controller write amplification, or NAND
writes.

Telemetry is observational. It cannot change publication, restore, cache
identity, or serving decisions.

Timing is diagnostic only. Missing timing data does not change whether a
stored entry may be used.

## Package map

| Module | Purpose |
|---|---|
| `spark_context_cache_connector.py` | Scheduler decisions, worker I/O, all-rank agreement, and vLLM callbacks |
| `spark_context_cache_config.py` | Validated settings, topology, and cache identity |
| `spark_context_cache_profiles.py` | Storage layout, record schema, geometry, and profile checks |
| `persistent_context_cache/cache_manifest.py` | Publication, lookup, restore, invalidation, and maintenance |
| `spark_context_cache_cuda_placement.py` | Attested C++/CUDA placement transaction |
| `spark_context_cache_cuda_restore.py` | Bounded reading, verification, and placement |
| `streaming/` | Default-off streaming publication research |
| `replication/` | Carrier-independent replication research |

`sparkcache.spark_context_cache_store` is the stable manifest-store import.

## Profiles and tests

Model names, checkpoints, topology, launch commands, measurements, and live
test records live under [`../deploy/`](../deploy/).

```bash
python -m pytest sparkcache -q
python -m ruff check sparkcache
```

The Python suite is GPU-free. CUDA execution requires a compatible build from
[`native/CMakeLists.txt`](native/CMakeLists.txt).
