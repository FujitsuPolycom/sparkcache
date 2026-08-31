# SparkCache package

The `sparkcache` package implements persistent, rank-local context storage for
vLLM's KV-Connector-V1 interface. The scheduler selects reusable prefixes;
each worker stores and restores only its physical tensor-parallel shard.

SparkCache is independent of the collective transport. A deployment may use
any vLLM-compatible network topology while SparkCache reads and writes the
worker's local filesystem.

## Request lifecycle

1. The scheduler hashes each eligible aligned prompt boundary in one pass.
2. Workers advertise structurally valid digests and their process generation.
3. The scheduler selects the longest digest present on every expected rank.
4. Each worker reads and authenticates its local manifest and payload objects.
5. Workers place restored state only after validation succeeds.
6. Any rank-local failure rejects the shared prefix and recomputes the request.
7. A completed prefill publishes immutable objects before its manifest.

Row-oriented storage may also publish authenticated aliases over an exact
manifest. Alias failure does not change exact-manifest eligibility.

## Module interfaces

| Module | Interface |
|---|---|
| `spark_context_cache_connector.py` | Scheduler admission, worker I/O, all-rank agreement, capacity, and vLLM callbacks |
| `spark_context_cache_config.py` | Validated settings, topology, and cache-identity construction |
| `spark_context_cache_profiles.py` | Storage mode, record schema, geometry, and deployment validation |
| `spark_context_cache_codec.py` | Row ownership, record packing, and digest helpers |
| `spark_context_cache_hybrid.py` | Opaque manager-page encoding and topology validation |
| `persistent_context_cache/cache_manifest.py` | Publication, lookup, restore, invalidation, and maintenance |
| `spark_context_cache_cuda_placement.py` | Checksum-attested SparkCache CUDA placement transaction |
| `spark_context_cache_cuda_restore.py` | Bounded read, verification, and placement orchestration |
| `spark_context_cache_restore_timing.py` | Structured asynchronous restore timing records |
| `streaming/` | Default-off write-behind publication components |
| `replication/` | Carrier-independent replication protocol |

`sparkcache.spark_context_cache_store` is the stable manifest-store import.
It re-exports the durable interface without creating another implementation.

## Configuration

Supply the connector through vLLM's `--kv-transfer-config`. Omitting the
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
    "spark_cache_store": true,
    "spark_cache_restore": true,
    "spark_cache_max_bytes": 214748364800,
    "spark_cache_low_watermark_bytes": 193273528320,
    "spark_cache_ttl_seconds": 0,
    "spark_cache_clear_once": ""
  }
}
```

The target digest identifies immutable checkpoint contents, not a mutable path
or tag. A separately loaded drafter uses policy `separate` and supplies
`spark_cache_draft_checkpoint_sha256`.

The model profile is part of cache identity. An unknown profile or unsupported
parallel geometry rejects connector startup. Profile documents provide exact
model settings and supported topology.

## Publication schemas

`spark_cache_publication_schema` defaults to `snapshot-v1`.

| Storage mode | Operator value | Persisted behavior |
|---|---|---|
| Row-oriented | `tail-cow-v1` | Immutable row tails and authenticated descriptor chains |
| Opaque manager pages | `tail-cow-v1` | Physical-page deltas and authenticated base graphs |

Each non-default schema uses a distinct cache namespace and therefore misses
entries published under `snapshot-v1`. Streaming-snapshot deployments reject
copy-on-write publication.

## Storage and integrity

`ManifestStore` publishes files in this order:

1. encode, hash, and fsync each immutable object;
2. link objects into the content-addressed directory and fsync it;
3. encode and fsync the manifest;
4. atomically link the manifest into its identity namespace and fsync it.

Startup discovery validates identity, geometry, descriptors, alias chains, and
referenced object size. Restore reads and hashes selected payloads before state
is released. `sweep_integrity()` performs an explicit full-payload diagnostic.

Persistent data never contains CUDA pointers, allocator block tables, physical
slot coordinates, or transport sequence numbers.

## Prefix reuse

### Longest stored exact prefix

The scheduler generates wire-compatible digests at every eligible boundary and
selects the longest digest advertised by every expected rank. A growing prompt
can therefore reuse an earlier stored boundary without a full-prompt match.

### Sparse row-prefix aliases

Row-oriented storage can publish bounded metadata over existing immutable
objects. Exact manifests take precedence over aliases with the same digest.
The metadata is authenticated before it participates in restore.

### Copy-on-write tails

Row-tail publication stores only rows after the selected earlier boundary. A
partial terminal object is replaced rather than modified.

Page-tail publication reuses only byte-identical complete pages. A boundary
inside a recurrent page replaces that page and retains earlier pages. The
authenticated graph is bounded to two deltas before compaction emits a complete
snapshot.

### Concurrent sharing

One leader may restore a persistent digest while bounded followers wait for the
same verified result. After every rank succeeds, followers attach through vLLM
block references. Allocation pressure releases retained leases before ordinary
serving allocation fails.

## SparkCache CUDA restore

SparkCache CUDA restore is opt-in. It requires the path and SHA-256 of a
compatible `libspark_cache_placement` artifact plus a supported mapped-host
arena size.

The restore transaction authenticates input objects, validates their logical
positions, places bytes into request-owned GPU destinations, and resumes the
parked request only after CUDA completion succeeds. Any error releases the
request for recomputation.

See [`native/README.md`](native/README.md) for the ABI and memory-ordering
contract. Deployment profiles state where GPU execution has been qualified.

## Capacity policy

`spark_cache_max_bytes` is a high watermark for allocated bytes under one
cache root. Crossing it triggers manifest-recency LRU eviction down to
`spark_cache_low_watermark_bytes`, which defaults to 90% of the high watermark.

`spark_cache_ttl_seconds` expires manifests by recency; zero disables TTL.
Maintenance preserves shared objects referenced by surviving roots and removes
invalid roots, expired roots, orphan objects, and incomplete publications.

One publisher per rank-local root is qualified. Multiple publishers may
temporarily exceed the watermark until maintenance reconciles physical use.

## One-shot cache clear

Set `spark_cache_clear_once` to a deliberate operator token when a deployment
must discard the configured rank-local cache contents before reuse:

```json
"spark_cache_clear_once": "storage-layout-reset-2026-08-31"
```

SparkCache locks the exact root and removes only its owned object, manifest,
alias, and index directories. It preserves the root, lock, completion markers,
unknown children, and sibling paths.

After durable removal, SparkCache writes a completion marker derived from the
token. Reusing that token is a no-op; a different token requests another clear.
A lock timeout or filesystem failure disables caching for that connector while
model serving continues.

The option is an operator action, not a cache-identity field. The root must be
absolute, non-broad, and free of symlinked components.

## Diagnostics

Each asynchronous restore emits a JSON record prefixed with
`spark-context-cache-restore-timing:`. Schema `sparkcache-restore-timing/v1`
separates queue wait, lookup, read and verification, reconstruction, device
submission, synchronization, and total service time.

Timing records are diagnostic only. Missing telemetry cannot make cached state
eligible or ineligible.

## Optional research paths

- **Streaming snapshots — research-only.** Default-off state machines are
  implemented, but each supported tensor inventory requires a separate
  deployment contract and live qualification.
- **Buddy replication — research-only.** Transaction, credit, retry, expiry,
  and reconnect state exist. No socket, NIXL, or RDMA carrier is included.

See [`../ROADMAP.md`](../ROADMAP.md) for research and unsupported scope.

## Model profiles and evidence

Model names, checkpoint identities, topology, launch commands, measurements,
and qualification records live under [`../deploy/`](../deploy/). A green
GPU-free suite establishes package behavior, not live serving qualification.

## Validation

```bash
python -m pytest sparkcache -q
python -m ruff check sparkcache
```

The Python suite is GPU-free. SparkCache CUDA execution requires a compatible
CUDA build from [`native/CMakeLists.txt`](native/CMakeLists.txt).
