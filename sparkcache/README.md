# SparkCache package

The `sparkcache` package implements persistent, rank-local context storage for
vLLM's KV-Connector-V1 interface. The scheduler admits reusable prefixes; each
worker stores and restores only its physical tensor-parallel shard.

The package is independent of SparkRing transport. A deployment may use a
switchless ring, a switched fabric, or ordinary Ethernet for vLLM collectives;
SparkCache reads and writes only the rank's local filesystem.

## Request lifecycle

1. The scheduler hashes the aligned prompt span with the model/layout identity.
2. Worker statistics report which digests are structurally valid on each
   physical rank and identify the worker process generation.
3. The scheduler admits an external prefix only after every physical rank in
   the tensor-parallel group reports the digest.
4. Each worker allocates request blocks, reads its local manifest and chunks,
   verifies complete encoded-chunk SHA-256 values, and installs the state.
5. Any rank-local failure publishes invalid block IDs; the patched vLLM
   scheduler discards the partial prefix and recomputes the request on every
   rank.
6. A completed prefill snapshots the aligned span and publishes immutable
   chunks before the manifest visibility edge.

## Module interfaces

| Module | Interface |
|---|---|
| `spark_context_cache_connector.py` | `SparkContextCacheConnector`; scheduler admission, worker I/O, quorum, hybrid-memory-allocator (HMA) groups, capacity, and vLLM callbacks |
| `spark_context_cache_config.py` | `ConnectorConfig`; validated immutable settings, topology, and cache-identity construction shared by scheduler and worker roles |
| `spark_context_cache_profiles.py` | `ModelProfile`; storage mode, record schema, chunk geometry, and deployment validation |
| `spark_context_cache_codec.py` | DCP row ownership, record packing, and digest helpers for per-token storage |
| `spark_context_cache_hybrid.py` | opaque HMA page encoding and topology validation |
| `persistent_context_cache/cache_manifest.py` | `ManifestStore`; durable publication, lookup, restore, invalidation, and maintenance |
| `spark_context_cache_native_placement.py` | `NativePlacementAdapter`; attested CUDA placement transaction |
| `spark_context_cache_native_restore.py` | bounded read/hash/slab orchestration for native placement |
| `streaming/factory.py` | scheduler and worker adapters for write-behind publication |
| `replication/` | carrier-independent transaction protocol; no network adapter is implemented |

`sparkcache.spark_context_cache_store` is the stable manifest-store import. It
re-exports the durable store interface without introducing another storage
implementation.

## Configuration

The complete `--kv-transfer-config` argument enables SparkCache. Omitting it
leaves the connector unloaded.

```json
{
  "kv_connector": "SparkContextCacheConnector",
  "kv_connector_module_path": "sparkcache.spark_context_cache_connector",
  "kv_role": "kv_both",
  "kv_load_failure_policy": "recompute",
  "kv_connector_extra_config": {
    "spark_cache_root": "/cache/sparkcache-model",
    "spark_cache_model_profile": "deepseek-v4-fp8-hma",
    "spark_cache_target_checkpoint_sha256": "<64 lowercase hex characters>",
    "spark_cache_draft_policy": "colocated_target",
    "spark_cache_store": true,
    "spark_cache_restore": true,
    "spark_cache_max_bytes": 214748364800,
    "spark_cache_low_watermark_bytes": 193273528320,
    "spark_cache_ttl_seconds": 0
  }
}
```

The target digest must identify immutable checkpoint contents, not a mutable
path or tag. A separately loaded drafter requires draft policy `separate` and
its own `spark_cache_draft_checkpoint_sha256`. Policy `colocated_target`
derives the draft identity from the target checkpoint.

The model profile is part of cache identity. Implemented profiles are defined
in `spark_context_cache_profiles.py`; an unknown profile or unsupported
TP/DCP/PP geometry fails startup.

## Storage and integrity

`ManifestStore` publishes files in this order:

1. encode, hash, and fsync each immutable chunk;
2. link each chunk into the content-addressed chunk directory and fsync it;
3. encode and fsync the manifest;
4. atomically link the manifest into its identity namespace and fsync the
   manifest directory.

Startup discovery validates identity, geometry, descriptors, and referenced
chunk existence/size without reading every payload. Restore is the payload
integrity boundary: selected chunks are read and hashed before state is
released. `sweep_integrity()` is an explicit payload-reading diagnostic.

Persistent data never contains CUDA pointers, allocator block tables,
physical slot coordinates, or transport sequence numbers.

## Capacity policy

`spark_cache_max_bytes` is a high watermark for filesystem-allocated bytes
under one cache root. Crossing it triggers manifest-recency LRU eviction down
to `spark_cache_low_watermark_bytes`, which defaults to 90% of the high
watermark. `spark_cache_ttl_seconds` expires manifests by recency; zero
disables TTL.

Maintenance counts shared chunks once, preserves any chunk referenced by a
surviving manifest, removes invalid/expired manifests, and collects orphan
chunks and incomplete manifest debris. The watermark is a post-commit
reclamation target, not a preallocation reservation. One publisher per
rank-local root is qualified; multiple publishers can transiently exceed the
watermark until maintenance reconciles physical use.

## Restore timing

Every asynchronous restore emits one compact JSON record prefixed with
`spark-context-cache-restore-timing:`. Schema
`sparkcache-restore-timing/v1` separates time waiting for a load worker from
manifest lookup, chunk read/verification/decoding, state reconstruction,
device-transfer submission, and CUDA-stream synchronization. It also reports
the selected span, chunk count, encoded hybrid-page bytes, service time, and
enqueue-to-completion time.

The record is diagnostic only. Missing or malformed timing observations do not
change whether cached state is accepted: SparkCache restores verified state or
recomputes the request. The legacy human-readable total remains available for
existing log consumers. That legacy total includes the best-effort manifest
recency touch after successful restoration; the structured `service_ms` ends
when placement completes and intentionally excludes that bookkeeping.

## Optional paths

- **Native direct restore — implemented.** Requires the checksum-attested
  `libspark_cache_placement` artifact and remains disabled unless the launch
  supplies its path and SHA-256.
- **Streaming snapshots — research-only.** The state machines are bounded and
  tested, but the registered tensor inventory and translator are specific to
  GLM-5.2 DCP4.
  The deployment contract uses a source-mounted vLLM lease-contract path;
  wheel-only streaming is unsupported. Opaque HMA page profiles reject
  streaming.
- **Buddy replication — research-only.** Transaction, credit, idempotency,
  expiry, and reconnect state are implemented. No socket, NIXL, or RDMA
  carrier is included.

## Support evidence

DeepSeek-V4 HMA pages are qualified at TP2/DCP1 and TP4/DCP1. DCP2/DCP4 HMA
pages are unsupported. GLM-5.2 EXL3 3.5-bpw per-token rows, identified by
SparkRing serving recipe `R7`, are qualified at TP4/DCP4. Per-token row storage
also has GPU-free coverage at TP1, TP2, and TP4 with DCP degrees that divide TP
and chunk geometry.

See:

- `../DEEPSEEK_V4_LIVE_VALIDATION.md` for TP2/DCP1 evidence;
- `../DEEPSEEK_V4_TP4_LIVE_VALIDATION.md` for TP4/DCP1 evidence;
- `../GLM52_DCP4_HISTORICAL_VALIDATION.md` for the GLM-5.2 EXL3 3.5-bpw
  TP4/DCP4 evidence;
- `../deploy/deepseek_v4/DCP_SUPPORT.md` for the HMA DCP limitation; and
- `../ROADMAP.md` for research-only and unsupported work.

## Validation

```bash
python -m pytest sparkcache -q
python -m ruff check sparkcache
```

The Python suite is GPU-free. Native CUDA execution requires a CUDA 13 build
from `native/CMakeLists.txt`.
