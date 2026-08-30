# SparkCache package

The `sparkcache` package implements persistent, rank-local context storage for
vLLM's KV-Connector-V1 interface. The scheduler admits reusable prefixes; each
worker stores and restores only its physical tensor-parallel shard.

The package is independent of SparkRing transport. A deployment may use a
switchless ring, a switched fabric, or ordinary Ethernet for vLLM collectives;
SparkCache reads and writes only the rank's local filesystem.

## Request lifecycle

1. The scheduler hashes every eligible aligned prompt boundary in one
   incremental pass and selects the longest reusable digest.
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
7. Row-oriented storage may publish authenticated sparse aliases over the
   durable exact manifest. Alias failure does not change exact-manifest success.

## Module interfaces

| Module | Interface |
|---|---|
| `spark_context_cache_connector.py` | `SparkContextCacheConnector`; scheduler admission, worker I/O, quorum, hybrid-memory-allocator (HMA) groups, capacity, and vLLM callbacks |
| `spark_context_cache_config.py` | `ConnectorConfig`; validated immutable settings, topology, and cache-identity construction shared by scheduler and worker roles |
| `spark_context_cache_profiles.py` | `ModelProfile`; storage mode, record schema, chunk geometry, and deployment validation |
| `spark_context_cache_codec.py` | DCP row ownership, record packing, and digest helpers for per-token storage |
| `spark_context_cache_hybrid.py` | opaque HMA page encoding and topology validation |
| `persistent_context_cache/cache_manifest.py` | `ManifestStore`; exact manifests, row-prefix aliases, durable publication, lookup, restore, invalidation, and maintenance |
| `spark_context_cache_cuda_placement.py` | `CudaPlacementAdapter`; attested SparkCache CUDA placement transaction |
| `spark_context_cache_cuda_restore.py` | bounded read/hash/slab orchestration for SparkCache CUDA placement |
| `spark_context_cache_native_hybrid_restore.py` | authenticated direct reads and multi-slab mapped-arena placement for opaque HMA pages |
| `spark_context_cache_restore_timing.py` | `sparkcache-restore-timing/v1` asynchronous restore records |
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
    "spark_cache_ttl_seconds": 0,
    "spark_cache_clear_once": ""
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

`spark_cache_clear_once` is empty by default. A non-empty value requests the
one-shot root clear described below; it is an operator action token, not a
cache-identity field.

`spark_cache_publication_schema` defaults to `snapshot-v1`. Profile storage
mode `per_token_rows` maps `tail-cow-v1` to the row-tail namespace. Storage
mode `block_pages_v1` maps the same operator value to the page-delta namespace.
Both values are part of cache identity and therefore cleanly miss
`snapshot-v1` entries. Streaming-snapshot deployments reject the option.

SparkCache CUDA restore uses these optional connector settings:

- `spark_cache_cuda_restore`;
- `spark_cache_cuda_placement_library` and
  `spark_cache_cuda_placement_library_sha256`;
- `spark_cache_cuda_placement_arena_bytes`; and
- `spark_cache_cuda_restore_io_workers`.

The equivalent environment variables begin with
`SPARK_CONTEXT_CACHE_CUDA_`. Legacy `native` configuration, environment, CLI,
and profile names remain accepted as compatibility aliases. A legacy-only
configuration warns once per process. Supplying canonical and legacy values
that disagree rejects startup. Generated configurations use only the CUDA
names. The terminology change does not alter cache identity or stored bytes.

## Storage and integrity

`ManifestStore` publishes files in this order:

1. encode, hash, and fsync each immutable chunk;
2. link each chunk into the content-addressed chunk directory and fsync it;
3. encode and fsync the manifest;
4. atomically link the manifest into its identity namespace and fsync the
   manifest directory.

Startup discovery validates identity, geometry, descriptors, authenticated
prefix-alias chains, and referenced chunk existence/size without reading every
payload. Restore is the payload integrity boundary: selected chunks are read
and hashed before state is released. `sweep_integrity()` is an explicit
payload-reading diagnostic.

Persistent data never contains CUDA pointers, allocator block tables,
physical slot coordinates, or transport sequence numbers.

## Capacity policy

`spark_cache_max_bytes` is a high watermark for filesystem-allocated bytes
under one cache root. Crossing it triggers manifest-recency LRU eviction down
to `spark_cache_low_watermark_bytes`, which defaults to 90% of the high
watermark. `spark_cache_ttl_seconds` expires manifests by recency; zero
disables TTL.

Maintenance counts shared chunks and prefix-descriptor segments once, preserves
objects referenced by surviving exact or alias roots, removes invalid or
expired roots, and collects orphan objects and incomplete publication debris.
The watermark is a post-commit reclamation target, not a preallocation
reservation. One publisher per rank-local root is qualified; multiple
publishers can transiently exceed the watermark until maintenance reconciles
physical use.

## One-shot cache clear

Set `spark_cache_clear_once` to an operator-chosen token when a deployment must
discard the configured rank-local SparkCache contents before reuse:

```json
"spark_cache_clear_once": "glm53-native-layout-2026-08-29"
```

The token must contain 1--128 letters, digits, periods, underscores, colons,
at signs, plus signs, or hyphens, and must begin with a letter or digit. The
configured cache root must be an absolute, non-broad path and cannot contain a
symlinked component.

Each process takes the root's exclusive maintenance lock. If the token has no
completion marker, SparkCache removes only `chunks/`, `manifests/`,
`prefix-aliases/`, and `prefix-index/` below that exact root. It never removes
the root, `.maintenance.lock`, `.sparkcache-clear-once/`, unknown root children,
or sibling model and JIT paths.

After every owned path is durably removed, SparkCache atomically records schema
`sparkcache-clear-once/v1` under `.sparkcache-clear-once/`. The marker filename
contains a domain-separated SHA-256 of the token rather than the token text. A restart
with the same token is a no-op, even if cache entries were published after the
clear. A different token requests another clear and preserves every earlier
completion marker.

Process-local and file-lock acquisition share one 30-second budget. A lock
timeout or filesystem error leaves the requested token incomplete and disables
persistent store, restore, streaming publication, and SparkCache CUDA restore for that
connector process. Model serving can continue without persistent cache use,
and a later startup retries the same token.

The option does not change `CacheIdentity`, digest values, chunk geometry, or
storage schemas. It is a destructive operator action scoped to the configured
rank-local root; use one deliberate token across the ranks that should clear.

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

## Prefix reuse

- **Longest stored exact boundary — implemented.** The scheduler generates
  wire-compatible digests at each eligible chunk boundary and selects the
  longest digest advertised by every expected physical rank. A grown
  conversation can reuse an earlier request boundary without requiring an
  exact full-prompt match.
- **Sparse row-prefix aliases — implemented.** Storage mode `per_token_rows`
  publishes bounded `sparkcache-prefix-alias/v1` metadata over existing
  immutable chunks. Descriptor segments contain at most sixteen chunk
  descriptors; default publication selects 4,096-token boundaries and retains
  at most 64 aliases. Exact manifests take precedence over aliases with the
  same digest.
- **Immutable row tails — implemented.** Setting
  `spark_cache_publication_schema` to `tail-cow-v1` selects a distinct cache
  namespace. The scheduler selects the longest earlier prefix advertised by
  every physical rank. Workers snapshot only rows after that boundary and
  publish `sparkcache-tail-manifest/v1` metadata over authenticated descriptor
  chains and immutable replacement-tail objects. A partial terminal chunk is
  replaced, never modified. Restore rejection recomputes the request. GPU-free
  regression coverage exists; live model-serving qualification does not.
- **Opaque-page aliases — unsupported.** A `block_pages_v1` chunk is a byte
  partition of one complete HMA boundary snapshot, not an independently usable
  token range, so arbitrary earlier-prefix aliases cannot be derived from its
  chunks.
- **Immutable block-page tails — implemented.** The
  `sparkcache-hybrid-page-delta/v1` codec reuses only byte-identical page
  prefixes and binds the base snapshot, layout, block counts, and semantic
  token boundaries. A boundary inside an HMA page replaces that complete page
  while retaining earlier byte-identical pages. At an exact recurrent-page
  boundary, vLLM may retain the replay-boundary page outside the advancing
  request block table. Its `SchedulerOutput.recurrent_boundary_blocks` hand-off
  names the pinned physical block by request, group, and token boundary.
  SparkCache defers a new recurrent request until a later cached scheduler step,
  when the preceding forward's hand-off can be observed. It then requires one
  matching entry for every recurrent group whose block size exactly divides the
  publication boundary; missing or contradictory proof skips publication
  rather than scanning later running or speculative state. At a boundary inside
  a recurrent page, the request table's partial page remains authoritative and
  an unexpected override is rejected. The
  `sparkcache-page-delta-manifest/v2` schema embeds its authenticated base graph
  and groups delta bytes into immutable objects of at most 64 MiB. A
  1,575,821,491-byte delta therefore uses at most 24 physical delta objects
  instead of 1,024 objects derived from logical token chunks. Ordered restore
  batches retain at most four object payloads in addition to one assembled
  delta buffer. Version 1 manifests remain readable. Cache identity, digest
  salts, the 256-token logical boundary, and the `page-tail-cow-v1` namespace
  are unchanged. Capacity maintenance retains shared objects after predecessor
  roots are removed. Restore reconstructs the verified full snapshot before
  Python/Torch or SparkCache CUDA placement. GPU-free regression coverage
  exists; live model-serving qualification does not. Direct placement from
  base and delta extents is unsupported by this schema. A graph contains at
  most two deltas. The following extension publishes a fresh flat snapshot,
  bounding
  reconstruction work and metadata ancestry.
- **Concurrent shared GPU prefix — implemented.** One leader restores a
  persistent digest. After every rank succeeds, up to sixteen waiting followers
  attach through vLLM block references. Two leases may remain reusable for
  fifteen seconds. Partial physical pages are copied into immutable blocks
  before attachment, and allocation pressure releases lease references before
  ordinary serving allocations are denied.

## Optional paths

- **SparkCache direct CUDA restore — implemented.** Requires the checksum-attested
  `libspark_cache_placement` artifact and remains disabled unless the launch
  supplies its path, SHA-256, and a supported mapped-host arena size. The
  `glm53-flash-hybrid` profile supports authenticated multi-slab page restore;
  other block-page profiles must opt in explicitly.
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

GLM-5.3 Flash opaque pages are qualified at TP4/DCP1 with BF16 DFlash2 using
seven draft tokens. Source revision
`2b86fb9d02fa3595cca5caa864b81aedce44b8bb` qualifies SparkCache direct CUDA restore,
multi-group recovery, and shared GPU-prefix reuse through C16 under a
32-sequence scheduler ceiling. Sparse row-prefix aliases have GPU-free coverage
but no live model-serving qualification.

See:

- `../DEEPSEEK_V4_LIVE_VALIDATION.md` for TP2/DCP1 evidence;
- `../DEEPSEEK_V4_TP4_LIVE_VALIDATION.md` for TP4/DCP1 evidence;
- `../GLM52_DCP4_HISTORICAL_VALIDATION.md` for the GLM-5.2 EXL3 3.5-bpw
  TP4/DCP4 evidence;
- `../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md` for the GLM-5.3 Python
  page-placement record;
- `../GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md` for GLM-5.3 SparkCache
  CUDA restore, recovery, and C2/C8/C16 shared-prefix evidence. The historical
  filename remains stable for existing links;
- `../deploy/deepseek_v4/DCP_SUPPORT.md` for the HMA DCP limitation; and
- `../ROADMAP.md` for research-only and unsupported work.

## Validation

```bash
python -m pytest sparkcache -q
python -m ruff check sparkcache
```

The Python suite is GPU-free. Native CUDA execution requires a CUDA 13 build
from `native/CMakeLists.txt`.
