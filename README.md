# SparkCache

SparkCache is a persistent, rank-local NVMe context cache for vLLM's
KV-Connector-V1 interface. Each tensor-parallel worker stores the KV state it
owns and restores that state from its own disk after an engine restart. Cache
traffic does not cross the network.

Distribution workflow: **implemented**. Version `0.1.0a1` builds as a checked
wheel and source distribution; publication to PyPI has not been performed. The
exact DeepSeek-V4 and GLM-5.2 configurations in the support table are
**qualified** on DGX Spark development systems. Other model, topology, and
vLLM combinations are implemented, research-only, or unsupported as stated in
that table.

## Installation

After version `0.1.0a1` is published, install it from PyPI:

```bash
python -m pip install sparkcache==0.1.0a1
```

The base package contains the dependency-free storage engine and tooling. A
vLLM serving environment already provides its compatible CUDA build, PyTorch,
and the exact vLLM revision required by the selected deployment profile. The
optional `connector` extra installs PyTorch for CPU-side development but does
not install or select vLLM:

```bash
python -m pip install 'sparkcache[connector]==0.1.0a1'
```

Configure vLLM with the installed connector module path
`sparkcache.spark_context_cache_connector`. Source deployments likewise add
the directory containing the `sparkcache` package to `PYTHONPATH`; they do not
add the package directory itself.

## System contract

SparkCache provides these invariants:

- **Fail-closed restore.** Content-addressed chunks are SHA-256 verified before
  restored blocks are released to inference. Missing, incompatible, or corrupt
  state becomes a cache miss and recompute.
- **Manifest-last durability.** Immutable chunks are fsynced before an atomic,
  fsynced manifest makes an entry discoverable.
- **Physical-rank quorum.** The scheduler admits an external prefix only after
  every tensor-parallel worker reports a compatible manifest from its current
  process generation.
- **Topology-bound identity.** Checkpoint digests, model layout, TP/DCP degree,
  physical rank, chunk geometry, record schema, and page-reuse policy determine
  the storage namespace. Incompatible configurations miss instead of aliasing.
- **Bounded optional work.** Store, restore, maintenance, streaming publication,
  and replication failures do not block unrelated inference. Native ownership
  failures remain fatal when continuing could expose or reuse blocks still read
  by CUDA.
- **Rank-local capacity policy.** High/low watermarks, optional TTL expiry,
  manifest-recency LRU eviction, and orphan collection account physical bytes
  across every identity namespace under one cache root.

## Implemented interfaces

| Path | Interface |
|---|---|
| `sparkcache.spark_context_cache_connector` | vLLM scheduler/worker connector, quorum, asynchronous store/restore, hybrid-memory-allocator (HMA) page handling, and capacity reporting |
| `sparkcache/spark_context_cache_config.py` | immutable connector configuration and cache-identity construction |
| `sparkcache/persistent_context_cache/cache_manifest.py` | content-addressed chunk and manifest store |
| `sparkcache/spark_context_cache_profiles.py` | named model-layout and geometry contracts |
| `sparkcache/spark_context_cache_native_placement.py` | checksum-attested native placement adapter |
| `sparkcache/streaming/` | bounded write-behind planner, block leases, native gather ring, journal, and progress runtime |
| `sparkcache/replication/` | carrier-independent buddy-replication protocol and receiver state machine |
| `sparkcache/runtime_patches/` | exact-hash vLLM ownership contracts |
| `deploy/deepseek_v4/` | DeepSeek-V4 checkpoint, overlay, launch, semantic, capacity, and corruption gates |
| `deploy/glm52_35bpw/` | GLM-5.2 EXL3 3.5-bpw TP4/DCP4 inspection-to-launch tooling |
| `deploy/deployment_contract/` | model-neutral inspection, command, port, source identity, overlay receipt, container launch, and semantic-gate mechanics |

## Support status

| Capability | Status | Evidence or limitation |
|---|---|---|
| Per-token row storage at TP1/TP2/TP4 and DCP1/DCP2/DCP4 | **implemented** | GPU-free topology and round-trip matrix; DCP must divide TP and profile chunk geometry |
| DeepSeek-V4 opaque HMA pages at TP2/DCP1 | **qualified** | `MULTI_MODEL_LIVE_VALIDATION.md`; 73,728-token restore completed in 549–571 ms per rank |
| DeepSeek-V4-Flash-0731 opaque HMA pages at TP4/DCP1 | **qualified** | `MULTI_MODEL_LIVE_VALIDATION.md`; 294,912-token restore completed in 1.88–2.30 seconds per rank |
| DeepSeek-V4 HMA pages at DCP2/DCP4 | **unsupported** | opaque page ownership and DSpark rolling-state sharding are undefined; see `deploy/deepseek_v4/DCP_SUPPORT.md` |
| GLM-5.2 EXL3 3.5-bpw per-token rows at TP4/DCP4 | **qualified** | SparkRing serving recipe `R7`; `MULTI_MODEL_LIVE_VALIDATION.md`; 225,536-token restore completed in 3.18–4.36 seconds per rank |
| Native direct restore | **implemented** | checksum-attested adapter and CPU-testable ABI/layout gates; disabled in qualified DeepSeek profiles |
| Streaming snapshots | **research-only** | GLM-5.2 DCP4 inventory only; not profile-general and disabled for opaque block pages |
| Buddy replication | **research-only** | protocol/state machines implemented; network carrier absent |
| Qwen recurrent-state persistence | **unsupported** | no profile, record schema, or live qualification |
| Longest-stored-prefix restore for grown conversations | **unsupported** | exact full-span digests only; design work is tracked in `ROADMAP.md` |

## Qualified DeepSeek-V4 TP4 result

The 2026-08-21 four-Spark qualification used
`deepseek-ai/DeepSeek-V4-Flash-0731`, vLLM
`0.1.dev1+ge2666d9a6.d20260810`, TP4/DCP1, DSpark speculation with five
draft tokens per step (`K5`), and
`fp8_ds_mla` KV pages.

| Prompt tokens | Restored tokens | Restore time per rank | End-to-end gate |
|---:|---:|---:|---:|
| 36,910 | 36,864 | 207–298 ms | 1.56 s |
| 73,774 | 73,728 | 335–412 ms | 1.64 s |
| 147,502 | 147,456 | 734–888 ms | 2.45 s |

The gate required four-rank manifest discovery, current-generation quorum,
nonzero external-prefix hit metrics, byte-identical semantic output, and an
immediate exact sentinel. Capacity and disposable-copy corruption gates also
passed. See [the complete evidence record](DEEPSEEK_V4_TP4_LIVE_VALIDATION.md).

## Qualified GLM-5.2 3.5-bpw result

The GLM-5.2 EXL3 3.5-bpw SparkCache composition, identified by SparkRing as
serving recipe `R7`, is qualified on four Sparks at TP4/DCP4. The recorded gate
covered durable store, coordinated runtime restart, four-rank restore, and
mixed cached/uncached concurrency.
See [the live evidence record](GLM52_DCP4_HISTORICAL_VALIDATION.md) and
[the deployment instructions](deploy/glm52_35bpw/README.md).

## vLLM compatibility

SparkCache supports two exact-hash vLLM source contracts:

- `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3` with the
  patches in `patches/vllm/`;
- vLLM build `e2666d9a6` with the lineage-specific patches in
  `patches/vllm-e2666d9a6/` and the lease contract
  `sparkcache/runtime_patches/vllm-kv-block-lease-contract-e2666d9a6.json`.

Any other whole-file hash is rejected until its ownership contract is derived
and tested.

## Validation

The default suite is GPU-free; vLLM is stubbed and torch runs on CPU:

```bash
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m ruff check .
```

CUDA 13 is required only to build and execute the optional native libraries
under `sparkcache/native/`.

`DEPLOYMENT_CONTRACT_PARITY_VALIDATION.md` records byte-identical inspection
and Docker-command results across eight model-specific source inspections and
six live rank inspections from four-Spark and two-Spark development systems.

## License

Apache-2.0.
