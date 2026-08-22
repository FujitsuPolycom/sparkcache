# SparkCache

SparkCache is a persistent, rank-local NVMe context cache for vLLM's
KV-Connector-V1 interface. Each tensor-parallel worker stores the KV state it
owns and restores that state from its own disk after an engine restart. Cache
traffic does not cross the network.

Distribution status for version `0.1.0a1`: **qualified**. Its immutable
package artifacts are available from
[PyPI](https://pypi.org/project/sparkcache/0.1.0a1/) and are identified by
[`MULTI_MODEL_LIVE_VALIDATION.md`](MULTI_MODEL_LIVE_VALIDATION.md). The wheel
and source distribution match the artifacts built by the protected release
workflow, and that exact wheel passed the DeepSeek-V4 and GLM-5.2
store/restart/restore matrix in the support table. Qualification does not
transfer to another package version, model, topology, or vLLM source contract.
Other combinations are implemented, research-only, or unsupported as stated
in that table.

Version `0.1.0a2` is **qualified** for the GLM-5.2 TP4/DCP4 persistent
store/restart/external-restore lane recorded in
[`GLM52_A2_LIVE_VALIDATION.md`](GLM52_A2_LIVE_VALIDATION.md). Its DeepSeek
profiles remain **implemented** and have not inherited `0.1.0a1`'s live
qualification. Release notes and qualification records identify the status of
each immutable artifact.

## Installation

Install the available dependency-free storage engine and verification tooling
from PyPI:

```bash
python -m pip install sparkcache
```

Reproducing the qualified release-wheel matrix requires the exact artifact:

```bash
python -m pip install sparkcache==0.1.0a1
```

The wheel and Python source distribution contain the importable `sparkcache`
package and exact-hash vLLM lease contracts. They intentionally omit the
repository-level `patches/` and `deploy/` trees and the optional native
C++/CUDA sources. A PyPI-only installation therefore does not construct a
supported vLLM serving runtime.

The optional `connector` extra installs PyTorch for CPU-side development. It
does not install vLLM, select a CUDA build, or patch a serving environment:

```bash
python -m pip install 'sparkcache[connector]'
```

Qualified serving deployments require the repository tag whose version
matches the installed package. The checkout supplies the model-specific
deployment transformers, vLLM patches, native sources, and evidence records:

```bash
git clone --branch v0.1.0a1 --depth 1 \
  https://github.com/FujitsuPolycom/sparkcache.git
cd sparkcache
python -m pip install '.[connector]'
```

Use [`deploy/deepseek_v4/`](deploy/deepseek_v4/README.md) for the qualified
DeepSeek-V4 profiles or
[`deploy/glm52_35bpw/`](deploy/glm52_35bpw/README.md) for the qualified
GLM-5.2 profile. Those builders attest the accepted serving-image sources and
produce source-bound overlay receipts. Configure vLLM with connector module
path `sparkcache.spark_context_cache_connector`. A source deployment adds the
directory containing the `sparkcache` package to `PYTHONPATH`; it does not add
the package directory itself.

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
| DeepSeek-V4 opaque HMA pages at TP2/DCP1 | **qualified** | `MULTI_MODEL_LIVE_VALIDATION.md`; the release wheel restored 73,728 tokens in 459.8–517.0 ms per rank |
| DeepSeek-V4-Flash-0731 opaque HMA pages at TP4/DCP1 | **qualified** | `MULTI_MODEL_LIVE_VALIDATION.md`; the release wheel restored 73,728 tokens in 413.9–494.6 ms per rank |
| DeepSeek-V4 HMA pages at DCP2/DCP4 | **unsupported** | opaque page ownership and DSpark rolling-state sharding are undefined; see `deploy/deepseek_v4/DCP_SUPPORT.md` |
| GLM-5.2 EXL3 3.5-bpw per-token rows at TP4/DCP4 | **qualified** | SparkRing serving recipe `R7`; `MULTI_MODEL_LIVE_VALIDATION.md`; the release wheel restored 225,536 tokens in 3.39–3.95 seconds per rank |
| Native direct restore | **implemented** | checksum-attested adapter and CPU-testable ABI/layout gates; disabled in qualified DeepSeek profiles |
| Streaming snapshots | **research-only** | GLM-5.2 DCP4 inventory only; not profile-general and disabled for opaque block pages |
| Buddy replication | **research-only** | protocol/state machines implemented; network carrier absent |
| Qwen recurrent-state persistence | **unsupported** | no profile, record schema, or live qualification |
| Longest-stored-prefix restore for grown conversations | **unsupported** | exact full-span digests only; design work is tracked in `ROADMAP.md` |

## Qualified release matrix

The exact `0.1.0a1` wheel passed cold store, coordinated engine restart,
all-rank manifest discovery, external restore, exact semantic output, and a
fresh post-restore canary in three serving profiles. Every profile used
`--max-num-batched-tokens 4096`; a value of `8192` is unsupported until the
exact profile passes a separate smoke at that value.

| Profile | Prompt / restored tokens | Restore time by rank |
|---|---:|---:|
| DeepSeek-V4-Flash-0731 TP2/DCP1 | 73,774 / 73,728 | 459.8, 517.0 ms |
| DeepSeek-V4-Flash-0731 TP4/DCP1 | 73,774 / 73,728 | 483.9, 413.9, 443.0, 494.6 ms |
| GLM-5.2 EXL3 3.5-bpw TP4/DCP4 | 225,555 / 225,536 | 3,954.2, 3,385.0, 3,649.6, 3,722.4 ms |

See [the complete release-wheel evidence](MULTI_MODEL_LIVE_VALIDATION.md) and
the model-specific deployment instructions under `deploy/`.

## vLLM compatibility

| Source contract | Status | Construction boundary |
|---|---|---|
| `vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3` with `patches/vllm/` | **implemented** | Patch sources and exact preimage hashes are published. A standalone public runtime builder is **unsupported**; a PyPI installation does not construct or attest this runtime. |
| vLLM build `e2666d9a6` with `patches/vllm-e2666d9a6/` and `sparkcache/runtime_patches/vllm-kv-block-lease-contract-e2666d9a6.json` | **qualified** | The DeepSeek-V4 and GLM-5.2 deployment builders apply exact lineage-specific patch chains, verify postimages, and bind the results to the SparkCache source tree. |

Any other whole-file hash is **unsupported** until its ownership contract is
derived and tested.

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
