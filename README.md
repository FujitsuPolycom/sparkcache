# SparkCache

> [!WARNING]
> **Alpha research software.** Pin exact artifacts for evaluation. APIs, cache
> formats, runtime contracts, and supported deployment profiles may change.

SparkCache is a persistent, rank-local context cache for vLLM's
KV-Connector-V1 interface. It stores completed prefill state on local NVMe and
reuses the longest compatible stored prompt prefix across requests and process
restarts.

Each worker stores only its physical rank's model state. Normal cache reads and
writes stay on that rank's filesystem; SparkCache does not add network traffic
to the ordinary restore path.

## Capabilities

| Capability | Status | Scope |
|---|---|---|
| Content-addressed persistent snapshots | **implemented** | Immutable objects, manifest-last publication, verified restore, and rank-local capacity control |
| Longest stored exact-prefix selection | **implemented** | One incremental digest pass; the longest candidate present on every expected rank wins |
| Sparse row-prefix aliases | **implemented** | Authenticated metadata over row-oriented snapshots |
| Complete opaque manager-page snapshots | **implemented** | Requires an explicit deployment profile |
| Copy-on-write row and page publication | **implemented** | Uses a separate cache namespace from complete snapshots |
| SparkCache CUDA page restore | **implemented** | Requires a checksum-attested placement library and a qualified deployment profile |
| Shared persistent bases and GPU prefixes | **implemented** | Bounded sharing; qualification remains deployment-specific |
| Streaming snapshot publication | **research-only** | Default-off and restricted to explicitly registered cache layouts |
| Buddy replication | **research-only** | Protocol state exists; no network carrier is included |
| Cross-topology or heterogeneous-TP reuse | **unsupported** | Cache identity remains topology- and physical-rank-bound |

Implementation status describes repository behavior. Live qualification
belongs to an exact model, runtime, topology, artifact, and workload. See the
[deployment profiles](#deployment-profiles) for those evidence boundaries.

## Install

Install the published package and connector dependencies:

```bash
python -m pip install 'sparkcache[connector]==0.1.0a2'
```

PyPI `0.1.0a2` has package-level GPU-free validation. Live serving status is
artifact-bound. See the [package guide](sparkcache/README.md) for connector
configuration and storage behavior.

For repository development:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git
cd sparkcache
python -m pip install -e '.[test,lint]'
```

## Deployment profiles

Model-specific settings, launch commands, artifact identities, and validation
results live with their deployment profiles.

| Profile family | Start here |
|---|---|
| GLM-5.3 Flash | [`deploy/glm53_flash/README.md`](deploy/glm53_flash/README.md) |
| GLM-5.2 EXL3 3.5-bpw | [`deploy/glm52_35bpw/README.md`](deploy/glm52_35bpw/README.md) |
| DeepSeek-V4 | [`deploy/deepseek_v4/README.md`](deploy/deepseek_v4/README.md) |

The profile documents distinguish implemented behavior, exact qualification,
bounded smoke evidence, research-only work, and unsupported configurations.

## Invariants

- Cache identity binds checkpoint content, model layout, topology, physical
  rank, record schema, chunk geometry, draft policy, and page-reuse policy.
- Identity, compatibility, all-rank availability, and payload integrity must
  pass before restored state reaches inference.
- Any unresolved restore becomes a cache miss and normal recomputation.
- Immutable objects are committed before an atomic, fsynced manifest exposes
  an entry.
- Optional cache work must not delay unrelated serving.
- Persistent files never contain CUDA pointers, allocator block tables, or
  transport sequence numbers.

## Documentation

| Topic | Document |
|---|---|
| Package interfaces and configuration | [`sparkcache/README.md`](sparkcache/README.md) |
| Interactive prefix and publication explorer | [`docs/sparkcache-prefix-explainer.html`](docs/sparkcache-prefix-explainer.html) |
| Research-only and unsupported work | [`ROADMAP.md`](ROADMAP.md) |
| Open correctness defects | [`DEFECTS.md`](DEFECTS.md) |
| Security and sensitive cache data | [`SECURITY.md`](SECURITY.md) |
| Contribution requirements | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

Deployment-specific quickstarts, image digests, measurements, and evidence
records are indexed by the profile directories above.

## Test and contribute

```bash
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m ruff check .
```

Behavioral changes require GPU-free regression coverage. Changes to cache
identity, digest salts, or persisted geometry must create clean misses against
incompatible entries. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## Acknowledgements

SparkCache builds on vLLM and model-runtime work from the local inference
community. Deployment profiles identify the exact upstream runtime, kernels,
models, and quantized artifacts they use.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
