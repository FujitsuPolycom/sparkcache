# SparkCache

> [!WARNING]
> SparkCache is experimental software. Pin package, image, and source revisions
> when you need a reproducible deployment.

SparkCache gives vLLM persistent memory for previously computed prompts. It
saves reusable context to local storage.

SparkCache can restore the longest matching prefix later—even after vLLM
evicts that prefix from its GPU KV cache, or after the model process, container,
or server restarts.

Repeated conversations, shared system prompts, and growing documents can skip
much of the repeated prompt processing.

The cache follows the model's distributed layout: every worker reads and
writes only the state owned by its physical rank. Normal cache traffic stays
on local storage rather than crossing the network.

Cached model state—the data vLLM needs to continue from a saved context—is
restored only after its identity, compatibility, and contents are verified.
If those checks do not pass, vLLM computes the prompt normally.

## Capabilities

| Capability | In plain language | Status |
|---|---|---|
| Persistent snapshots | Save reusable context as immutable, verified objects. | **implemented** |
| Independent restore and publication controls | Restore existing entries without capturing new snapshots, or enable either direction separately. | **implemented** |
| Longest exact-prefix selection | Reuse the longest stored part of a prompt, not only a complete prompt match. | **implemented** |
| Sparse row-prefix aliases | Point to reusable earlier row boundaries without copying their payloads. | **implemented** |
| Complete manager-page snapshots | Preserve model-managed pages whose state is not exposed as ordinary rows. | **implemented** |
| Copy-on-write publication | Store only changed row tails or changed physical pages when extending a stored context. | **implemented** |
| SparkCache CUDA restore | Move verified page data into request-owned GPU blocks through a C++/CUDA path. | **implemented** |
| Shared bases and GPU prefixes | Read a common stored base once and let bounded concurrent requests share the restored GPU prefix. | **implemented** |
| Streaming publication | Gather completed rows while inference continues. | **research-only** |
| Buddy replication | Copy stored objects to another host for repair. The protocol exists, but no network carrier is included. | **research-only** |
| Cross-topology reuse | Reuse one stored entry across different physical shard layouts. | **unsupported** |

Status describes repository behavior. Model, runtime, topology, and live-test
details belong to the deployment profiles linked below.

## How it works

1. The scheduler hashes eligible prompt boundaries in one pass.
2. Every rank reports which matching entries it can read.
3. The scheduler chooses the longest entry available on every expected rank.
4. Each worker verifies and restores its local state.
5. If any check fails, vLLM computes the prompt normally.
6. Completed prefills publish immutable objects before exposing a manifest.

## Install

Install the published Python package:

```bash
python -m pip install 'sparkcache[connector]==0.1.0a2'
```

PyPI publishes version `0.1.0a2`. This repository declares version
`0.1.0a3`; install from a source checkout when working with that unreleased
repository version.

For repository development:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git
cd sparkcache
python -m pip install -e '.[test,lint]'
```

The package provides the connector and storage implementation. A working
model deployment also needs a compatible vLLM runtime and a deployment
profile.

## Deployment profiles

Profiles keep model-specific settings, image identities, launch commands,
measurements, and known limits out of the generic cache design.

| Model family | Guide |
|---|---|
| GLM-5.3 Flash | [Four-node SparkRing quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md) · [SparkCache integration notes](deploy/glm53_flash/README.md) |
| GLM-5.2 EXL3 3.5-bpw | [`deploy/glm52_35bpw/README.md`](deploy/glm52_35bpw/README.md) |
| DeepSeek-V4 | [`deploy/deepseek_v4/README.md`](deploy/deepseek_v4/README.md) |

## Core rules

- Cached state reaches inference only after identity, compatibility,
  all-rank availability, and payload-integrity checks succeed.
- A rejected restore becomes an ordinary cache miss and recomputation.
- Cache work must not delay unrelated serving.
- Immutable objects are written before an atomic manifest exposes an entry.
- Cache identity includes model layout, checkpoint contents, topology,
  physical rank, storage schema, and page-reuse policy.
- Persistent files never contain CUDA pointers, allocator block tables, or
  transport sequence numbers.

## Documentation

| Topic | Document |
|---|---|
| Package setup and configuration | [`sparkcache/README.md`](sparkcache/README.md) |
| CUDA placement and snapshot libraries | [`sparkcache/native/README.md`](sparkcache/native/README.md) |
| Interactive prefix explorer | [`docs/sparkcache-prefix-explainer.html`](docs/sparkcache-prefix-explainer.html) |
| Research ideas and unsupported designs | [`ROADMAP.md`](ROADMAP.md) |
| Open correctness defects | [`DEFECTS.md`](DEFECTS.md) |
| Security | [`SECURITY.md`](SECURITY.md) |
| Contributing | [`CONTRIBUTING.md`](CONTRIBUTING.md) |

## Development

```bash
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m ruff check .
```

Behavior changes need GPU-free regression tests. Changes to cache identity,
digest salts, or stored geometry must make incompatible entries miss cleanly.

## Acknowledgements

SparkCache builds on vLLM and work from the local inference community.
Deployment profiles identify the exact upstream runtime, kernels, models, and
quantized artifacts they use.

## License

Apache-2.0. See [`LICENSE`](LICENSE).
