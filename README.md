# SparkCache

> [!WARNING]
> **Alpha research software.** Pin exact artifacts and use SparkCache for
> evaluation. APIs, cache formats, runtime contracts, and supported profiles
> may change.

SparkCache is a persistent, rank-local NVMe context cache for vLLM's
KV-Connector-V1 interface. It reuses verified prefills across requests and
process restarts and can select the longest compatible stored prompt prefix.

Each tensor-parallel worker stores only its physical rank's model state.
Normal cache reads and writes remain on that rank's local filesystem.

## GLM-5.3 upstream runtime and artifacts

GLM-5.3 serving depends primarily on Local Inference Lab's
[Jovian Judgement vLLM work](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
and [B12X](https://github.com/local-inference-lab/b12x) Blackwell kernels.

The public ARM64 route uses the
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
target and the BF16
[`GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410)
external draft. These artifacts are not interchangeable with the separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

## Capabilities

| Capability | Status | Evidence boundary |
|---|---|---|
| Content-addressed persistent snapshots | **implemented** | Immutable objects, manifest-last publication, verified restore, capacity control |
| Longest stored exact-prefix selection | **implemented** | One incremental token-digest pass; longest all-rank candidate wins |
| Sparse row-prefix aliases | **implemented** | Authenticated `per_token_rows` metadata; GPU-free regression coverage |
| Complete opaque manager-page snapshots | **implemented** | Named GLM and DeepSeek profiles only |
| Immutable row tails | **implemented** | GPU-free coverage; no live model-serving qualification |
| Physical-page delta publication | **implemented** | Bounded exact GLM-5.3 TP4 evidence; not generally qualified |
| SparkCache CUDA page restore | **implemented** | Exact source-runtime qualification and public-image C4 smoke |
| Different-root shared-base reads | **implemented** | Exact C8 TP4 evidence; one authenticated base read per rank |
| Shared GPU exact-prefix blocks | **qualified** | Exact GLM-5.3 source runtime through C16 |
| Streaming snapshots | **research-only** | GLM-5.2 DCP4 inventory; disabled for opaque pages |
| Buddy replication | **research-only** | Protocol state exists; no network carrier is included |
| Cross-topology or heterogeneous-TP reuse | **unsupported** | Identity remains topology- and physical-rank-bound |

## Start with the public GLM-5.3 ARM64 image

The canonical public image is **implemented and TP4 smoke-verified, not
generally qualified**. Pull it by immutable digest:

```bash
export GLM53_IMAGE='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5'
docker pull "$GLM53_IMAGE"
```

Model checkpoints are not embedded. Continue with the
[GLM-5.3 Jovian Judgement r7 GB10 TP4 quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_JJ_R7_GB10_TP4_QUICKSTART.md).
The [image record](deploy/glm53_flash/JJ_R7_ARM64_IMAGE.md) pins source,
model, image, label, and bounded C4 smoke identities.

## Supported scope

| Runtime | Topology | Status | Start here |
|---|---|---|---|
| Public Jovian Judgement r7 ARM64 + SparkCache | 4 GB10 systems · TP4/DCP1 | **implemented; TP4 C4 smoke-verified** | [Public image record](deploy/glm53_flash/JJ_R7_ARM64_IMAGE.md) |
| GLM-5.3 exact source runtime | 4 GB10 systems · TP4/DCP1 | **qualified for its named artifact and workloads** | [SparkCache CUDA restore record](GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md) |
| GLM-5.2 EXL3 3.5-bpw | 4 GB10 systems · TP4/DCP4 | **qualified for named package artifacts** | [Deployment guide](deploy/glm52_35bpw/README.md) |
| DeepSeek-V4-Flash-0731 | 2 or 4 GB10 systems · DCP1 | **qualified for named package artifacts** | [Deployment guide](deploy/deepseek_v4/README.md) |

Artifact-specific historical evidence remains in its named record. It is not a
substitute for the public-image quickstart.

## Architecture and invariants

- Cache identity binds checkpoint content, model layout, topology, physical
  rank, record schema, chunk geometry, draft policy, and page-reuse policy.
- Identity, compatibility, all-rank availability, and payload integrity must
  all pass before restored state reaches inference.
- Any unresolved restore becomes a cache miss and normal recomputation.
- Immutable objects are committed before an atomic, fsynced manifest exposes
  an entry.
- Optional cache work must not delay unrelated serving.
- Persistent files never contain CUDA pointers, allocator block tables, or
  transport sequence numbers.

## Documentation

| Topic | Document |
|---|---|
| Public GLM-5.3 ARM64 image and bounded smoke | [`JJ_R7_ARM64_IMAGE.md`](deploy/glm53_flash/JJ_R7_ARM64_IMAGE.md) |
| GLM-5.3 connector and deployment profiles | [`deploy/glm53_flash/README.md`](deploy/glm53_flash/README.md) |
| Physical-page deltas and shared-base evidence | [`GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md`](GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md) |
| SparkCache CUDA restore and shared GPU prefixes | [`GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md`](GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md) |
| Interactive prefix and publication explorer | [`docs/sparkcache-prefix-explainer.html`](docs/sparkcache-prefix-explainer.html) |
| Package interfaces and configuration | [`sparkcache/README.md`](sparkcache/README.md) |
| Research and unsupported work | [`ROADMAP.md`](ROADMAP.md) |
| Open correctness defects | [`DEFECTS.md`](DEFECTS.md) |

## Install

Install the published package and connector dependencies:

```bash
python -m pip install 'sparkcache[connector]==0.1.0a2'
```

PyPI `0.1.0a2` has package-level GPU-free validation. Live serving status is
artifact-bound. The public ARM64 image above contains source features that are
not represented by the PyPI package version alone.

For repository development:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git
cd sparkcache
python -m pip install -e '.[test,lint]'
```

## Test and contribute

```bash
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m ruff check .
```

Behavioral changes require GPU-free regression coverage. Changes to cache
identity, digest salts, or persisted geometry must create clean misses against
incompatible entries. See [`CONTRIBUTING.md`](CONTRIBUTING.md).

## License

Apache-2.0. See [`LICENSE`](LICENSE).
