# Historical GLM-5.3 Flash SparkCache image record

This document preserves the qualification evidence for the superseded
`sha256:cd4045b...` image. The canonical public image and run procedure are in
[`JJ_R7_ARM64_IMAGE.md`](JJ_R7_ARM64_IMAGE.md). Values below describe only the
historical artifact. Canonical public artifact identities are recorded in the
linked Jovian Judgement r7 image document.

## Status

**Experimental community derivative; author-supported ephemeral test build.**
This image is not the recommended community image and is not an official
NVIDIA, vLLM, local-inference-lab, B12X, Inco AI, Z.AI, SparkRing, or
SparkCache release.

Support owner: `FujitsuPolycom`. The image remains an ephemeral test build
until the support owner explicitly adopts a longer maintenance commitment.

GLM runtime performance and correctness come primarily from Local Inference
Lab's Jovian Judgement
[`vLLM@da4d7be6`](https://github.com/local-inference-lab/vllm/commit/da4d7be6c97434f6942292ed8abbf4b32dc44355),
with Blackwell kernels from
[`B12X@2fcf23a0`](https://github.com/local-inference-lab/b12x/commit/2fcf23a0ce269be27b2e03fece73d46e90e6aeea).
The qualified service pairs the image with
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
with the BF16
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410).
The draft is not Local Inference Lab's separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

## Image and digest

```text
ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943
```

The Linux/ARM64 image ID observed on every qualified rank was
`sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290`.
The image does not contain target-model or draft-model weights.

Recorded publication artifacts:

- runtime SPDX SBOM SHA-256
  `f5261dbed089a52923d4b2e2d5204aa889f0031ff57e324022b16c584068cd43`;
- SparkCache overlay SPDX SBOM SHA-256
  `14450bf58a8b08fd1997706de4fefc9ae4681ebe990176b6c195e296b32c9d27`;
- runtime source receipt SHA-256
  `afec2f37aa0adfea8c330f0a8e25c393c6276c3e8b8e0f855932f7f2d4ba45df`;
- standalone SLSA/Sigstore provenance attestation: `Not generated`.

## Based on

Exact parent:

```text
ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd
```

Parent image ID:
`sha256:7e8c0ebcb2001efb4cdab0ec9d20d53972e62db3688230044e22e61ffb1d35d5`.

The parent is a FujitsuPolycom SparkRing community runtime built from:

- NVIDIA CUDA
  `nvidia/cuda:13.0.3-base-ubuntu24.04@sha256:56d9d8183e2181a20be6b0d3801d1f056a0e75c17706df939ba207b126e1cb9c`;
- PyTorch ARM64 build environment
  `pytorch/manylinuxaarch64-builder@sha256:f91599c49f526c77d01b68286f2bf943a5fd6a432d7e3f0afcc5784825908fe9`;
- SparkRing runtime source revision
  `862db89b1dd905e0ce3197f1d7b64b8a5802dbf1`.

The parent retains the NVIDIA container license boundary plus Apache-2.0 and
BSD-3-Clause component notices. The overlay adds SparkCache under Apache-2.0.

## Build recipe

Public source-bound overlay files:

- [`Containerfile`](Containerfile), SHA-256
  `a2b65f3600950855cbfa00d82d532de1fbced3f4fa26c4bf1e59c3b6a519abd9`;
- [`build_public_image.py`](build_public_image.py), SHA-256
  `b72466799e2fe569ecdee3a536cfb4606d2599da0a20cd398ca03aa99e21a3e6`;
- [`PUBLISHING.md`](PUBLISHING.md) for SBOM and immutable publication.

Complete overlay build command:

```bash
git clone https://github.com/FujitsuPolycom/sparkcache.git
git -C sparkcache checkout --detach 3860a2250193a6679ac6bac857af53e0757841f8
python sparkcache/deploy/glm53_flash/build_public_image.py \
  --repository sparkcache \
  --base-image ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd \
  --output-image sparkring-glm53-sparkcache:local \
  --output glm53-sparkcache-build-receipt.json
```

The parent runtime build is public in SparkRing
[`runtime/glm53-flash/BUILD.md`](https://github.com/FujitsuPolycom/sparkring/blob/862db89b1dd905e0ce3197f1d7b64b8a5802dbf1/runtime/glm53-flash/BUILD.md).

## Source commits, pull requests, patches, and overlays

SparkCache:

- source commit `3860a2250193a6679ac6bac857af53e0757841f8`;
- deployable source-tree SHA-256
  `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`;
- VMM exemption patch
  `patches/vllm-da4d7be/020-sparkcache-vmm-exemption.patch`, SHA-256
  `370b498eebf44b4e52a2d2751fa249ad4bd3d0b6fd951b063a161fb06febbe99`;
- VMM preimage manifest SHA-256
  `e0eb1b64d15812f122450f2e32323f0c907c640b8f8ccc270c77037bb9909b85`;
- seven-file vLLM lease contract SHA-256
  `2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1`.

vLLM fork:

- repository `local-inference-lab/vllm`, branch `dev/jovian-judgement`, commit
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`;
- direct commits: GLM model implementation
  `e0db84abedb4a85f93d130252e54b73c0f3ed695`, B12X integration
  `0c878821cf46c99729c7936bcbd4d868ad40e44e`, C4 state support
  `4dbd82b9ced13114f90e93b8b6fae0966c942a3b`, C4 behavior
  `1036123e935177900122c14d3cf02ad67b5422aa`, and DFlash speculation
  `e7097feb6fcdf57911cd68884420af2d80600dd7`;
- merged pull requests: #486, B12X C4 decode-context parallelism, merge
  `15d3f79439eadc396a57e253c955aa149def94ea`; #489, single-application
  MoE routing, merge `015dcd423d6aabf843c8ad69074ff67d35c2a395`; #493,
  captured custom-operation resource lifetime, merge
  `067c37d974ca2b775d95e51e8fec234929f4e2c4`; #494, independent target
  and draft KV formats, merge `e91c7e68f5863a27c79d2773205678be7d8ff132`;
  #497, multimodal processor revision binding, merge
  `05d85f603097fe7678d7dda2d522613d9dc61f46`; and #499, native
  serialized MXFP8 DFlash2 projections, merge
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`.

Other parent components:

- `local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`;
  no associated pull request was found;
- NVIDIA NCCL `v2.30.7-1` commit
  `73cf112295c33aee2b895f329f592f2a9b4b0f97`, base tree
  `3e7de6f92f0190d1afe9f05642e634cbf43ae4c9`, SparkRing patch
  `spark_transport/nccl/nccl-2.30.7-switchless-cycle.patch` SHA-256
  `6709063fa1c25055ae77a9397dea5d89643f8211d25e7990bdd11597d08c0dde`,
  and patched tree `abdeb053b94c3f6d472cd55ae2b79ca821299009`;
- InstantTensor `0.1.9` source distribution SHA-256
  `d8692b97991c1a5fb2db7905b9a6ae90a7f967c7ddd853d35e41caa146750c02`.

No model, B12X, vLLM, or NCCL source file is copied from an undocumented
image. The source receipts and license files are stored inside the parent.

## Changes from the base image

Only derivative-specific changes are listed here:

- copies the SparkCache source tree to `/opt/sparkcache-src/sparkcache`;
- copies and conditionally applies the one VMM exemption patch after exact
  preimage verification;
- verifies the VMM postimage and seven-file vLLM lease contract;
- adds `PYTHONPATH=/opt/sparkcache-src`;
- adds the SparkCache Apache-2.0 license and OCI source/base/revision labels.

The overlay inherits GLM-5.3 execution, B12X kernels, DFlash2, CUDA graphs,
the source-built NCCL library, runtime environment defaults, and
`ENTRYPOINT ["vllm"]` unchanged from the parent.
Additional derivative patches, file overlays, package installations,
entrypoint changes, or environment defaults: `N/A`.

## Parent build flags, environment defaults, and entrypoint

The parent builds B12X as a Python 3.12 wheel and NCCL with
`NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121`. It sets:

```text
VLLM_PLUGINS=
VLLM_B12X_MOE_FP4_FORCE_A16=0
CUTE_DSL_ARCH=sm_121a
FLASHINFER_CUDA_ARCH_LIST=12.1f
TORCH_CUDA_ARCH_LIST=12.1a
CMAKE_CUDA_ARCHITECTURES=121
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LD_PRELOAD=/opt/sparkring/nccl/libnccl.so.2
VLLM_NCCL_SO_PATH=/opt/sparkring/nccl/libnccl.so.2
ENTRYPOINT=["vllm"]
```

## Tested configuration

- Hardware: four NVIDIA DGX Spark systems, one NVIDIA GB10 GPU each.
- Driver: NVIDIA `580.173.02` on every rank.
- Kernel: `6.17.0-1029-nvidia` on every rank.
- Container engine: Docker client and server `29.2.1`.
- Runtime: Linux/ARM64, CUDA `13.0.3`, Python `3.12`, PyTorch `2.13.0`.
- Topology: four-host direct RoCE cycle; TP4, DCP1, PP1.
- Target:
  `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`;
  ModelOpt mixed precision with NVFP4 target experts and MXFP8 embedded MTP
  expert. Unquantized base-checkpoint revision: `UNKNOWN — needs verification`.
- Draft:
  `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`,
  BF16, seven speculative tokens, draft TP4, CC BY-NC-ND 4.0.
- KV mode: 12 GiB FP8 GPU KV per rank; measured capacity 549,950 tokens.
- External cache: SparkCache maximum 48 GiB and low watermark 40 GiB per
  rank; rank-local NVMe; SparkCache direct CUDA restore and streaming disabled.
- Graphs: target `FULL_AND_PIECEWISE`, DFlash FULL, capture sizes
  8, 16, 32, 64, 128, and 256.
- Scheduler: asynchronous scheduling, chunked prefill, native prefix caching,
  524,288-token model limit, 8,192 batched tokens, and 32 sequences.
- Prefill: Triton KDA backend. Load format: stock safetensors.

The launch interface and complete sanitized profile are in the
[SparkRing quickstart](https://github.com/FujitsuPolycom/sparkring/blob/e85396b9e6491c9846a7cc398fdb7287b5211d94/docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md).

## Validation commands

The quickstart provides the complete preflight, immutable cluster pull,
launcher, health, persistent request, coordinated stop/start, metrics, log,
image-ID, and RDMA commands. The request harness is
[`qualification_request.py`](qualification_request.py).

Core request commands:

```bash
python deploy/glm53_flash/qualification_request.py \
  --endpoint http://<rank-0-address>:8015 \
  --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 \
  --kind persistent --output persistent.json
python deploy/glm53_flash/qualification_request.py \
  --endpoint http://<rank-0-address>:8015 \
  --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 \
  --kind semantic --output semantic.json
```

## Validation results

- Every rank committed the same 8,192-token context digest.
- After coordinated container replacement, every rank restored 8,192 tokens
  in 156.8–171.8 ms; the restored request took 1.902 seconds.
- DFlash emitted 504 tokens from 72 drafts. The historical semantic qualifier
  proved continued generation and the expected marker suffix; it did not prove
  exact visible output because the qualifier accepted preceding reasoning text.
- The cache-disabled profile from the same image emitted 231 tokens from 33
  drafts, passed semantic validation, and made zero external-cache queries.
- Every rank retained 24 RTS worker QPs with zero preemptions, restarts, OOMs,
  or fatal-log matches.
- SparkRing tests: 1,885 passed, 9 skipped; Ruff passed.
- SparkCache tests: 621 passed, 4 skipped; Ruff passed; Python 3.11, 3.12,
  3.13, and distribution-build CI passed.
- Anonymous digest pulls passed on all four ranks after Docker logout.

## Performance claims

N/A. The results above are functional observations. No throughput,
performance-regression, or latency-distribution claim is made, so no
performance baseline is implied.

## Known limitations

- MTP drafting, other checkpoints, other topologies, spans over 8,192 tokens,
  SparkCache direct CUDA restore, streaming snapshots, throughput, and soak behavior:
  `Not tested`.
- Exact semantic output was not established by the historical suffix-only
  receipt. Requalification must use the exact-content qualifier.
- FlashKDA prefill and InstantTensor checkpoint loading: `unsupported` by this
  source-built image.
- The optional `deep_ep` import emits a duplicate-NCCL warning before vLLM
  selects `/opt/sparkring/nccl/libnccl.so.2` and serves successfully.
- The exact unquantized target base-checkpoint revision is
  `UNKNOWN — needs verification`.
- A rebuild has implemented status until its own immutable digest passes the
  same live checks.

## Support contact or issue tracker

- Support owner: `FujitsuPolycom`.
- Derivative issues: <https://github.com/FujitsuPolycom/sparkcache/issues>.
- Parent runtime and transport issues:
  <https://github.com/FujitsuPolycom/sparkring/issues>.
- Dedicated Discord support thread: `UNKNOWN — needs verification`.

Do not post this image in a main model channel until a dedicated support
thread exists and its URL replaces the unverified field above. Problems must
be reproduced with the connector omitted before assignment to the parent or
another upstream project.

## Upstream useful work

The SparkCache overlay changes are maintained in
[`FujitsuPolycom/sparkcache` PR #19](https://github.com/FujitsuPolycom/sparkcache/pull/19).
No local-inference-lab pull request has been opened for the connector-specific
VMM exemption. Upstream submission status: `Not submitted`.
