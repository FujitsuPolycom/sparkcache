# GLM-5.3 Flash SparkCache community-image announcement

## Status

**Community derivative.** This is not an official vLLM, Z.AI, Inco AI,
NVIDIA, or B12X release. The exact TP4/DCP1 configuration below is
**qualified**; configurations outside that record remain unsupported or
unqualified as stated below.

## Image and immutable identity

The qualification used one locally built image per Spark because each host had
a different immutable parent image ID:

| Rank | Parent image ID | Derived image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

No registry image is published by this record. A publisher must report the
resulting registry manifest digest; the tag
`glm53-flash-sparkcache:da4d7be-glm53-hybrid` is not an immutable identity.

## Based on

- Rank-local GLM-5.3 ARM64 parent images identified above. SparkCache does not
  publish those parent images.
- `local-inference-lab/vllm` branch `dev/jovian-judgement` at
  `da4d7be6c97434f6942292ed8abbf4b32dc44355`.
- `local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
- NCCL 2.30.7 binary SHA-256
  `ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`.
  The qualification record does not establish that binary's source commit.

## Build recipe

Use [`build_image.py`](build_image.py) and [`Containerfile`](Containerfile).
The builder verifies the local parent image ID before invoking Docker. The
container build verifies SparkCache source-tree SHA-256
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`,
the VMM patch preimage/postimage, and seven coherent vLLM safety-contract files.

## Source revisions, pull requests, and patches

The complete target, draft, vLLM, B12X, and patch provenance is in
[`README.md`](README.md). The vLLM runtime includes merged pull requests
[#493](https://github.com/local-inference-lab/vllm/pull/493),
[#494](https://github.com/local-inference-lab/vllm/pull/494),
[#497](https://github.com/local-inference-lab/vllm/pull/497), and
[#499](https://github.com/local-inference-lab/vllm/pull/499).

## Inherited behavior

The parent image supplies GLM-5.3 model execution, B12X attention/MoE/linear
kernels, FlashKDA prefill, DFlash2 execution, CUDA-graph support, InstantTensor,
and SparkRing NCCL. Inherited OCI labels describe parent-image capabilities;
the qualified service mounts the BF16 DFlash checkpoint named below rather
than the parent's MXFP8 DFlash checkpoint.

## Changes introduced by this derivative

- Adds the `glm53-flash-hybrid` persistent context-cache profile.
- Stores sparse-MLA/C4 pages and aligned KDA/GDN recurrent checkpoints on
  rank-local NVMe.
- Binds target and draft checkpoint identities into a distinct cache namespace.
- Adds a SparkContextCacheConnector-specific VMM exemption with exact
  preimage/postimage validation.
- Verifies coherent vLLM block-lifetime and recurrent block-table semantics.

## Tested configuration

- Four NVIDIA DGX Sparks in a direct RoCE ring; TP4, DCP1, PP1.
- Target:
  `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
- Draft:
  `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`,
  BF16, seven draft tokens.
- 524,288-token model limit, 8,192-token scheduler budget, 32 sequences,
  12 GiB FP8 KV cache per rank.
- Async scheduling, chunked prefill, prefix caching, `FULL_AND_PIECEWISE`
  target graphs, and DFlash FULL graphs.

## Validation results

[`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md)
records the commands and receipts. The exact artifact stored and restored
8,192 tokens across a coordinated restart in 147.2–194.0 ms per rank. The
post-restore request completed in 1.509 seconds. The semantic canary passed,
DFlash produced exactly seven tokens per draft, and every rank remained free
of OOMs, restarts, and fatal-log matches.

## Known limitations

- No throughput-neutrality claim or span larger than 8,192 tokens is qualified.
- MTP, other topologies, other checkpoints, native restore, and streaming
  snapshots are not qualified by this image record.
- Parent images are not publicly reproducible from this repository.
- The target quantization repository does not identify its unquantized source
  checkpoint revision.
- The NCCL binary is checksum-bound but not source-commit-bound.
- The DFlash checkpoint is CC BY-NC-ND 4.0 and is not included in the image.

## Support owner and issue tracker

SparkCache-specific issues belong at
<https://github.com/FujitsuPolycom/sparkcache/issues>. Reproduce a problem with
the derivative and its parent image before assigning it to vLLM, B12X, Z.AI,
Inco AI, or NVIDIA maintainers.

## Upstream contributions

Generally useful cache behavior belongs in focused SparkCache pull requests.
Changes to vLLM block ownership, recurrent-cache semantics, or model execution
belong in focused `local-inference-lab/vllm` pull requests with their own
source and validation contracts.
