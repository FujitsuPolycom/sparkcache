# GLM-5.3 Flash SparkCache community-image qualification record

## Artifact status

This document is a **qualified historical artifact record** for the four
rank-local images listed below. It is not a release announcement for the
repository's image builder and does not identify a published registry image.

The images are FujitsuPolycom community derivatives, not official vLLM, Z.AI,
Inco AI, NVIDIA, B12X, or SparkCache release artifacts. Qualification applies
only to the exact TP4/DCP1 configuration and 8,192-token restore recorded here.

## Immutable image identity

Each NVIDIA DGX Spark used a different local parent image and therefore a
different derived image:

| Rank | Parent image ID | Derived image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

No registry digest is associated with this record. The local tag
`glm53-flash-sparkcache:da4d7be-glm53-hybrid` is a mutable build handle, not an
artifact identity.

## Source and dependency identity

The images contain SparkCache source-tree SHA-256
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`
from repository revision
[`b635de0d8cf5278ea382f1bfd070f19e201460fc`](https://github.com/FujitsuPolycom/sparkcache/tree/b635de0d8cf5278ea382f1bfd070f19e201460fc).

The exact builder and container recipe used for this artifact remain available
at that immutable revision:

- [`build_image.py`](https://github.com/FujitsuPolycom/sparkcache/blob/b635de0d8cf5278ea382f1bfd070f19e201460fc/deploy/glm53_flash/build_image.py);
- [`Containerfile`](https://github.com/FujitsuPolycom/sparkcache/blob/b635de0d8cf5278ea382f1bfd070f19e201460fc/deploy/glm53_flash/Containerfile).

The runtime dependencies were:

- `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`;
- `local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`;
- NCCL 2.30.7 binary SHA-256
  `ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`.

The NCCL checksum identifies the binary used in qualification. No source
manifest binds that binary to an NCCL source commit.

The target checkpoint was
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
The draft checkpoint was
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
with BF16 weights and seven draft tokens.

## Artifact behavior

The derivative added the `glm53-flash-hybrid` persistent context-cache profile,
stored sparse-MLA/C4 pages and aligned KDA/GDN recurrent checkpoints on
rank-local NVMe, bound target and draft identities into a distinct cache
namespace, and applied an exact-input VMM compatibility exemption.

The parent supplied GLM-5.3 execution, B12X kernels, FlashKDA prefill, DFlash2,
CUDA graphs, InstantTensor, and SparkRing NCCL. Neither model checkpoint was
embedded in the derived images.

## Qualified configuration and result

- Four NVIDIA DGX Spark systems in a direct RoCE ring;
- TP4, DCP1, and PP1;
- 524,288-token model limit;
- 8,192-token scheduler budget;
- 32-sequence scheduler ceiling;
- 12 GiB FP8 KV cache per rank;
- asynchronous scheduling, chunked prefill, prefix caching,
  `FULL_AND_PIECEWISE` target graphs, and DFlash FULL graphs.

[`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md)
contains the commands, source contract, and receipts. The four images stored
and restored 8,192 tokens across a coordinated restart in 147.2--194.0 ms per
rank. The restored request completed in 1.509 seconds. A historical canary
found the `SPARKCACHE_GLM53_OK` suffix but did not require exact visible
content. DFlash produced seven tokens per draft, and all ranks remained free
of OOMs, restarts, and fatal-log matches.

## Qualification boundary

This artifact record does not qualify native direct restore, shared GPU-prefix
leases, a span larger than 8,192 tokens, another topology, another checkpoint,
MTP, streaming snapshots, or throughput neutrality. Native restore and shared
GPU-prefix behavior from source revision
`2b86fb9d02fa3595cca5caa864b81aedce44b8bb` are described in
[`GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md`](../../GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md).

The parent images are not publicly reproducible from this repository. The
target quantization repository does not identify its unquantized source
revision. The DFlash checkpoint uses CC BY-NC-ND 4.0 and is not included in the
images.

## Support

Report SparkCache-specific defects at
<https://github.com/FujitsuPolycom/sparkcache/issues>. Include the derived image
ID, parent image ID, SparkCache source digest, vLLM revision, checkpoint
identities, and topology.
