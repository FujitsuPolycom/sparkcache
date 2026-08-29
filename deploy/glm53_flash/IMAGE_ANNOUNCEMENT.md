# GLM-5.3 Flash SparkCache community image

Status: **community derivative**; **qualified** only for the tested
configuration below. This is not an official NVIDIA, vLLM,
local-inference-lab, B12X, Inco AI, Z.AI, or SparkCache release.

Image and digest:
`ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`

Based on:
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`

Build recipe:
[`build_public_image.py`](build_public_image.py),
[`Containerfile`](Containerfile), and [`PUBLISHING.md`](PUBLISHING.md).

Source commits and patches:

- SparkCache `3860a2250193a6679ac6bac857af53e0757841f8`; deployable tree SHA-256
  `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.
- `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`,
  including merged pull requests 486, 489, 493, 494, 497, and 499.
- `local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
- NVIDIA NCCL `v2.30.7-1` commit
  `73cf112295c33aee2b895f329f592f2a9b4b0f97`, patched tree
  `abdeb053b94c3f6d472cd55ae2b79ca821299009`.
- SparkCache VMM exemption
  `patches/vllm-da4d7be/020-sparkcache-vmm-exemption.patch` and the
  seven-file vLLM lease contract.

Changes from the parent:

- Installs the `glm53-flash-hybrid` SparkCache connector.
- Persists target sparse-MLA/C4 pages and aligned KDA/GDN recurrent state on
  rank-local NVMe.
- Binds target and draft checkpoint identities into the cache namespace.
- Applies and verifies the connector-specific VMM exemption and vLLM
  KV-block lifetime contract.

Tested configuration:

- Four directly cabled NVIDIA DGX Spark systems; TP4, DCP1, PP1.
- Target
  `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
- Draft
  `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`,
  BF16, seven speculative tokens, TP4.
- 524,288 model tokens, 8,192 batched tokens, 32 sequences, 12 GiB FP8 GPU
  KV per rank, 48 GiB SparkCache per rank, Triton KDA prefill, async scheduler,
  and `FULL_AND_PIECEWISE` target graphs.

Validation results:

- 8,192 tokens restored on every rank after coordinated container replacement
  in 156.8–171.8 ms; request duration 1.902 seconds.
- DFlash emitted 504 tokens from 72 drafts; semantic canary passed.
- Cache-disabled comparison from the same image also passed and produced 231
  tokens from 33 drafts with zero external-cache queries.
- Every rank retained 24 RTS worker QPs with zero preemptions, restarts, OOMs,
  or fatal-log matches.

Known limitations:

- Functional single observations do not establish throughput or soak behavior.
- MTP, other checkpoints or topologies, restored spans over 8,192 tokens,
  native direct restore, and streaming snapshots are unqualified.
- The runtime uses stock safetensors loading and Triton KDA prefill;
  InstantTensor loading and FlashKDA prefill are unsupported.
- The BF16 DFlash checkpoint is CC BY-NC-ND 4.0 and is not in the image.

Support: report SparkCache defects at
<https://github.com/FujitsuPolycom/sparkcache/issues>. Reproduce runtime or
kernel failures without the connector before assigning them to an upstream
project. The complete evidence is
[`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md).
