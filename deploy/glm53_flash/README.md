# GLM-5.3 Flash hybrid context-cache deployment

Status: **qualified** for the exact GLM-5.3 Flash TP4/DCP1 source deployment
with DFlash2 using seven draft tokens in
[`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md).
GPU-free tests cover hybrid-page store/restore and exact vLLM source
verification. The live record covers an 8,192-token persistent restore; its
limitations do not transfer qualification to other source trees, checkpoints,
topologies, scheduler budgets, or span lengths.

The `glm53-flash-hybrid` profile stores opaque target-model cache pages for
GLM-5.3 Flash. The stored transaction includes sparse-MLA pages, C4 selector
tail bytes, and the KDA/GDN recurrent checkpoint at the aligned persistent
boundary. The profile requires vLLM `--mamba-cache-mode align` and rejects
other recurrent-cache modes.

External speculative-draft state is recomputed after a target-prefix restore.
The seven-token DFlash2 deployment, identified as `dflash7` in launch commands,
pins
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
with BF16 weight SHA-256
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.
The selected MTP or DFlash checkpoint SHA-256 remains part of `CacheIdentity`,
so entries created with different draft models cannot share a namespace.

The derived image applies a narrow VMM compatibility exemption because
`SparkContextCacheConnector` does not register GPU virtual addresses with an
external transport. It verifies the seven-file KV-block lease contract for
`local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`
against the Python files installed in the serving image.

## Source provenance

The target checkpoint is
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
Its published configuration identifies ModelOpt
`0.39.0.dev290+gf9d9a71de.d20260407` as the quantization producer. Routed
experts in target layers 3–44 use NVFP4, and the embedded MTP expert layer 45
uses MXFP8. The checkpoint repository does not identify the exact source
checkpoint revision used for quantization, so this deployment does not infer
one.

The serving engine is the `local-inference-lab/vllm` fork on branch
`dev/jovian-judgement`, pinned at
`da4d7be6c97434f6942292ed8abbf4b32dc44355`. The runtime contains direct branch
commits for GLM-5.3 model support (`e0db84abe`), B12X integration (`0c878821c`),
C4 cache pages (`4dbd82b9c`), C4 behavior (`1036123e9`), and DFlash speculation
(`e7097feb6`). Its merged pull-request lineage includes
[#493](https://github.com/local-inference-lab/vllm/pull/493) for captured
custom-operation resource lifetime,
[#494](https://github.com/local-inference-lab/vllm/pull/494) for independent
target/draft KV formats,
[#497](https://github.com/local-inference-lab/vllm/pull/497) for processor
revision binding, and
[#499](https://github.com/local-inference-lab/vllm/pull/499) for serialized
MXFP8 DFlash projections. Pull request #499 explicitly names #493 and #494 as
deployment dependencies. The BF16 DFlash checkpoint does not exercise the
MXFP8-only projection path, but the exact pinned runtime includes it.

The runtime image pins `local-inference-lab/b12x` commit
`2fcf23a0ce269be27b2e03fece73d46e90e6aeea` (`Accept runtime QSA cache page
sizes`). GitHub reports no pull request associated with that commit.

The VMM exemption in
`patches/vllm-da4d7be/020-sparkcache-vmm-exemption.patch` adapts the existing
SparkCache patch
`patches/vllm-e2666d9a6/020-sparkcache-vmm-exemption.patch` to the exact
GLM-5.3 runtime preimage. No GLM-5.3 model or kernel source is copied into this
repository.

Build from the repository root after recording the exact local parent image
ID:

```bash
python deploy/glm53_flash/build_image.py \
  --base-image glm53-flash-spark:source-locked \
  --base-image-id sha256:<64-lowercase-hex> \
  --source-sha256 <64-lowercase-hex> \
  --output-image glm53-flash-sparkcache:da4d7be-glm53-hybrid
```

The builder resolves `--base-image` with `docker image inspect` and rejects an
identity different from `--base-image-id`. The image is not published by
SparkCache, so this recipe does not provide a standalone public vLLM runtime
builder.

The serving command must use `SparkContextCacheConnector`, role `kv_both`,
load-failure policy `recompute`, model profile `glm53-flash-hybrid`, and exact
64-character lowercase SHA-256 identities for both target and draft
checkpoints. Streaming snapshots and native direct restore remain disabled for
this profile.

`build_kv_transfer_config` validates the syntax and namespace role of caller-
supplied checkpoint identities; it does not inspect model mounts. A launcher
must verify the target artifact manifest and the DFlash config and weight
SHA-256 values before starting vLLM. The qualified SparkRing TP4 launcher owns
that artifact-verification boundary. Supplying an unverified digest is
unsupported.

## Compatibility

The profile creates a distinct cache namespace. It does not modify the
`glm52-nvfp4` profile, its alias, or any GLM-5.2 cache identity field.
