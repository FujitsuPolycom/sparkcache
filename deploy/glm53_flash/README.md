# GLM-5.3 Flash hybrid context-cache deployment

Status: **qualified** for the exact GLM-5.3 Flash TP4/DCP1 OCI image
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

The qualified overlay contains SparkCache commit
`3860a2250193a6679ac6bac857af53e0757841f8` from branch
`codex/glm53-public-image`; its deployable source-tree SHA-256 is
`6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2`.

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

Pull the qualified image by immutable digest:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943
```

Its source-built parent is
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
Build an overlay from that immutable parent with:

```bash
python deploy/glm53_flash/build_public_image.py \
  --repository . \
  --base-image ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd \
  --output-image glm53-flash-sparkcache:local \
  --output glm53-sparkcache-build-receipt.json
```

The builder verifies the parent digest, parent image ID, OCI source labels,
SparkCache tree, VMM patch preimage and postimage, and seven-file vLLM lease
contract. A rebuilt image has implemented status until its own immutable
digest passes the four-rank checks.

[`PUBLISHING.md`](PUBLISHING.md) defines the source-bound public image path. It
builds the SparkCache overlay from an immutable SparkRing runtime digest,
records the parent and output identities and requires an SPDX SBOM.

The serving command must use `SparkContextCacheConnector`, role `kv_both`,
load-failure policy `recompute`, model profile `glm53-flash-hybrid`, and exact
64-character lowercase SHA-256 identities for both target and draft
checkpoints. Streaming snapshots and native direct restore remain disabled for
this profile.

`build_kv_transfer_config` validates the syntax and namespace role of caller-
supplied checkpoint identities; it does not inspect model mounts. A launcher
must verify the target artifact manifest and the DFlash config and weight
SHA-256 values before starting vLLM. The qualified SparkRing TP4 launcher owns
that artifact-verification boundary at
the `codex/glm53-flash-sparkcache-tp4` branch of
[`FujitsuPolycom/sparkring`](https://github.com/FujitsuPolycom/sparkring/tree/codex/glm53-flash-sparkcache-tp4).
Supplying an unverified digest is unsupported.

## Compatibility

The profile creates a distinct cache namespace. It does not modify the
`glm52-nvfp4` profile, its alias, or any GLM-5.2 cache identity field.
