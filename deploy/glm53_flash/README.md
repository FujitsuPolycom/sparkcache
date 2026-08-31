# GLM-5.3 Flash hybrid context-cache deployment

## Support status

The `glm53-flash-hybrid` profile is **qualified** for GLM-5.3 Flash at
TP4/DCP1 with BF16 DFlash2 using seven draft tokens under two exact runtime
records:

- [`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md)
  records an 8,192-token persistent restore through the Python page-placement
  path at 147.2--194.0 ms per rank.
- [`GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md`](../../GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md)
  records SparkCache direct CUDA restore of a 131,072-token prefix, multi-group
  recovery, and bounded shared GPU-prefix reuse through C16.
- [`GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md`](../../GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md)
  records bounded exact physical-page delta restore and different-root
  shared-base reads for one PR535 TP4 image.

Qualification applies only to the checkpoint revisions, source contracts,
topology, settings, and immutable artifacts named in those records. It does
not transfer to another vLLM tree, target or draft checkpoint, TP/DCP geometry,
scheduler configuration, or registry image.

The canonical public image route is the Jovian Judgement r7 ARM64 image in
[`JJ_R7_ARM64_IMAGE.md`](JJ_R7_ARM64_IMAGE.md). Its status is **implemented and
TP4 smoke-verified, not generally qualified**. The record pins the immutable
digest, public source composition, exact target and draft artifacts, and the
four-host run procedure.

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5
```

Continue with the
[GLM-5.3 Jovian Judgement r7 GB10 TP4 quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_JJ_R7_GB10_TP4_QUICKSTART.md).

The superseded Python-placement image at digest `sha256:cd4045b...` retains its
qualified 8,192-token evidence in
[`IMAGE_ANNOUNCEMENT.md`](IMAGE_ANNOUNCEMENT.md). That historical record is not
the run procedure for the Jovian Judgement r7 image.

SparkCache CUDA restore of a 131,072-token prefix and shared GPU-prefix reuse
remain qualified only for the source-bound runtime named in the SparkCache CUDA
validation record. The public Jovian Judgement r7 image has bounded C4 smoke
evidence, not that broader qualification.

## Upstream runtime and artifacts

GLM runtime performance and correctness come primarily from Local Inference
Lab's [Jovian Judgement vLLM work](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement).
The canonical public image uses public source snapshot
[`FujitsuPolycom/vllm@331573d2`](https://github.com/FujitsuPolycom/vllm/commit/331573d20bd47e78327ed8d8b4d2e6d350bbb1ab),
tree `927f52a0085bcecfd2ba679e5abebe1a62623daf`, and
[`B12X@6255090a`](https://github.com/local-inference-lab/b12x/commit/6255090a03b12c3f7d552102a02fac0b542fb8c9),
tree `0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad`. It uses
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
and the BF16
[`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410).
The external draft is not Local Inference Lab's separate
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

## Stored state and prefix behavior

The profile stores opaque target-model pages under storage mode
`block_pages_v1`. One transaction contains sparse-MLA pages, C4 selector tail
bytes, and KDA/GDN recurrent checkpoints at one aligned persistent boundary.
The profile requires vLLM `--mamba-cache-mode align` and rejects other
recurrent-cache modes.

The logical cache geometry is 256 tokens. The scheduler computes compatible
digests at every eligible boundary and selects the longest exact snapshot
advertised by all four physical ranks. A grown conversation can therefore
reuse an earlier request boundary when that exact snapshot still exists on
every rank.

Opaque page chunks are authenticated byte partitions of one complete boundary
snapshot; they are not independent 256-token KV objects. Sparse prefix aliases
are **implemented** only for `per_token_rows`. Creating an earlier GLM prefix
by truncating an opaque page manifest is **unsupported**. Opt-in physical-page
delta publication uses the distinct `page-tail-cow-v1` identity and
authenticates the complete base-plus-delta graph. Publication, direct
SparkCache CUDA restore, shared-base coalescing, and bounded eight-lane restore
are **implemented with bounded exact TP4 evidence**. General qualification
remains specific to a model, topology, runtime, and workload. The PR535
validation record identifies the evidence boundary.

Concurrent requests for the same persistent digest use one restore leader.
After all workers report successful restoration, patched vLLM retains the
leader's normalized multi-group block table as a bounded shared-prefix lease.
The implementation permits sixteen waiting followers, two retained leases,
and a fifteen-second lease lifetime. Partial physical pages are copied into
dedicated immutable blocks before followers can attach. Lease rejection skips
the optimization and lets requests restore or recompute normally.

## Checkpoint and runtime identity

The target checkpoint is
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
Its published configuration identifies ModelOpt
`0.39.0.dev290+gf9d9a71de.d20260407` as the quantization producer. Routed
experts in target layers 3--44 use NVFP4, and embedded MTP expert layer 45 uses
MXFP8. The repository does not identify the unquantized source revision, so
this deployment does not infer one.

The DFlash2 checkpoint is
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`.
Its BF16 weights have SHA-256
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.
External speculative-draft state is recomputed after target-prefix restore.
The draft checkpoint digest remains part of `CacheIdentity`, so another draft
model cannot share this namespace.

The serving engine is `local-inference-lab/vllm` branch
`dev/jovian-judgement` at
`da4d7be6c97434f6942292ed8abbf4b32dc44355`. The runtime includes GLM-5.3
support, B12X integration, C4 cache pages, and DFlash speculation. Its merged
pull-request lineage includes
[#493](https://github.com/local-inference-lab/vllm/pull/493),
[#494](https://github.com/local-inference-lab/vllm/pull/494),
[#497](https://github.com/local-inference-lab/vllm/pull/497), and
[#499](https://github.com/local-inference-lab/vllm/pull/499).

The historical qualified source runtime pins `local-inference-lab/b12x` commit
`2fcf23a0ce269be27b2e03fece73d46e90e6aeea`. GitHub reports no pull request
associated with that commit.

The image applies exact-input SparkCache patches for KV ownership, the VMM
exemption, multi-group restore recovery, bounded shared-prefix retention, and
follower attachment. The ten-file contract
`sparkcache/runtime_patches/vllm-kv-block-lease-contract-da4d7be.json` verifies
whole-file hashes and required symbols against installed vLLM source. An
unrecognized hash is unsupported.

## Historical da4d7be overlay construction

The following source-bound procedure reproduces the superseded da4d7be
Python-placement artifact. It does not build the Jovian Judgement r7 image.
Pull the historical artifact by immutable digest:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943
```

Its parent is
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
The public artifact remains bound to SparkCache revision `3860a2250193a6679ac6bac857af53e0757841f8`;
it does not contain the later SparkCache CUDA/shared-prefix source described
above.

Build from the repository root after recording the exact local parent image
ID and SparkCache source-tree digest:

```bash
python deploy/glm53_flash/build_image.py \
  --base-image glm53-flash-spark:source-locked \
  --base-image-id sha256:<64-lowercase-hex> \
  --source-sha256 <64-lowercase-hex> \
  --output-image glm53-flash-sparkcache:da4d7be-glm53-hybrid
```

The builder rejects a parent whose resolved image ID differs from
`--base-image-id`. The container build verifies the SparkCache source digest,
patch inputs and outputs, and the ten-file vLLM contract.

### Source-built e10536a runtime

`Containerfile.e10536a` overlays SparkCache on a parent built from
`local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3`.
Its four exact-input patches and ten-file contract are implemented but not
qualified. The public da4d7be OCI artifact at digest
`sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`
and its 8,192-token evidence remain separate.

Build the overlay only after verifying that the local parent carries the exact
vLLM commit in its OCI labels:

```bash
python deploy/glm53_flash/build_image.py \
  --containerfile deploy/glm53_flash/Containerfile.e10536a \
  --base-image sparkring-glm53-runtime:e10536a-source-arm64 \
  --base-image-id sha256:<64-lowercase-hex> \
  --source-sha256 <64-lowercase-hex> \
  --output-image sparkring-glm53-sparkcache:e10536a-source-arm64
```

The external-draft configuration retains the BF16 DFlash2 checkpoint at depth
five, isolating the vLLM revision from a speculative-model change. The
embedded-draft configuration uses the target checkpoint's MTP layer with a
maximum depth of five. Acceptance-length adaptation is opt-in and uses an
initial depth of three with a 32-step window when enabled.

External DFlash, static embedded MTP, and adaptive embedded MTP use distinct
`draft_checkpoint` identity values. Switching between them therefore selects
a different cache namespace and recomputes instead of reusing entries from an
incompatible speculative policy.

### Source-built runtime with live-tensor B12X KDA binding

`Containerfile.b12x-kda-adaptive-mtp` overlays SparkCache on a parent built
from `local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5`.
The runtime contains internal MTP5, acceptance-based adaptive draft depth, and
the B12X KDA implementation that binds metadata once and operates on live
layer tensors. Construction support is **implemented**; four-rank serving is
**unqualified**.

```bash
python deploy/glm53_flash/build_image.py \
  --containerfile deploy/glm53_flash/Containerfile.b12x-kda-adaptive-mtp \
  --base-image sparkring-glm53-runtime:b12x-kda-adaptive-mtp-0b67266a-arm64 \
  --base-image-id sha256:<64-lowercase-hex> \
  --source-sha256 <64-lowercase-hex> \
  --output-image sparkring-glm53-sparkcache:b12x-kda-adaptive-mtp-0b67266a-arm64
```

The dedicated patch directory and eleven-file contract reject any attested
vLLM file whose preimage or postimage differs from the pinned revision. The
overlay leaves SparkCache wire values, digest salts, chunk geometry, and
cache-identity fields unchanged. Adaptive and static embedded MTP profiles use
separate draft-state identity digests.

The connector must use role `kv_both`, load-failure policy `recompute`, model
profile `glm53-flash-hybrid`, and exact lowercase SHA-256 identities for both
checkpoints. `build_kv_transfer_config` validates configuration syntax and
identity shape; the launcher remains responsible for hashing mounted model
artifacts.

An operator can set string key `spark_cache_clear_once` in
`kv_connector_extra_config` to clear only SparkCache-owned data under each
configured rank-local root. The token is durably remembered after successful
removal, so an unchanged static launch configuration does not clear again after
a restart. A different token requests another clear. Lock timeout or removal
failure disables persistent caching for that connector process and leaves the
token incomplete for retry; it does not delay model serving indefinitely. See
[`sparkcache/README.md`](../../sparkcache/README.md#one-shot-cache-clear) for
token, root-path, and deletion-scope rules.

SparkCache CUDA restore is disabled unless the launch supplies all of:

- `SPARK_CONTEXT_CACHE_CUDA_RESTORE=1`;
- an attested `SPARK_CONTEXT_CACHE_CUDA_PLACEMENT_LIBRARY` path;
- its `SPARK_CONTEXT_CACHE_CUDA_PLACEMENT_LIBRARY_SHA256`;
- a 64, 128, or 256 MiB
  `SPARK_CONTEXT_CACHE_CUDA_PLACEMENT_ARENA_BYTES` value.

The qualified 128K runtime used two host restore workers, two SparkCache CUDA placement
lanes, and two 256 MiB mapped-host arenas per rank. Streaming snapshots remain
unsupported for opaque page storage.

The matching four-Spark launch and artifact-verification procedure is pinned
at
[`FujitsuPolycom/sparkring@6e9e3ac`](https://github.com/FujitsuPolycom/sparkring/blob/6e9e3acef62886a71531310673463972944b2b84/docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md).

## Concurrency benchmark

`concurrency_benchmark.py` sends synchronized C2, C8, or C16 cohorts to an
OpenAI-compatible endpoint. Scenario `identical-prefix` repeats one prompt;
`shared-trunk` adds deterministic request-specific tails. `--pretokenize`
removes repeated API-side chat tokenization from the timed interval.

The tool does not warm, clear, or inspect SparkCache. The operator prepares the
storage condition and records it with `--cache-state`. Each JSON receipt binds
the model, prompt hashes, request results, and min/p50/p95/max latency.

```bash
python -m deploy.glm53_flash.concurrency_benchmark \
  --endpoint http://spark-r0:8000 \
  --model local-inference-lab/GLM-5.3-Flash-NVFP4 \
  --concurrency 16 \
  --scenario identical-prefix \
  --cache-state hot \
  --pretokenize \
  --output receipts/glm53-c16-identical-hot.json
```

The default fixture reproduces the recorded 131,072-token persistent prefix.
See the SparkCache CUDA restore validation record for exact runtime identities, results,
and committed receipts.

## Compatibility

The SparkCache CUDA placement path, longest exact-boundary search, and shared GPU lease
do not change `CacheIdentity`, digest values, 256-token logical geometry, exact
manifest format, or existing chunk bytes. Missing or incompatible state remains
a cache miss followed by ordinary computation.

The `glm53-flash-hybrid` namespace remains distinct from `glm52-nvfp4` and all
DeepSeek namespaces. Cross-topology and heterogeneous-TP reuse are unsupported.
