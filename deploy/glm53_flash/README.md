# GLM-5.3 Flash hybrid context-cache deployment

## Support status

The `glm53-flash-hybrid` profile is **qualified** for GLM-5.3 Flash at
TP4/DCP1 with BF16 DFlash2 using seven draft tokens under two exact runtime
records:

- [`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md)
  records an 8,192-token persistent restore through the Python page-placement
  path at 147.2--194.0 ms per rank.
- [`GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md`](../../GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md)
  records native direct restore of a 131,072-token prefix, multi-group
  recovery, and bounded shared GPU-prefix reuse through C16.

Qualification applies only to the checkpoint revisions, source contracts,
topology, settings, and immutable artifacts named in those records. It does
not transfer to another vLLM tree, target or draft checkpoint, TP/DCP geometry,
scheduler configuration, or registry image.

The qualified 8,192-token Python-placement artifact is public at digest
`sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943`.
Its immutable parent and build provenance are recorded in
[`IMAGE_ANNOUNCEMENT.md`](IMAGE_ANNOUNCEMENT.md).

Native 131,072-token restore and shared GPU-prefix reuse are qualified only for
the source-bound runtime named in the native validation record. No public OCI
digest carries that runtime. [`PUBLISHING.md`](PUBLISHING.md) requires every
rebuilt digest to complete its own four-rank qualification.

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
by truncating an opaque page manifest is **unsupported**. Tail-only GLM
publication requires a page-semantic format and a distinct cache namespace.

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

The image pins `local-inference-lab/b12x` commit
`2fcf23a0ce269be27b2e03fece73d46e90e6aeea`. GitHub reports no pull request
associated with that commit.

The image applies exact-input SparkCache patches for KV ownership, the VMM
exemption, multi-group restore recovery, bounded shared-prefix retention, and
follower attachment. The ten-file contract
`sparkcache/runtime_patches/vllm-kv-block-lease-contract-da4d7be.json` verifies
whole-file hashes and required symbols against installed vLLM source. An
unrecognized hash is unsupported.

## Image construction

Pull the qualified public Python-placement artifact by immutable digest:

```bash
docker pull ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943
```

Its parent is
`ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd`.
The public artifact remains bound to SparkCache revision `3860a2250193a6679ac6bac857af53e0757841f8`;
it does not contain the later native/shared-prefix source described above.

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

The dedicated patch directory and eleven-file contract reject any vLLM source
preimage or postimage outside the pinned revision. The overlay leaves
SparkCache wire values, digest salts, chunk geometry, and cache-identity fields
unchanged. Adaptive and static embedded MTP profiles use separate draft-state
identity digests.

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

Native page restore is disabled unless the launch supplies all of:

- `SPARK_CONTEXT_CACHE_NATIVE_RESTORE=1`;
- an attested `SPARK_CONTEXT_CACHE_NATIVE_LIBRARY` path;
- its `SPARK_CONTEXT_CACHE_NATIVE_LIBRARY_SHA256`;
- a 64, 128, or 256 MiB `SPARK_CONTEXT_CACHE_NATIVE_ARENA_BYTES` value.

The qualified 128K runtime used two host restore workers, two native placement
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
See the native restore validation record for exact runtime identities, results,
and committed receipts.

## Compatibility

The native placement path, longest exact-boundary search, and shared GPU lease
do not change `CacheIdentity`, digest values, 256-token logical geometry, exact
manifest format, or existing chunk bytes. Missing or incompatible state remains
a cache miss followed by ordinary computation.

The `glm53-flash-hybrid` namespace remains distinct from `glm52-nvfp4` and all
DeepSeek namespaces. Cross-topology and heterogeneous-TP reuse are unsupported.
