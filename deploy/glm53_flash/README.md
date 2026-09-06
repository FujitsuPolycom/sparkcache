# GLM-5.3 Flash integration

SparkCache provides persistent prefix storage for compatible GLM-5.3 Flash
vLLM deployments. Runnable multi-host recipes live in SparkRing because that
repository assembles the model runtime, transport, image, and operator settings.

## Run the model

Use the
[four-node SparkRing quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_JJ_R8_GB10_SPARKCACHE_TP4_QUICKSTART.md).
It uses one Linux/ARM64 image for TP4 with DCP1, DCP2, or DCP4 and documents
both modes:

- `SPARKCACHE_ENABLED=1` enables persistent SparkCache alongside vLLM's GPU
  prefix cache.
- `SPARKCACHE_ENABLED=0` runs with vLLM's GPU prefix cache alone.

The quickstart is the source of truth for the image digest, model revisions,
launch variables, storage paths, and four-host procedure. Keeping those values
in SparkRing prevents a copied deployment recipe here from becoming stale.

## SparkCache contract

The `glm53-flash-hybrid` profile stores target-model manager pages owned by
each physical rank. The profile uses 256-token logical boundaries for prefix
identity while preserving the runtime's complete physical page geometry.

A restore is accepted only when every expected rank reports the same compatible
prefix and verifies its local payload. Rejected or missing state becomes an
ordinary vLLM cache miss and prompt computation. External speculative-draft
state is recomputed after target-prefix restoration.

Complete snapshots, physical-page copy-on-write publication, SparkCache CUDA
placement, and bounded shared-prefix reuse are implemented in the repository.
Live behavior still depends on the exact vLLM source, model artifacts,
topology, launch settings, and cache namespace supplied by the deployment.

## Upstream components

GLM-5.3 execution and GB10 performance depend primarily on Local Inference
Lab's [Jovian Judgement vLLM work](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement)
and [B12X kernels](https://github.com/local-inference-lab/b12x). The SparkRing
quickstart identifies the exact revisions and model artifacts, including
[GLM-5.3-Flash-NVFP4](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4)
and the BF16
[DFlash2 draft](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2).
The BF16 draft is distinct from Local Inference Lab's
[MXFP8 DFlash2 checkpoint](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8).

## Evidence and historical records

- [`qualification.json`](../../evidence/glm53-flash-dcp4-page-tail/qualification.json)
  records byte-exact 98,304-to-131,072-token page-tail publication, restart
  restore, corruption rejection, valid-base fallback, and repaired restore on
  the GLM-5.3 Flash TP4/DCP4 deployment identified inside the receipt.
- [`page-tail-cow-v2/qualification.json`](../../evidence/glm53-flash-dcp4-page-tail-v2/qualification.json)
  records direct sparse page capture, flat authenticated delta stages,
  byte-exact chained growth and restart restore, corruption rejection, and a
  concurrent-serving interference measurement for its source-bound TP4/DCP4
  artifact.
- [`GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md`](../../GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md)
  records source-bound SparkCache CUDA placement and shared-prefix evidence.
- [`GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md`](../../GLM53_PR535_PAGE_DELTA_RESEARCH_VALIDATION.md)
  records bounded physical-page delta and shared-base evidence.
- [`GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md`](../../GLM53_FLASH_DFLASH7_LIVE_VALIDATION.md)
  records an earlier Python-placement runtime.
- [`JJ_R7_ARM64_IMAGE.md`](JJ_R7_ARM64_IMAGE.md) and
  [`IMAGE_ANNOUNCEMENT.md`](IMAGE_ANNOUNCEMENT.md) preserve superseded image
  identities and their exact evidence. They are not operator instructions for
  the SparkRing quickstart linked above.
