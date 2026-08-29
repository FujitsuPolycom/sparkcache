# GLM-5.3 Flash BF16 DFlash2 TP4/DCP1 validation

Status: **qualified** for the immutable image and serving contract below.

## Artifact and conditions

| Property | Qualified value |
|---|---|
| SparkCache image | `ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:cd4045bba2a0f3dc55361560f8c3a3f171939854db28d48dfdae58eed9c44943` |
| Image ID on every rank | `sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290` |
| Parent runtime | `ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:864adfe68f458223e186a19844ac80c7adc7365e5db1f25e109b85fc19850dcd` |
| SparkCache source | commit `3860a2250193a6679ac6bac857af53e0757841f8`; tree SHA-256 `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2` |
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`, BF16 |
| Runtime | `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`; B12X `2fcf23a0ce269be27b2e03fece73d46e90e6aeea` |
| Topology | four DGX Spark systems; direct RoCE cycle; TP4/DCP1/PP1 |
| Limits | 524,288 model tokens; 8,192 batched tokens; 32 sequences |
| GPU KV | 12 GiB FP8 per rank; measured capacity 549,950 tokens |
| SparkCache | 48 GiB maximum; 40 GiB low watermark per rank |
| Execution | async scheduler; chunked prefill; native prefix caching; Triton KDA prefill; `FULL_AND_PIECEWISE` target graphs; FULL DFlash graphs |

The image contains no target or DFlash weights. The target repository's
published ModelOpt configuration uses NVFP4 target experts and an MXFP8
embedded MTP expert; it does not identify the unquantized base-checkpoint
revision. Inco AI publishes the BF16 DFlash checkpoint under CC BY-NC-ND 4.0.

NCCL was built from NVIDIA `v2.30.7-1` commit
`73cf112295c33aee2b895f329f592f2a9b4b0f97` with SparkRing's original
switchless-cycle patch. The patched tree is
`abdeb053b94c3f6d472cd55ae2b79ca821299009`; the loaded library SHA-256 is
`5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3`.

## Measurement and result

A deterministic request created an 8,192-token reusable span. Every rank
committed digest `a6d78b05026f...`. All four containers were replaced without
deleting rank-local cache data. Each worker discovered one local manifest
with zero rejections. One request safely recomputed while worker inventories
reached the scheduler; the following identical request restored the span.

| Measurement | Result |
|---|---|
| Cache content per rank | 1 manifest; 32 chunks; 103,890,664 bytes |
| Commit time by rank | 271.2, 271.2, 267.6, 313.8 ms |
| Restore time by rank | 162.7, 171.8, 158.8, 156.8 ms |
| External queried / hit tokens | 16,457 / 8,192 |
| Restored request | 1.902 seconds; validation passed |
| DFlash | 72 drafts; 504 draft tokens; 170 accepted tokens |
| Semantic canary | 2.840 seconds; `stop`; semantic match |
| Health | 24 RTS worker QPs/rank; zero preemptions, restarts, OOMs, or fatal matches |

The same image was then launched without `--kv-transfer-config`. Its semantic
canary passed in 2.703 seconds. DFlash produced 231 tokens from 33 drafts;
external-cache queries remained zero; connector logs were absent; and all
process and RDMA conditions passed.

## Conclusion

The recorded image restores persistent target context after a coordinated
TP4 restart while BF16 DFlash2 retains its seven-token speculation invariant.
It also serves correctly when the external connector is omitted.

## Reproduce

The pull, source-build, cluster configuration, launch, tail, metrics, restart,
and request commands are in
[`deploy/glm53_flash/README.md`](deploy/glm53_flash/README.md) and the
SparkRing branch's `docs/GLM53_FLASH_DFLASH2_BF16_SPARKCACHE_TP4_QUICKSTART.md`.
The deterministic request program is
[`deploy/glm53_flash/qualification_request.py`](deploy/glm53_flash/qualification_request.py).

## Limitations

This is a functional single-observation qualification, not a throughput,
latency-distribution, or soak result. It covers one 8,192-token restored span.
MTP drafting, other draft checkpoints, other topologies, native direct
restore, streaming snapshots, FlashKDA prefill, and InstantTensor checkpoint
loading are unsupported by this record. The optional `deep_ep` import can
emit a duplicate-NCCL warning before vLLM selects the source-built NCCL
library.
