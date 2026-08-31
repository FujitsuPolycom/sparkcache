# GLM-5.3 Jovian Judgement r7 ARM64 SparkCache image

## Status

**Implemented and TP4 smoke-verified; not generally qualified.** The immutable
Linux/ARM64 image completed one bounded four-request publication and restart
restore on four NVIDIA GB10 systems. The result does not qualify another
topology, checkpoint, concurrency, prompt size, cache geometry, or sustained
workload.

The GLM runtime comes from Local Inference Lab's
[Jovian Judgement vLLM work](https://github.com/local-inference-lab/vllm/tree/dev/jovian-judgement).
The public source snapshot is
[`FujitsuPolycom/vllm@331573d2`](https://github.com/FujitsuPolycom/vllm/commit/331573d20bd47e78327ed8d8b4d2e6d350bbb1ab),
with Git tree `927f52a0085bcecfd2ba679e5abebe1a62623daf`. Blackwell
kernels come from
[`B12X@6255090a`](https://github.com/local-inference-lab/b12x/commit/6255090a03b12c3f7d552102a02fac0b542fb8c9),
with Git tree `0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad`.

The smoke used the
[`GLM-5.3-Flash-NVFP4@520de24e`](https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4/tree/520de24eabf507659eaef7c70f14fd584527facc)
target and the BF16
[`GLM-5.3-Flash-DFlash2@dc77ff1c`](https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2/tree/dc77ff1c99eeb2df044ee3d4f0094eb033fee410)
external draft at depth seven. The target configuration SHA-256 is
`676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996`;
its safetensors index SHA-256 is
`0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb`.
The draft configuration SHA-256 is
`c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573`;
its weights SHA-256 is
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.

## Pull and run

Pull the immutable image rather than a mutable tag:

```bash
export GLM53_IMAGE='ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5'
docker pull "$GLM53_IMAGE"
docker image inspect "$GLM53_IMAGE" --format '{{.Id}}'
```

The expected image ID is
`sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4`.
Model checkpoints are not embedded. Use the
[GLM-5.3 Jovian Judgement r7 GB10 TP4 quickstart](https://github.com/FujitsuPolycom/sparkring/blob/54d9df70ee7f6fe9195a6b1983341497791be845/docs/GLM53_JJ_R7_GB10_TP4_QUICKSTART.md)
for checkpoint mounts, four-rank networking, per-rank launch commands,
readiness checks, and the OpenAI-compatible endpoint.

## Immutable runtime identity

| Role | Durable identifier |
|---|---|
| Child manifest | `sha256:f012dd915c0fff0be384820c2d72cd015b83b9b33c3f980445dd718a807cd0c5` |
| Child image configuration | `sha256:6af83baabb239db6b05e379401daf93c8f51694f81483c2781f6014c30e31db4` |
| Parent manifest | `sha256:11922064b342de1fc98f0ef85e6648843c8fa7eb3e4f4353c6ad82d6e457dde0` |
| Parent image configuration | `sha256:8cff7a250f16bfb89df23d29f9233dbb1c700a780dcec86a64c535a71aee88be` |
| vLLM source | `FujitsuPolycom/vllm@331573d20bd47e78327ed8d8b4d2e6d350bbb1ab`, tree `927f52a0085bcecfd2ba679e5abebe1a62623daf` |
| B12X source | `6255090a03b12c3f7d552102a02fac0b542fb8c9`, tree `0bb58d0dcc10e29e00ff9850c0d719fca1aba5ad` |
| NCCL library | SHA-256 `5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3` |
| SparkCache source | `dcbe040d339f243621163b0c6ed4ce96462403d8`, tree `861562a7f5cb867be4313a2979027bc4f499cb31`, deployable source SHA-256 `9cf50afd04e385975a487a0129645bd294e0395012424995569a9b50a7c447f1` |
| CUDA placement library | SHA-256 `d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c` |

The child and parent config digests were resolved from their published registry
manifests. The active Python composition labels are
`org.sparkring.vllm.sparkcache-composition`, `org.sparkring.vllm.tree`,
`org.sparkring.b12x.composition`, and `org.sparkring.b12x.tree`.

Inherited labels preserve compiled-extension and lower-layer provenance.
Compiled vLLM extensions retain `VLLM_BUILD_COMMIT=3633d61c...`; an
intermediate runtime label names `da4d7be6...`. Active Python source is
`331573d2...`, but the image does not claim that every compiled extension was
rebuilt from that Python composition.

Inherited `org.glm53.dflash2.*` labels also describe lower image layers. In particular,
`org.glm53.dflash2.checkpoint-revision=b6d33...` and
`org.glm53.dflash2.mxfp8-quant-plumbing=v2` are lineage and plumbing labels,
not the mounted BF16 `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c` draft used by the
smoke. These inherited labels are not the active Python, B12X composition, or
draft artifact identity. The
parent and component identifiers are part of the build receipt, not claims
that another artifact with similar tags is equivalent.

## Bounded TP4 smoke evidence

The smoke used a fresh rank-local cache root and four concurrent requests. The
publication responses were exactly `red`, `blue`, `green`, and `black`; their
canonical result-set SHA-256 was
`edb9c082fc6fe1b99004fa4c04d9e4b53d0525fe5410313ba13f18f2dc09ffbc`.
Every rank recorded a 605,690,671-byte cache root, four manifests, and the
one-shot clear completion marker.

All four engine processes were then stopped and restarted against the preserved
roots. One small inference established scheduler manifest readiness. The four
responses again matched their exact codewords; their canonical result-set
SHA-256 was
`02a0c0fafa95294008cd1b9a8a6269dabc0d161c10c307bc1922f1b9aa20c100`.
Every request used external restore. Client latency ranged from 0.561595 to
1.582937 seconds, and rank-local cache service ranged from 277 to 394 ms.

This smoke proves publication, full-process restart discovery, and external
restore for the stated C4 input. It does not prove shared-base read
coalescing, throughput, SSD endurance, C8/C16 behavior, corruption recovery,
or general deployment qualification.

The machine-readable receipt is
[`jj-r7-arm64-public-image-c4-smoke.json`](../../evidence/glm53-jj-r7-arm64/jj-r7-arm64-public-image-c4-smoke.json).
