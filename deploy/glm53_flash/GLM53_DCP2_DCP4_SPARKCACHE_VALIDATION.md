# GLM-5.3 Flash SparkCache DCP2 and DCP4 evidence

**Status: research-only evidence.** This record does not qualify a public image
or a general deployment profile. It covers bounded TP4 restores at 8,192 and
9,216 stored tokens; it does not cover large contexts, concurrency, injected
faults, or long-duration serving.

The machine-readable record is
[`evidence/glm53-dcp/tp4-dcp2-dcp4-sparkcache.json`](../../evidence/glm53-dcp/tp4-dcp2-dcp4-sparkcache.json).

## Runtime identity

| Component | Identity |
|---|---|
| vLLM composition | `331573d20bd47e78327ed8d8b4d2e6d350bbb1ab` |
| B12X composition | `6255090a03b12c3f7d552102a02fac0b542fb8c9` |
| NCCL library | SHA-256 `5f1c3f10d5ace66d4ba584415bbfe42b6ac1a0a9116a3b81dcbe50516ad924b3` |
| SparkCache CUDA placement library | SHA-256 `d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c` |
| SparkCache behavioral base commit | `9561923405bbb52ac7ff1810d3119a41b7b58b45` |
| SparkCache deployable source | SHA-256 `40de372dda64dd25f493584b2ba3dae81c4350d424d3cf00cfea92452dac170c` |

The behavioral base commit does not contain the DCP changes by itself. The
deployable source digest identifies the complete uncommitted package tree used
for these runs.

The target was
`local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc`.
Its configuration and safetensors index SHA-256 values were
`676382abd1e90a6c85f0c8f33d45441ecd45fd514fd7b63ce5610e732d8e4996` and
`0d1d9e6b226e76520e182de10d4e7194cc885c5cb1bf885bb90de1916ce312cb`.

The BF16 draft was
`incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410`
at depth seven. Its configuration and weight SHA-256 values were
`c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573` and
`b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b`.

Both runs used a 524,288-token model limit, an 8,192-token scheduler budget,
`cp_kv_cache_interleave_size=4`, full-CKV gathering, and
complete `snapshot-v1` publication in the `manager-pages-v2` namespace.
Page-delta and tail-only publication under DCP were not tested.

## Accepted results

| Topology | Publication | Restore | Result |
|---|---|---|---|
| TP4/DCP2 | 9,216 tokens; per-rank snapshot 118.7–132.9 ms; commit 157.3–189.3 ms | Python placement 234.8–264.5 ms for 78,751,393 bytes per rank; CUDA placement 151.1–174.1 ms | Exact `red` |
| TP4/DCP4 | 8,192 tokens; per-rank snapshot 62.4–94.2 ms; commit 147.2–180.7 ms | CUDA placement 118.8–133.7 ms for 62,953,633 bytes per rank | Exact `red` |

## Rejected diagnostic

An earlier TP4/DCP2 run authenticated its manifests and payload bytes but
produced degenerate restored reasoning. It is not accepted evidence.

`MambaSpec` does not expose a `dcp_replicated` attribute even though its block
allocation is replicated across DCP ranks. Treating the recurrent group as
DCP-sharded doubled its logical page width and selected the wrong physical
checkpoint. SparkCache now classifies `recurrent_align` groups as replicated
when the attribute is absent. The corrected runs above require both structural
verification and the exact semantic response.
