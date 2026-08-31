# GLM-5.3 PR535 physical-page delta research validation

Date: 2026-08-30

## Status

This record establishes **research-only live evidence** for physical-page
delta publication and SparkCache CUDA restore on one exact GLM-5.3 TP4/DCP1
runtime. A fresh 98,304-token base was extended to 131,072 tokens, all four
ranks published the same bounded delta geometry, and an engine restart restored
the result with exact `blue` output. A later eight-request run returned eight
exact, distinct results while all eight persistent restores shared one
authenticated base read per rank.

This is not a general deployment qualification and does not authorize
`tail-cow-v1` for production. It does not establish another model, checkpoint,
topology, physical-page geometry, vLLM source, or concurrent-reader bound.
Persistent-restore readiness required one model inference after startup because
worker manifest inventories do not reach the scheduler through the HTTP health
endpoint.

## Runtime identity

Identifiers are listed by semantic role so the observation does not depend on
container tags or conversation history.

| Responsibility | Durable identifier |
|---|---|
| vLLM PR535 source | `local-inference-lab/vllm@ead9d8a4e21b3818b21ec6f4d4d94564dd60c3f8` |
| SparkRing runtime composition | `FujitsuPolycom/sparkring@6da4865d440608a46eada50f27b2fff0e698c574` |
| B12X | `local-inference-lab/b12x@b1d541f9e71a35f030d45fae437630fff7507c2a` |
| SparkCache content installed in the runtime | `59ac0b04db6035a9a9d2a52e92405ceaf84daa40` |
| SparkCache source-tree SHA-256 | `21035ceeeab514b573dffc9a8b415b246fad8cd11ad83c633717b37f2bf6dd1b` |
| CUDA placement library SHA-256 | `d57509052b73853bcc8e3c3f47bb81748d87b9cbd8d908fc20d4c79a09aa400c` |
| Local research image SHA-256 | `sha256:becf556650dff79a9959aef371ea861187db248bd0f46c3ebfbd26759e458818` |

The image was not published. The running containers were
`glm53-pr535-sc59ac-c8-01-r0` through
`glm53-pr535-sc59ac-c8-01-r3` on `spark-r0` through `spark-r3`. Rank 0 served
HTTP on port 8015.

## Checkpoint identity

| Role | Repository revision | Runtime identity |
|---|---|---|
| Target | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` | CacheIdentity SHA-256 `a35e6bf2875c1875609b8deaec404c07c6cc80259e4222fc0b51e649498bd6b9` |
| Draft | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` | config SHA-256 `c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573`; weights and CacheIdentity SHA-256 `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` |

The draft used BF16 weights, TP4 loading, and seven speculative proposal
tokens. It was the external DFlash2 checkpoint above, not an MXFP8 draft.

## Serving and cache conditions

- topology: TP4/DCP1, one worker on each of `spark-r0` through `spark-r3`;
- target KV dtype: FP8; target model dtype: BF16;
- B12X attention, MoE, and linear backends; FlashKDA prefill;
- `--block-size 256`, `--max-model-len 262144`,
  `--max-num-batched-tokens 4096`, and `--max-num-seqs 16`;
- split physical target pages: 512 tokens through
  `VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=512`;
- split physical recurrent pages: 512 tokens through
  `VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=512`;
- logical SparkCache digest boundary: 256 tokens;
- publication setting: operator value `tail-cow-v1`, authenticated identity
  value `page-tail-cow-v1`, and `block_pages_v1` storage;
- SparkCache CUDA restore: eight load lanes, eight I/O workers, and one
  256 MiB mapped arena per lane, for a 2 GiB per-rank mapped-arena ceiling;
- rank-local cache limit: 40 GiB with a 32 GiB low watermark;
- fresh host root:
  `/var/tmp/glm53-pr535-sc78-tp4-01/sparkcache-context/pr535-sc78-tp4-01`,
  mounted as `/cache/jit/sparkcache-context/pr535-sc78-tp4-01`;
- one-shot clear token: `glm53-pr535-sc78-tp4-01`.

The 512-token physical settings are part of the authenticated
`quantization_layout`. The public SparkCache head changes neither
`CacheIdentity` wire fields nor digest salts. Incompatible physical geometry
therefore cleanly misses instead of aliasing these entries.

## Fresh publication observations

The cache root was empty before this sequence. Exact semantic checks used
equality, not suffix matching.

| Prefix | Semantic result | Server observation | Snapshot time by rank 0/1/2/3 |
|---:|---|---|---|
| 4,096 | exact `base` | semantic publication canary | 87.2 / 115.2 / 122.9 / 124.9 ms |
| 32,768 | not used as the semantic conclusion | 3,276.9 prompt tokens/s | 727.0 / 652.6 / 601.1 / 704.9 ms |
| 98,304 | exact `base` | 9,829.7 prompt tokens/s | 2,057.4 / 1,814.8 / 1,621.3 / 2,264.9 ms |

The server throughput values are vLLM's ten-second logger observations, not
end-to-end request throughput benchmarks.

## 128K page-delta result

The 131,072-token `blue` entry extended the exact 98,304-token base. Every
rank's committed `sparkcache-page-delta-manifest/v2` recorded:

| Manifest field | Value |
|---|---:|
| Base committed tokens | 98,304 |
| Result committed tokens | 131,072 |
| Base snapshot bytes | 656,949,411 |
| Delta encoded bytes | 252,534,308 |
| Delta objects | 4 |
| Restored result page bytes | 859,160,739 |

The four delta objects were three 67,108,864-byte objects and one
51,207,716-byte object. Manifest-last publication made the result visible only
after its immutable objects were committed.

After stopping and restarting all four engine containers against the preserved
fresh root, the request returned exact `blue` in 1.983 seconds client time. All
four workers reported `outcome: verified` for the same 131,072-token digest.

| Rank | Cache service | Authenticated read | Placement + CUDA completion |
|---:|---:|---:|---:|
| 0 | 1,574.575 ms | 950.894 ms | 115.311 ms |
| 1 | 1,656.597 ms | 1,037.445 ms | 112.067 ms |
| 2 | 1,644.642 ms | 1,014.611 ms | 116.103 ms |
| 3 | 1,662.062 ms | 1,030.706 ms | 116.339 ms |

`Placement + CUDA completion` is the sum of the structured `h2d_submit` and
`cuda_sync` phases. Client time also includes scheduler, live-token, and model
work, so it must not be compared directly with one worker's cache-service
time.

## Concurrent different-root restore

Eight concurrent 16,384-token requests used distinct result roots derived from
one 8,192-token physical-page base. All eight responses matched their exact
expected codewords, and all eight requests entered persistent restore. Client
latency was 1.392026--2.337652 seconds.

Each rank emitted one `sparkcache-page-base-restore-flight/v1` record for the
eight admitted restores: `participants=8`, `physical_base_reads=1`,
`avoided_base_reads=7`, and `base_bytes=100868258`. Every rank reported the
same sharing counts. Private deltas, reconstructed-result authentication, and
CUDA placement remained request-specific. No semantic, checksum, or placement
error was observed. Base-read time was 161.461--188.138 ms across ranks, and
worker cache-service observations were 516.0--990.5 ms.

The workers discovered their persistent manifests during startup, but the
scheduler's manifest quorum remained empty until worker statistics returned
from a real model execution. An HTTP health request does not execute the model.
A five-token prompt with a one-token completion established scheduler readiness
before the recorded cohort. A separate health-only startup run returned all
eight exact results but restored seven and safely recomputed one; which request
recomputed depended on arrival order.

The recorded result establishes bounded direct-CUDA base-read coalescing across
eight different result roots. It also establishes that deployment readiness
must include a small inference before persistent-hit expectations are measured.

## Interpretation and limits

The live result establishes that this exact runtime can publish a bounded
physical-page delta, restart, authenticate the complete base-plus-delta graph,
place only verified result bytes, and preserve exact model semantics. It also
shows that publication scales with the changed physical pages for this
98K-to-128K extension rather than writing another complete result snapshot.

The result does not establish retained host-base caching, production tail-only
safety, C16 behavior, corruption recovery for this specific graph, or another
split-page geometry. It does not establish persistent-hit readiness from HTTP
health alone. Failures to prove graph
identity, integrity, or compatibility remain cache misses and recomputation;
optional cache work must not delay serving.

## Evidence source

This record was transcribed from live `docker inspect` identity labels, the
rank-local committed manifests, exact-response client assertions, and
`sparkcache-restore-timing/v1` worker records. The live cache roots, containers,
model mounts, and JIT roots were preserved after collection. No registry
artifact was published and no production setting was enabled.
