# Asynchronous capture and persistent restore on four GB10 GPUs

Status: **qualified** for the cases and revisions below. The
[machine-readable record](connector-job-gb10-tp4-dcp1.json) contains physical-rank
capture evidence, post-retirement counters, restored-prefix digests, and source
receipt hashes.

## Conditions

The model is `local-inference-lab/GLM-5.3-Flash-NVFP4-Spark`, revision
`df116c4fb16b1d37ae43d2cfd624de26ffbc832e`, served on four GB10 GPUs with TP4,
DCP1, three speculative tokens, 8192-token forward batches, and 512-token cache
blocks. Continuation coalescing and token-sharded mHC prefill are enabled.
Compact index-cache gathering is disabled. SparkCache uses asynchronous
connector-job capture with two 512-MiB native slots per physical rank and CUDA
restore into verified destination pages.

| Component | Revision |
| --- | --- |
| SparkCache | `2bc05bc9e94a4344758e48db36f69a46dafa6946` |
| vLLM | `df62335d8248587f8d3fd1d9a234d1c162a9b84d` |
| B12X | `0b6d61c37c87ae49d2f9d20d38b9da023146e243` |
| Container image | `sha256:cd92adc4436c61290dbecc35362db447c7ae69e1d03bdb99704af0ed6c517b38` |

## Results

| Check | Measurement | Result |
| --- | --- | --- |
| Request completion | Four cold 8K/16K requests with one generated token | All four ranks completed native capture; pending work, delayed work, retained pages, and uncertain reads returned to zero after each case. |
| Request cancellation | Two active 8K/16K requests cancelled during generation | The engine logged cancellation; all four ranks completed capture retirement and the same ownership counters returned to zero. |
| Persistent 8K restore | Captured prefix loaded after all model processes restarted | 7680 tokens restored; every rank matched the stored prefix digest; the exact-answer check passed. |
| Persistent 16K restore | Captured prefix loaded after all model processes restarted | 15872 tokens restored; every rank matched the stored prefix digest; the exact-answer check passed. |

The results establish that capture can outlive request completion or cancellation
without retaining source pages indefinitely, and that these persisted prefixes
remain usable across a full model-process restart. Tail recomputation accounts
for the difference between prompt length and restored length.

## Limits

The additional guards described in [whole-prefix restore
ownership](../PRIVATE_RESTORE_SAFETY.md) have CPU coverage but were not installed
in this image. Their integrated GPU validation remains required. The standalone
JJ recovery revision `9b87df5d47b9c7163d1105ac5ea8c0a088baafc9` has CPU and source-contract
coverage; this record does not qualify that different composition.

These checks do not qualify SparkCache on TP2, cross-composition state reuse,
multimodal requests, arbitrary concurrent eviction, or every native failure
condition. They make no throughput or model-quality claim.
