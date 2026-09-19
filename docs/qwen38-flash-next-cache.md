# Qwen3.8 Flash Next persistent cache

Status: **qualified for the bounded TP2 and TP4 configurations below**. This is
not long-duration or general multimodal accuracy qualification.
The profile is `qwen38-flash-next-hybrid` in
`sparkcache/spark_context_cache_profiles.py`.

The profile uses opaque manager pages for attention KV and aligned recurrent
GDN checkpoints. Registered engine groups determine page shape, layer coverage
and reuse boundaries. External speculative-draft state and boundary hidden
activations are recomputed; the draft checkpoint remains part of cache identity.
TP ranks retain separate storage namespaces because Qwen attention KV is sharded,
not replicated across TP ranks.

## Geometry and identity

Persistent chunks contain 32 logical tokens. The tested R37 TP2 configuration
resolves its requested 16-token blocks to 2,848-token physical manager pages;
32 divides both this physical geometry and the logical alignment relationship.
The generic divisibility validator remains enabled. Storage and lookup still
require valid complete page boundaries; a 32-token chunk does not imply that
every 32-token prefix can be restored.

The quantization layout `qwen38-flash-next-hybrid-block-pages-v1` and positional
layout `qwen38-flash-next-mrope-v1` create a distinct namespace from GLM and
DeepSeek profiles. Existing profile identities and wire formats are unchanged.
Use verified checkpoint digests and a separate root for each serving composition.

## Validation requirements

The asynchronous manager-page capture interface admits up to 64 groups and
256 layer sources. Python and CUDA bindings must advertise the same group
capacity; a 16-group library is rejected. Descriptor structure sizes and the
on-disk page format are unchanged. Applications must use the matching rebuilt
snapshot library and its checksum, not an inherited binary with a different
capacity. GPU-free Python and C++ tests cover group ordering and capacity
boundaries; they do not prove CUDA copy correctness.

- Verify the installed vLLM connector-job source contract and native libraries.
- Publish an exact-answer prefix, restart both processes, and prove disk restore
  occurred before accepting the answer as restore correctness evidence.
- Verify incompatible input and corrupted state become misses and recomputation.
- Exercise image/video requests without cross-media cache aliasing.
- Confirm bounded cache work does not retain unsafe page references or prevent
  serving progress, and record host memory during the workload.

`sparkcache/test_qwen38_profile.py` covers layout separation, sharded ownership,
and admission of logical and physical TP2 page geometry. These GPU-free checks
do not establish runtime persistence correctness.

## Bounded GPU evidence

### Four-node QAD persistence and corruption recovery

The [TP4 qualification record](qwen38-tp4-cache-qualification.json) binds
SparkRing 2026.09.3, Qwen QAD revision `629bc321`, TP4/DCP1 and the matching
64-group snapshot library. The packaged SparkCache runtime source is identical
to the Qwen-support implementation in this repository; the record identifies
the full commits, image digest and configuration.

Two independent text fixtures restored 7,200 tokens each on all four physical
ranks after every serving process restarted, with correct exact answers.
Changing one byte in only the first fixture's rank-zero payload caused an
explicit SHA-256 rejection. The scheduler rescheduled 7,200 affected tokens;
the request returned the correct answer with zero cached-token credit. The
independent second fixture still restored 7,200 tokens on all four ranks.

The fault was applied with all workers stopped, only inside an isolated store
whose container mount, ownership marker, manifest and payload hashes were checked.
Production cache entries were not modified. This establishes bounded TP4 text
persistence and failed-restore recomputation, not performance, general media
accuracy, full-context pressure or request-salt isolation.

### Two-node R37 configuration

The serving composition used vLLM tree
`8e0bb7da60e882c095385222d5878981a2666b60` and B12X tree
`bd931d9e6b22ae21b92af0ab66ac45281658bf2c` over the SparkRing R37 image.
The checkpoint was `local-inference-lab/Qwen3.8-Flash-Next-NVFP4` revision
`ada4da32a583a78aa47299f45a70603c950490b8`. Tests used TP2/DCP1, native 262144
context, 16 sequences, 8192 batch, MTP3, 24 GiB FP8 KV per rank, and media
limits of three images and one video with 16-frame sampling.

| Condition | Measurement | Result |
|---|---|---|
| Text cold publication | 50804 prompt tokens, 48416 published; 19.81 s | Three retrieval keys correct; both ranks committed |
| Same prompt after both processes restarted | 48416 cached tokens; 3.98 s | Three keys correct; both ranks logged disk restore |
| First text key changed | Zero cached tokens | Changed key returned correctly |
| Three images and a six-second video with filler | 13710 prompt tokens, 11392 published | Image colors and video order correct |
| Same media after both processes restarted | 11392 cached tokens; 4.93 s | Colors and temporal order correct |
| Blue image replaced with yellow | Zero cached tokens | Yellow returned correctly |
| One bit flipped in a rank-zero text object | SHA mismatch, scheduler rescheduled 48416 tokens | Full recomputation, zero cached tokens, correct answer |
| Sixteen concurrent short JSON requests | 16/16 exact matches | Requested concurrency completed |
| Near-limit text retrieval | 257504 prompt tokens; 124.85 s | All three retrieval keys correct |

Request times include generated output and are single observations, not matched
performance benchmarks. Synthetic media checks do not qualify arbitrary video
lengths, resolutions or concurrent-media memory use. Corruption was injected
only into an isolated synthetic cache entry. Capture and publication queues
were empty after the checks. Long-duration stability remains unqualified.
