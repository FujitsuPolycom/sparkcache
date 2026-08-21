# GLM-5.2 EXL3 3.5-bpw TP4/DCP4 SparkCache validation

Status: **qualified**.

This record defines the evidence boundary for a live SparkCache deployment on
four NVIDIA DGX Sparks serving the fixed-MTP4 GLM-5.2 EXL3 3.5-bpw recipe
identified by SparkRing as `R7`, with tensor parallel degree 4 and decode
context parallel degree 4. Qualification remains bound to that model, cache
identity, topology, and runtime contract.

## Source evidence

SparkRing commit
[`41374b5bf2a04d0ebc15b9729c6d0f5422c41e13`](https://github.com/FujitsuPolycom/sparkring/blob/41374b5bf2a04d0ebc15b9729c6d0f5422c41e13/sparkcache/README.md#measured-2026-07-28-live-four-dgx-sparks-dcp4)
published the live result in `sparkcache/README.md`. The document describes a
GLM-5.2 layout containing target KV, sparse-indexer state, and speculative
draft KV, sharded across four DCP ranks and stored on rank-local disks.

## Conditions and result

| Condition | Measurement | Result |
| --- | ---: | --- |
| Four DGX Sparks, TP4/DCP4, fresh prefill and durable store | 32.9 seconds | All four ranks committed |
| Full runtime restart and cold restore | 2.11 seconds | Passed |
| Full runtime restart and warm restore | 1.34-1.42 seconds | Passed; 15-24 times faster than the recorded prefill |
| Mixed cached and uncached concurrency | 16 requests | Zero failures; cached requests were approximately 10 times faster than novel prefills |

## Conclusion

The GLM-5.2 EXL3 R7 3.5-bpw TP4/DCP4 deployment qualified rank-local durable
store, full-restart restore, and mixed-request concurrency. The qualification
does not transfer to a rebuilt image or a changed checkpoint, cache identity,
parallel topology, or runtime source contract; those artifacts require the
the checkpoint, topology, runtime-source, store/restart/hit, semantic, and
concurrency gates against their own immutable identifiers.

`MULTI_MODEL_LIVE_VALIDATION.md` records the 2026-08-21 qualification of the
exact EXL3 R7 3.5-bpw profile with SparkCache source tree
`33fbe426045a64b4c46a957c39ebad7cfc85db35be0925ef77a017bf3e53adec`,
including a 225,536-token four-rank external restore and semantic canaries.
