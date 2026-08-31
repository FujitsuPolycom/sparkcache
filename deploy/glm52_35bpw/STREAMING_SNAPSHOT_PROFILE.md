# GLM-5.2 streaming-snapshot profile

**Status: research-only.** This profile defines the only registered tensor
inventory for SparkCache streaming publication. GPU-free coverage does not
qualify live serving.

## Runtime contract

The profile requires:

- GLM-5.2 with decode-context parallel degree 4;
- 79 target cache sources at 368 bytes per owned token;
- 22 sparse-indexer sources at 132 bytes per owned token;
- colocated MTP state;
- mapped-host staging, ring depth 2, and 64 MiB slots; and
- the exact vLLM block-lease contract selected by the deployment artifact.

`Glm52ReadyViewTranslator` fixes target sources before sparse-indexer sources.
A 1,024-row macro payload is exactly 32,743,424 bytes. The translator splits
that payload into canonical 256-token storage records and derives DCP-owned
logical positions.

The model-serving factory retains every contiguous row alias for the ring
lifetime. It rejects copy-producing or noncontiguous views and any inventory
that differs from the declared source count, stride, order, or draft policy.

## Evidence boundary

GPU-free tests cover scheduler delayed-free behavior, worker lease completion,
preemption, manifest-last publication, abort visibility, and byte-identical
translation of synthetic READY views.

Qualification requires a live cache-off versus streaming-cache comparison on
the exact runtime, checkpoint, topology, and artifact. The comparison must
include time to first token, decode throughput, cancellation, restart, and
continued-generation equivalence.

See [`README.md`](README.md) for the complete model deployment and
[`../../sparkcache/streaming/OPT_IN.md`](../../sparkcache/streaming/OPT_IN.md)
for the generic opt-in contract.
