# vLLM patch overlay

This directory publishes the two independently written vLLM compatibility
patches required by SparkCache:

- `010-sparkcache-async-rollback.patch` resets speculative-output placeholder
  state when an asynchronous KV restore fails and the request is rescheduled.
- `020-sparkcache-vmm-exemption.patch` permits SparkCache with PyTorch
  expandable segments because this connector does not register KV-cache GPU
  memory with an external device.

Both patches are pinned fail-closed to official
`vllm-project/vllm@fcc614141e5e9ab18cb304c476f7feed2a9552e3`.
`preimages.json` records the exact upstream file hashes. The public runtime
builder verifies each preimage before applying a patch and refuses fuzz or an
unexpected source tree.

The patch directory covers only SparkCache ownership and allocator
compatibility. Model kernels, low-bit KV formats, speculative decoding, and
hardware-specific vLLM integrations are outside this repository's patch
contract.
