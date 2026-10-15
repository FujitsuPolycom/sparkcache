# DeepSeek-V4.1-Flash deployment tooling (research-only)

Status: **research-only**. Nothing in this directory has run against a
serving deployment; no SparkCache entry, gate report, or measurement exists
for DeepSeek-V4.1-Flash. Hardware qualification is a separate, authorized
step on the DGX Spark installer profile that SparkRing maintains.

The `deepseek-v41-b12x-hma` model profile in
[`sparkcache/spark_context_cache_profiles.py`](../../sparkcache/spark_context_cache_profiles.py)
stores DeepSeek-V4.1-Flash B12X pages as opaque manager pages. vLLM declares
two cache classes that SparkCache never persists: request-private
sliding-window groups (prefill replay reserve 128 tokens, decoder and draft
layers) and per-request compressor rings. A persistent boundary therefore
stops at least 128 tokens before the end of the prompt and the engine
recomputes the rest, reproducing a cache-off run.

## Checkpoint identity

The checkpoint is `deepseek-ai/DeepSeek-V4.1-Flash` at revision
`dba1be0a40aa45a94ad051997016db3960a90277`. The pins beside this README come
from SparkRing's installer profile `profiles/deepseek-v41-flash-tp4`:
`checkpoint-sha256sums.txt` is that profile's 77-file `SHA256SUMS` manifest
(SHA-256 `89de6bc77d0028b2b61b3cad12885342d0856f4c115c4f9328e91cf025c3d16f`),
`model_config.sha256` is the profile's pinned `config.json` digest, and the
draft block records the profile's DSpark speculative configuration with the
B12X load format and backends. The served KV representation is the model's
`b12x_dsv41` cache dtype tag. The checkpoint's `quantization_config` was not
readable offline; the exact configuration bytes are bound by
`model_config.sha256` instead.

```sh
python deploy/deepseek_v41/checkpoint_identity.py derive
python deploy/deepseek_v41/checkpoint_identity.py verify-root /srv/models/DeepSeek-V4.1-Flash/dba1be0a40aa
```

`derive` recomputes the identity from the pins and compares the recorded
receipt:

```
0b447c086ca74be5e7ed9e98c424637f1f8015ee38fd1e767370415377ed401e
```

`verify-root` hashes a local checkpoint directory and compares every file
with `checkpoint-sha256sums.txt`, refusing missing and extra files. The
derived identity is an input to `CacheIdentity`
(`spark_cache_target_checkpoint_sha256`), not the complete runtime
namespace; the topology digest and profile layouts complete it.

## Gates

| Tool | Proves |
|---|---|
| `semantic_gate.py` | A long cold miss writes a stable reference answer; a restored hit reproduces it and both phases answer the publication canary. Requests pin `thinking: false`. |
| `corruption_gate.py` | A bit-flipped chunk copy is rejected as corrupt and invalidated back to a clean miss, without touching the source root. |
| `capacity_gate.py` | Quota eviction keeps the newest entry under a byte limit and an open transaction is never evicted. |

`python -m pytest deploy/deepseek_v41 -q` runs the CPU tests for this
directory. The semantic gate needs a live serving endpoint; its CPU tests
exercise the injected HTTP seam only.

## What is missing before this profile leaves research-only

- A recorded hardware topology inventory (the conformance fixture in
  [`sparkcache/test_deepseek_v41_topology.py`](../../sparkcache/test_deepseek_v41_topology.py)
  is inferred from the model's spec declarations, not from a rank dump).
- A derived SparkRing image layer and installer profile (SparkRing
  repository work).
- CUDA page restore and asynchronous page capture remain disabled; the
  profile refuses `spark_cache_cuda_restore`.
- A live validation record. No `DEEPSEEK_V41_*` validation document exists
  until a hardware run produces one.
