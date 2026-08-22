# Historical validation: SparkCache 0.1.0a1 candidate wheel 8e111d42

Date: 2026-08-21.

Document role: historical evidence. This record identifies a pre-release
candidate whose wheel SHA-256 begins with `8e111d42`. The candidate is
**qualified** only for the bounded DeepSeek-V4 conditions below; its GLM-5.2
store/restore result is **research-only**. This record does not describe the
published SparkCache 0.1.0a1 distributions.

[`MULTI_MODEL_LIVE_VALIDATION.md`](MULTI_MODEL_LIVE_VALIDATION.md) is the
canonical qualification record for the published SparkCache 0.1.0a1 wheel,
whose SHA-256 begins with `87c17d8d`.

## Distribution identity

| Artifact | Identity |
|---|---|
| Wheel | `sparkcache-0.1.0a1-py3-none-any.whl` |
| Wheel SHA-256 | `8e111d42f53e823f10179fbad96235aa0bad6ca8791d729494c12fadb1acedaf` |
| Deployable source-tree SHA-256 | `5d42f9fd41c0b9483ba2ba958f9f62cfaee67741f6839d0bbbec11497b8535d1` |
| Connector module | `sparkcache.spark_context_cache_connector` |

The wheel passed `twine check`, an isolated dependency-free installation
probe, runtime-resource inspection, and exact-image imports on all six hosts
used by the tests. The runtime wheel contains no repository test modules or
native research probes.

## DeepSeek-V4 TP2/DCP1 qualification

### Conditions

| Property | Value |
|---|---|
| Hosts | two NVIDIA DGX Spark systems |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Runtime image | `sha256:d316c96d7e1b77ea5459e778c869289c59f7d8a55f3ce274a2082ba48c1be9f6` |
| Checkpoint identity | `6c8f3d2d3b48707541b88f32f22ef3f0f8a6b57d8523281e2b8d3cdb0ae9a023` |
| Parallelism | TP2/DCP1 |
| Connector-configuration SHA-256 | `207382c80ff9b62f20f0818625e9ba97da88789af610428ad4f5ebc386ba64b4` |
| Capacity | 200 GiB high, 180 GiB low per rank; TTL disabled |

### Measurements and result

- A deterministic 73,774-token fresh request returned exactly
  `SPARKCACHE_OK:9540` in 44.41 seconds.
- Both ranks stored 73,728 tokens. Snapshot time was 538.2 ms on rank 0 and
  488.7 ms on rank 1; durable commit time was 2,697.2 ms and 2,877.5 ms.
- After a coordinated engine restart, each rank checked and offered one
  manifest and rejected none.
- The manifest-inventory publication request, 73,728-token restore, exact
  semantic comparison, and post-restore canary completed in 2.20 seconds.
- Restore time was 427.2 ms on rank 0 and 491.6 ms on rank 1.
- External-prefix metrics reported 73,814 queried tokens and 73,728 hit
  tokens. The restored response was exactly `SPARKCACHE_OK:9540`; the canary
  was exactly `SPARKCACHE_CANARY_OK`.

Conclusion: candidate wheel
`8e111d42f53e823f10179fbad96235aa0bad6ca8791d729494c12fadb1acedaf` is
qualified for this exact DeepSeek-V4 TP2/DCP1 deployment. The evidence does
not qualify another wheel, model, checkpoint, runtime image, or topology.

## GLM-5.2 TP4/DCP4 bounded result

### Conditions that passed

- The wheel was hash-verified and imported from its immutable installation on
  all four NVIDIA DGX Spark hosts.
- Four stopped candidate containers passed inspection parity against the
  qualified source-backed GLM-5.2 3.5-bpw serving configuration.
- All four ranks started with the package-qualified connector, regenerated
  image-bound exact-state receipts for CUDA graph capture sizes 1 through 40,
  captured the required piecewise and full graphs, and reached API health.
- Four-rank cache inventory and bounded-capacity reporting passed.

### Interrupted measurement

A deterministic 12,000-record fresh request advanced to approximately 155,700
live prompt tokens. Two ranks then disappeared simultaneously from both their
management network and the direct-ring network. The remaining ranks stayed at
96% GPU utilization while waiting on the broken collective, and the request
returned HTTP 500 after repeated 60-second shared-memory broadcast warnings.
No semantic reference, manifest, or completed cache entry was produced.

Conclusion: package installation, canonical connector loading, GLM-5.2 model
startup, exact-state attestation, and graph capture passed for candidate wheel
`8e111d42f53e823f10179fbad96235aa0bad6ca8791d729494c12fadb1acedaf`.
Persistent store and restore are **research-only** for that artifact and
deployment because host loss interrupted the required gate. The observation
does not identify a wheel, connector, cache, or model-startup defect.
