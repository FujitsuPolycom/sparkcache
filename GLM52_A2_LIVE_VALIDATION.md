# SparkCache 0.1.0a2 GLM-5.2 live validation

Status: **qualified** for persistent store, coordinated restart, all-rank
manifest discovery, external restore, exact final content, and continued
generation under the exact GLM-5.2 configuration in this record.

Full assistant reasoning-trace equality is an **inconclusive measurement**,
not a qualification claim. The model produced different non-empty reasoning
traces across repeated temperature-zero and fixed-seed requests while keeping
the same exact final content and `finish_reason="stop"`.

## Artifact and conditions

| Attribute | Value |
|---|---|
| Package tag | `v0.1.0a2` |
| Source commit | `42707c727fca6bedf26e7137d0445547ddf2bf03` |
| Release workflow | `32546257578` |
| Wheel SHA-256 | `3345b8c574951a8204377b0c27f53765c84b96ab4f5a8ec1ac147574dba7568b` |
| Source distribution SHA-256 | `55cfd89777a752cb93a9ed3e020a45d04f5665b030f0baf9ac414ce570fa9adc` |
| Hardware | Four directly cabled NVIDIA DGX Sparks |
| Runtime image | `sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513` |
| Model revision | `9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Checkpoint identity | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Parallelism | TP4/DCP4, `ag_rs`, interleave one, fixed MTP4 |
| Model limit | 262,144 tokens |
| Scheduler budget | 4,096 tokens |
| Semantic completion budget | 512 tokens |
| Cache root | Fresh rank-local root per physical rank |

The running `0.1.0a1` containers were preserved as stopped rollback inputs.
Before each `0.1.0a2` launch, the four Q40 exact-state receipts were hashed and
moved to unique hash-bearing backup names. Every launch generated and attested
a fresh receipt before serving health was accepted.

## Cold store

The deterministic 12,000-record request contained 225,555 prompt tokens. It
finished with `finish_reason="stop"`, returned final content exactly
`SPARKCACHE_OK:9540`, and recorded a non-empty 318-character combined
reasoning-and-content body.

Every rank snapshotted and committed 225,536 reusable tokens with cache digest
`fd441fa9535cd5eba7a261b0ef908cebe278d1667fd676d12a3e016c65d4d31a`.
Each rank stored one manifest, 881 chunks, and 1,803,702,724 physical bytes.

| Rank | Snapshot | Durable commit |
|---:|---:|---:|
| 0 | 1,429.9 ms | 11,127.6 ms |
| 1 | 1,130.1 ms | 9,099.3 ms |
| 2 | 1,114.2 ms | 9,223.6 ms |
| 3 | 1,270.2 ms | 9,432.4 ms |

## Coordinated restart and external restore

After the coordinated restart, every rank reported
`checked=1 offered=1 rejected=0`. The bounded generation-scoped quorum
protocol formed four-rank admission, and vLLM reported 225,607 external-cache
queried tokens, 225,536 external-cache hit tokens, and zero local-prefix hit
tokens for the first restored request.

| Rank | Restore time |
|---:|---:|
| 0 | 4,171.2 ms |
| 1 | 3,588.4 ms |
| 2 | 3,697.5 ms |
| 3 | 3,172.2 ms |

The restored request returned final content exactly `SPARKCACHE_OK:9540` with
`finish_reason="stop"`. A fresh post-restore request returned exactly
`SPARKCACHE_CANARY_OK`, also with `finish_reason="stop"`.

## Reasoning-trace determinism result

The cold combined body SHA-256 was
`37918110cc3c8ee55f97f57bcbf314beda1b4ff2dd1035eb723f9854f654b366`.
One repeated request reproduced that body exactly; another produced an
88-token, 286-character body with SHA-256
`28d5228b234712e7e4de8c88ad9871dc643f821d93d7f6a3143d183b861ab214`.

Three requests with explicit seed zero also produced three different body
hashes and lengths while returning identical final content:

- `42dbdb03d2c0aeb0b3b3a14c94846be2772331ee0a8cacf8c9cce467f042483b`, 305 characters;
- `a6878822f4e9a9639d17e63bfec6b98093cba0b17edbcfb7227befa11a1d222a`, 277 characters;
- `a83ea242c380bf61038a081eb11b470629d1ed288c0dcd1536f6e27897310ac9`, 366 characters.

Conclusion: non-empty-body and truncation checks are valid safety gates, but
full GLM reasoning-trace equality is not a deterministic cache-correctness
oracle under this runtime. Cache qualification therefore rests on all-rank
external restore, exact final content, non-truncated completion, and fresh
continued generation. Reasoning-body equality remains an inconclusive
diagnostic.

## Limitations

This record qualifies only the exact `0.1.0a2` wheel and GLM TP4/DCP4 lane.
It does not qualify the `0.1.0a2` DeepSeek TP2/DCP1 or TP4/DCP1 profiles, a
different runtime image, a different checkpoint, another scheduler budget,
streaming snapshots, or native restore.
