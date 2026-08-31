# SparkCache 0.1.0a1 release qualification

Status: **qualified** for the exact artifacts and serving configurations in
this record.

## Release artifact

The protected GitHub Actions release run
[`32535319768`](https://github.com/FujitsuPolycom/sparkcache/actions/runs/32535319768)
built the distributions from tag `v0.1.0a1` at commit
`5344192526d328b5cda3417c857b7ffb048fca8a`.

| Artifact | SHA-256 |
|---|---|
| `sparkcache-0.1.0a1-py3-none-any.whl` | `87c17d8dab5052f5a7833349dc9b99b76a3b6531ca6f0d3deff812f724fecdcc` |
| `sparkcache-0.1.0a1.tar.gz` | `8a463a64c55d03d4084d9e364767c510c43b109b78f4de6f99ac9131eef6bee6` |

The exact wheel was installed on every physical rank before live validation.
Publication promoted the stored workflow artifact without rebuilding it. A
clean download from PyPI reproduced both hashes, and a fresh virtual
environment imported package version `0.1.0a1` from its `site-packages`
directory.

The common serving scheduler budget was
`--max-num-batched-tokens 4096`. A value of `8192` is **unsupported** until the
exact model profile passes its own cold-store, coordinated-restart,
external-hit, and post-restore canary smoke.

## Common storage and failure checks

Before model serving, every rank passed GPU-free import, physical-capacity,
invalid-manifest, and corruption checks with the release wheel:

- capacity maintenance reduced an 8,417,280-byte fixture to 2,105,344 bytes,
  reclaimed 6,311,936 bytes, evicted two manifests, deleted three chunks, and
  collected one orphan;
- an invalid manifest was removed during capacity maintenance and its
  unreferenced chunk was deleted, covering defect `D-13`;
- a bit-flipped disposable entry was reported as `corrupt` and invalidated,
  while the source entry remained unchanged.

These checks require a state that cannot be proven correct to become a cache
miss. They do not authorize serving unverified bytes.

## DeepSeek-V4-Flash-0731 TP2/DCP1

| Attribute | Qualified value |
|---|---|
| Hardware | Two directly cabled NVIDIA DGX Sparks |
| Runtime image | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Checkpoint identity | `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f` |
| Model limit | 131,072 tokens |
| Scheduler budget | 4,096 tokens |
| Sequence limit | 6 |
| KV allocation | 17,179,869,184 bytes/rank, `fp8_ds_mla` HMA pages |
| Parallelism | TP2/DCP1, DSpark K5 greedy speculation |
| Capacity policy | 200 GiB high, 180 GiB low, TTL disabled |

A cold 73,774-token deterministic prompt returned exactly
`SPARKCACHE_OK:9540`. Both ranks snapshotted and committed 73,728 reusable
tokens with digest
`1b2a4d1b6faf7685d1641e298f16c5ef97ad40a279c0a451ab4c0438d2922925`,
288 chunks, and 300,904,448 stored bytes per rank.

After a coordinated restart, both ranks reported
`checked=1 offered=1 rejected=0`. They restored the entry asynchronously in
459.8 ms and 517.0 ms. vLLM reported 73,814 external-cache queried tokens and
73,728 external-cache hit tokens. The response remained exact and the fresh
post-restore canary returned `SPARKCACHE_CANARY_OK`.

## DeepSeek-V4-Flash-0731 TP4/DCP1

| Attribute | Qualified value |
|---|---|
| Hardware | Four directly cabled NVIDIA DGX Sparks |
| Runtime image | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` |
| Checkpoint identity | `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f` |
| Model limit | 524,288 tokens |
| Scheduler budget | 4,096 tokens |
| Sequence limit | 32 |
| KV allocation | 34,359,738,368 bytes/rank, `fp8_ds_mla` HMA pages |
| Parallelism | TP4/DCP1, DSpark K5 greedy speculation |
| Capacity policy | 200 GiB high, 180 GiB low, TTL disabled |

A cold 73,774-token deterministic prompt returned exactly
`SPARKCACHE_OK:9540`. All ranks snapshotted and committed 73,728 reusable
tokens with digest
`c17e6fbaefea740b4a83890f20d0e72e792cf9db9429633340e3c48d41d02d1c`,
288 chunks, and 300,908,544 stored bytes per rank.

After a coordinated restart, all ranks reported
`checked=1 offered=1 rejected=0`. Physical ranks 0 through 3 restored the entry
asynchronously in 483.9 ms, 413.9 ms, 443.0 ms, and 494.6 ms. vLLM reported
73,814 external-cache queried tokens and 73,728 external-cache hit tokens. The
response remained exact and the fresh post-restore canary returned
`SPARKCACHE_CANARY_OK`.

## GLM-5.2 EXL3 3.5-bpw TP4/DCP4

| Attribute | Qualified value |
|---|---|
| Hardware | Four directly cabled NVIDIA DGX Sparks |
| Runtime image | `sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513` |
| Model revision | `9ab9579774cc432df91567a36f6e9e863e0d4c9f` |
| Checkpoint identity | `9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd` |
| Model limit | 262,144 tokens |
| Scheduler budget | 4,096 tokens |
| Sequence limit | 8 |
| KV allocation | 9,250,000,000 bytes/rank, dynamic `nvfp4_ds_mla` rows |
| Parallelism | TP4/DCP4, `ag_rs`, interleave one, fixed MTP4 |
| Graphs | `FULL_AND_PIECEWISE`, query rows 1 through 40 |
| Capacity policy | 200 GiB high, 180 GiB low, TTL disabled |

Startup registered 101 persistent layers: 79 target-KV layers and 22
sparse-indexer layers. The Q40 runtime refuses to replace an existing exact-
state receipt. Before each coordinated restart, all four containers were
stopped, every receipt hash was recorded, and each receipt was moved to a
unique hash-preserving backup name. Every launch generated and attested a
fresh Q40 receipt before serving health was accepted.

A cold 225,555-token deterministic prompt returned exactly
`SPARKCACHE_OK:9540`. Every rank snapshotted and committed 225,536 reusable
tokens with digest
`fd441fa9535cd5eba7a261b0ef908cebe278d1667fd676d12a3e016c65d4d31a`,
881 chunks, and 1,803,702,724 stored bytes per rank.

After a coordinated restart, all ranks reported
`checked=1 offered=1 rejected=0`. DCP ranks 0 through 3 restored the entry
asynchronously in 3,954.2 ms, 3,385.0 ms, 3,649.6 ms, and 3,722.4 ms. vLLM
reported 225,633 external-cache queried tokens, 225,536 external-cache hit
tokens, and zero local-prefix hit tokens. The all-rank inventory prime returned
`2`, the long response remained exact, and the fresh post-restore canary
returned `SPARKCACHE_CANARY_OK`.

## Conclusion and limitations

The exact `0.1.0a1` wheel is qualified for durable store, coordinated engine
restart, all-rank manifest discovery, external restore, exact semantic output,
continued generation, capacity maintenance, and corruption handling in the
three recorded profiles.

This record does not qualify another wheel, source tree, runtime image,
checkpoint, scheduler budget, topology, cache geometry, or vLLM source
contract. DeepSeek DCP2 and DCP4, streaming snapshots, SparkCache CUDA restore in these
profiles, buddy replication, and longest-stored-prefix reuse for a growing
conversation remain unsupported or research-only as stated in `README.md`.
