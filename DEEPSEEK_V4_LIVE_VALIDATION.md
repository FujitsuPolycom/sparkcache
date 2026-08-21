# DeepSeek-V4 block-page live validation

Status: **qualified** for the exact TP2/DCP1 configuration and evidence scope
recorded below.

## Scope

On 2026-08-20, the `deepseek-v4-fp8-hma` profile was validated on two ARM64
DGX Spark hosts running one TP2/DCP1 vLLM service. The runtime was
`0.1.dev1+ge2666d9a6.d20260810` plus the asynchronous-rollback and
hybrid-memory-allocator (HMA)
load-failure recovery scheduler patches. The served checkpoint was
`deepseek-v4-flash-0731` with FP8 KV cache and DSpark speculative decoding.

The worker registered 170 tensors across five heterogeneous block-size cache
groups exposed through vLLM's HMA API. Their block sizes were 256, 64, 64, 4,
and 8 tokens. The 256-token group keeps full history;
the other groups expose reusable windows of 128, 128, 8, and 128 tokens.
SparkCache stored opaque pages only for the full-history range or the final
reusable window at the exact persistent boundary. Null or duplicate physical
block identifiers fail closed.

Both hosts independently built the ARM64 derived image
`sha256:e91b4788760789741060f6345e96ba387cb00b35d6e1ce865c5963404409b147`
from rootfs input
`faeca9770b27e15af2d1a81d167d211b34995f67b86cb7b464520f2fbf4fb07f`.
The image contains connector source
`05e3e4120cb8d6d799d434a4c49563aecaf3bfcd979d8fecc46dee0e6bbd1479`
and runs with four data-only mounts; maintained runtime overlays are baked
into the image.

The bounded-capacity gate used derived image
`sparkring/dsv4-sparkcache:aa0bbca-capacity-v7`, image ID
`sha256:780ddbb41c1fa34be0680b29da70b2b413959d52f684d5f919757a2dc79fcc37`,
on both hosts. It retained the same model, vLLM, TP2/DCP1, FP8-KV, and DSpark
configuration as the semantic gate.

## Gate and result

The deterministic semantic request contained 6,958 prompt tokens. Two numeric
facts were separated by thousands of padding tokens, and the model had to add
them and return exactly `SPARKCACHE_OK:9540`. SparkCache aligned the reusable
prefix to 6,912 tokens and stored group page counts `[27, 2, 2, 2, 16]` under
context digest `af19bc736bb6...` on both ranks.

Rank 0 detached the GPU pages in 65.7 ms and committed them in 357.7 ms. Rank
1 detached them in 80.0 ms and committed them in 372.5 ms. Each rank's 27
content-addressed chunks carried 42,294,661 encoded bytes (about 40.3 MiB).

Both vLLM containers were stopped and started. Startup discovery reported one
checked and offered manifest per rank. A sub-threshold arithmetic request
returned `5` and published the post-restart worker quorum. Repeating the
semantic request produced:

- scheduler external-cache hit for 6,912 tokens;
- rank 0 verified restore in 116.5 ms;
- rank 1 verified restore in 108.4 ms;
- `SPARKCACHE_OK:9540`, byte-identical to the pre-restart completion;
- a fresh post-restore arithmetic canary returned exactly `42`;
- `Explain SparkCache in three sentences.` returned three coherent sentences;
- both rank containers healthy after the request.

## Bounded NVMe capacity gate

Each rank-local root `/cache/sparkcache-dsv4` was configured with
`spark_cache_max_bytes=214748364800` (200 GiB),
`spark_cache_low_watermark_bytes=193273528320` (180 GiB), and
`spark_cache_ttl_seconds=0` (TTL disabled). Startup maintenance ran before
manifest discovery and reported:

| Rank | Allocated bytes after maintenance | Bytes reclaimed | Manifests evicted | Chunks collected | Orphan chunks collected |
|---:|---:|---:|---:|---:|---:|
| 0 | 3,582,099,456 | 1,608,474,624 | 0 | 19 | 19 |
| 1 | 3,582,091,264 | 1,608,470,528 | 0 | 19 | 19 |

Both reports satisfied the 200 GiB high watermark. Discovery then checked
and offered four manifests per rank. Repeating the 6,912-token semantic entry
produced `SPARKCACHE_OK:9540` through an external-cache hit, with verified
restore times of 87.0 ms on rank 0 and 94.8 ms on rank 1; a fresh canary
returned `42`. The cleanup therefore reclaimed unreferenced data without
invalidating reusable entries.

After a fifth manifest was published, both bounded containers were restarted
together again. Startup maintenance measured 3,624,464,384 allocated bytes on
rank 0 and 3,624,456,192 on rank 1, reclaimed nothing, and discovery offered
all five manifests per rank. Two distinct 6,912-token entries written before
that restart then restored successfully: digest `af19bc736bb6` in 71.4 ms on
rank 0 and 121.8 ms on rank 1, and digest `a79c32a50f0e` in 86.0 ms on rank 0
and 67.5 ms on rank 1. Both returned exactly `SPARKCACHE_OK:9540`; their
post-restore canaries returned `42`.

The reproducible-image qualification used
`sparkring/dsv4-sparkcache:4e8e085-capacity-v8`, image ID
`sha256:d316c96d7e1b77ea5459e778c869289c59f7d8a55f3ce274a2082ba48c1be9f6`
on both hosts. This image includes an explicit high-to-low hysteresis
regression:
once pre-cleanup physical use crosses the high watermark, orphan or TTL
cleanup cannot stop reclamation between the low and high watermarks. During
the validation boot, rank 0 measured 3,754,733,568 bytes and rank 1 measured
3,754,725,376 bytes, reclaimed nothing, and each rank offered seven manifests.
The persisted `a79c32a50f0e` entry restored 6,912 tokens in 90.2 ms on rank 0
and 105.7 ms on rank 1, returned exactly `SPARKCACHE_OK:9540`, and left the
`42` canary coherent.

The deterministic physical-NVMe capacity gate
`deploy/deepseek_v4/capacity_gate.py` was also run in a separate empty root on
both hosts from that qualified image. It published three entries, verified that
maintenance skipped while another transaction remained open, aborted that
transaction to leave an orphan chunk, and then applied a 5 MiB high/3 MiB low
policy. Both hosts reported the same result:

- allocated bytes decreased from 8,417,280 to 2,105,344;
- 6,311,936 bytes were reclaimed;
- the two least-recently-used manifests were evicted and the newest survived;
- three chunks were deleted, including the aborted transaction's orphan; and
- the resulting store satisfied both the high and low watermarks.

The gate exercises actual filesystem allocation accounting, LRU selection,
open-transaction exclusion, and orphan collection. It does not simulate
filling a 200 GiB root or qualify model-serving load behavior.

## Limitations

This evidence qualifies one TP2/DCP1 development appliance. Model-serving load
behavior and arbitrary DeepSeek-V4 deployments are unsupported. Block-page
storage has bounded NVMe maintenance for end-of-prefill asynchronous
snapshots. Native restore, streaming snapshots, and DCP-sharded block pages
are unsupported by this profile. The qualified service used the Python
asynchronous restore path; every chunk was
checksum-verified before its pages were installed.

The high watermark triggers reclamation after publication; it is not an
allocation reservation. The qualified layout uses one publisher per
rank-local root. Multiple publisher processes sharing one root have
independent conservative byte estimates and can transiently exceed the high
watermark until periodic maintenance reconciles the physical store. The
reuse policy remains part of cache identity, so entries created by an
incompatible full-history encoding clean-miss. Enabled maintenance reclaims
their chunks when no compatible manifest references them.
