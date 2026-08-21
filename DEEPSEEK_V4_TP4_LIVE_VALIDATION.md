# DeepSeek-V4-Flash-0731 TP4/DCP1 live validation

Date: 2026-08-21. Status: **qualified** on four DGX Spark hosts connected as a
direct ring at TP4/DCP1. DCP2 and DCP4 are unsupported by this evidence.

## Frozen inputs

| Input | Qualified value |
|---|---|
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Served name | `deepseek-v4-flash-0731` |
| Complete checkpoint digest | `bd6b0117ca28997acc9f22022814bb6bc50b5c3e1bc466d148b1d45067fe714f` |
| Checkpoint inventory | 74 files, 166,898,661,074 bytes on every rank |
| Image manifest | `ghcr.io/fujitsupolycom/gb10-vllm-serving@sha256:6fc26fdad81a18f0fff67ce0a05f6d90165625ea2e1cac8a6f39bfb462017028` |
| Local image ID | `sha256:50036224411e5ef04d651730c56d794111991b37981ec76ca81b66ea7d35dae7` on all four ranks |
| vLLM | `0.1.dev1+ge2666d9a6.d20260810` |
| Parallelism | TP4, PP1, DCP1, stock NCCL over the direct ring |
| Speculation | DSpark speculative decoding (`K5`, `b12x`) with greedy draft sampling |
| KV | `fp8_ds_mla`, 32 GiB per rank, primary opaque-page block size 256 through vLLM's hybrid-memory-allocator (HMA) API |
| Request limit | 524,288 tokens; 32 sequences |
| SparkCache source digest | `66fa46e2f0eae14c0973661df4f836b8e278e54488e16319e0037d75c3fc2708` |
| Scheduler overlay | stock `1ea341...` -> patch 011 `d4ebec...` -> patch 031 `2f34aa...` |
| vLLM config overlay | `fbc581...` -> patch 020 `71c4f9...` |
| Capacity | 200 GiB high / 180 GiB low per rank-local NVMe root; TTL disabled |

All four checkpoint content manifests were byte-identical. All four generated
overlay receipts were byte-identical (`dfc1ece2...`), and all serving binds
were inspected before start: model, source, scheduler, and config read-only;
cache and JIT roots read-write and disjoint.

## Cache-off baseline and 6,912-token persistence gate

The qualified image served first with SparkCache disabled. Health,
model metadata, and exact `BASELINE_OK` output passed. The cache-enabled stack
then served exact `SPARKCACHE_LIVE_OK` output.

The deterministic semantic miss contained 6,958 prompt tokens. Every physical
rank snapshotted and committed the same 6,912-token aligned digest
`3f84d4b97a5d...`; each rank-local entry occupied 42,294,661 encoded bytes
(about 41 MiB allocated). After a coordinated four-container restart:

- all four workers reported `checked=1 offered=1 rejected=0`;
- the scheduler reported a full four-rank quorum and external hit;
- vLLM reported 6,912 external-prefix-cache hit tokens;
- rank restore times were 79.8, 90.9, 93.0, and 113.0 ms across repeated
  qualification runs;
- the long answer remained exactly `SPARKCACHE_OK:9540`; and
- the immediate fresh echo canary remained exactly
  `SPARKCACHE_CANARY_OK`.

## 36,864- to 147,456-token restart/restore sweep

Three more deterministic entries were stored, followed by one full four-rank
restart. Every rank rediscovered all four manifests
(`checked=4 offered=4 rejected=0`).

| Prompt tokens | Aligned restored tokens | Snapshot/rank | NVMe commit/rank | Restore/rank | End-to-end hit gate |
|---:|---:|---:|---:|---:|---:|
| 36,910 | 36,864 | 189–240 ms | 1.34–1.57 s | 207–298 ms | 1.56 s |
| 73,774 | 73,728 | 369–460 ms | 2.63–3.03 s | 335–412 ms | 1.64 s |
| 147,502 | 147,456 | 1.08–1.31 s | 6.09–6.49 s | 734–888 ms | 2.45 s |

The sweep produced 258,048 external-prefix-cache hit tokens, zero local-prefix
hit tokens for those restored prefixes, four-rank quorum for every digest,
byte-identical semantic answers, and an immediate exact sentinel after each
restore. The four entries occupied about 1.086 GiB per rank.

## Capacity and corruption gates

The physical-NVMe capacity gate passed identically on all four hosts. It
reduced 8,417,280 allocated bytes to 2,105,344, reclaimed 6,311,936 bytes,
evicted the two oldest manifests, deleted three chunks including one orphan,
retained the newest entry, and satisfied the low-watermark target.

The corruption gate copied the smallest live entry into a disposable root:
27 chunks, 42,294,661 bytes. The copy verified healthy; flipping one byte made
lookup return `corrupt`; invalidation removed the copied manifest and damaged
chunk; the serving source root was unchanged.

## Deployment constraints established by qualification

1. Image manifest `6fc26f...`, which resolves to local image `500362...`, is
   the qualified image. Image `02881...` is unsupported for this profile
   because its Quack/CUTLASS pair fails at first forward with
   `cute.core.ThrMma` missing.
2. The containerized overlay command must run as a module from `/src`; running
   the file path directly cannot import the sibling `deploy` package.
3. The overlay builder must initialize its temporary directory as its own Git
   root. Otherwise, a temporary directory nested under an unrelated checkout
   makes `git apply` silently target the outer repository.
4. Base-image GLM exact-graph environment names, including variables for the
   40-query-row CUDA graph capture state (`Q40`), survive Docker inspection. The
   DeepSeek transform strips them from its source contract and records them in
   `SPARKRING_EXPLICITLY_UNSET`; the direct vLLM entrypoint ignores GLM image
   attestation state.
5. Post-restart discovery reports reach the scheduler on the first worker
   round trip. The hit gate must first send a short request that publishes
   every rank's cache inventory to the scheduler (a quorum prime), or the
   long request correctly fails closed to recompute.
6. The DSpark build can answer the same trivial arithmetic prompt differently
   under async and synchronous scheduling (`42` versus `34`). Arithmetic is
   therefore not a valid cache-corruption oracle. Exact echo sentinels are
   stable before and immediately after external restores.

## Qualification-time service identifiers

The qualification used containers `deepseek0731-sparkcache-r0` through `r3`.
Rank 0 exposed API port 8100; the other ranks were headless. The collective
rendezvous used port 29600 on the rank-0 direct-ring interface. These dated
container names record test conditions and do not assert that the service
remains available.

```bash
ssh <rank-0-management-host> \
  'docker logs -f --tail 200 deepseek0731-sparkcache-r0'
```
