# Flat-history restore cost on GLM-5.3 Flash

Status: **research-only**. These measurements compare SparkCache `50bd6d9`
with the surviving-tail reader and authenticated-header cache at `e389b32`.
They cover one 32-turn synthetic agent session per arm and one same-pair
comparison on two DGX Sparks, Draupnir positions 6-7.

## Conditions

| Item | Value |
|---|---|
| Measurement time | 2026-10-15 00:31-01:21 UTC |
| Model | `local-inference-lab/GLM-5.3-Flash-NVFP4-MXFP8-CSF-QAD` |
| Checkpoint revision | `dec48abd33efa73c3bb7c95b74eee10cad34f9be` |
| Runtime image | `sparkring:sparkcache-gab86b70` |
| Image ID | `sha256:e1607979b7f0f00217c8deae170185a57f693721f1bc88368dcc536181a495f7` |
| Package selection | Each build's SparkCache sources bind-mounted read-only ahead of the installed package on `PYTHONPATH` |
| Profile | `glm53-flash-nvfp4-sparkcache-tp2` |
| Parallelism and scheduling | TP2/DCP1, MTP3, asynchronous scheduling |
| Cache geometry | 2,048-token blocks, 10 GiB KV capacity, two 512 MiB capture slots |
| Ownership contract | Connector-job leases, `vllm-connector-jobs-gab86b70.json` |
| Persistence | CUDA page restore, `tail-cow-v2`, snapshot interval 0, maximum span 262,144 tokens, 32 GiB cache root with 28 GiB low watermark |
| Fabric | Direct ConnectX-7 link between each pair |
| Instrumentation | SparkCache DEBUG logging for `read_source_bytes`; clocks not locked |

The client sent token-ID prompts to `/v1/completions` with temperature 0,
`max_tokens` 64, seed 0, and streamed usage. Each turn reset the GPU prefix
cache, sent the session request, waited for both ranks to commit the boundary,
sent two four-token requests to let the scheduler absorb the commit, reset
the GPU prefix cache again, and sent a fresh request. Prompts did not include
generated model output.

## Same-pair results

The turn-32 entry contained 31 stages and 186,368 tokens; the fresh request
contained 187,889 prompt tokens. Restore and commit timings use the slower
rank. TTFT is client-observed for the fresh request.

| Measurement | Baseline `50bd6d9` | Tail-read/header-cache build `e389b32` |
|---|---:|---:|
| `restore_read` | 18.8 s | 1.5 s |
| Publication commit | 19.7 s | 5.5 s |
| Turn-32 TTFT | 22.8 s | 3.7 s |
| `load_failed` | 0 | 0 |
| Source bytes read, rank 0 | 20,252 MB | 1,407 MB |
| Page bytes placed, rank 0 | 1,331 MB | 1,331 MB |

A separate restart test on positions 4-5 restored the 30-stage,
182,272-token entry. Each cold process read 3,393 MB versus 1,380 MB warm:
2.46 times the read volume. Cold `restore_read` was 5.0-6.1 s versus
1.3-1.7 s before restart. Both retaining and dropping the OS page cache gave
similar cold timings. A repeat request in the same process took 1.0-1.5 s.

## Interpretation and limits

The measurements support reduced restore and publication work for this
configuration. They do not qualify either build for serving. No hardware
measurement of the documentation and deployment-pin preparation is claimed.

Every restore passed SparkCache's digest checks and `load_failed` remained
zero. Independent bit-for-bit restore correctness and output-token equality
were not evaluated. Results from one session per arm do not establish
repeatability across workloads, models, or topologies.

The source record is `sircl-handoff/research/ring8-hw-20261013/` under
`sparkcache-restore-fix/RESULTS.md`. Its `raw/runs/<arm>/requests.jsonl` and
per-turn logs contain the measurements; `scripts/analyze_rf.py` and
`scripts/make_tables_rf.py` generate the tables. These evidence files are
retained outside this repository.

## Stage-limit write cost

Status: **implemented; hardware tuning remains research-only**. The connector
defaults to four `tail-cow-v2` stages; `spark_cache_page_delta_stage_limit_count`
set to `0` disables compaction. Appending a fifth stage writes the complete
result, including that turn's attention tail and recurrent checkpoint, from
CPU publication data. The full result does not pass through the GPU capture slot.

For GLM-5.3-Flash CSF at TP2, full attention costs **6,732 B per new token**
and every stage also writes about **76 MB of recurrent state** (76,323,851 B
in the CPU fixture). A sparse extension of d tokens therefore writes about
6,732*d + 76,323,851 bytes, excluding headers and page rounding.

Compaction at T retained tokens writes about 6,732*T + 76,323,851 bytes.
Relative to a sparse extension, it adds 6,732*(T-d) bytes: the retained
attention prefix is rewritten, while recurrent state is still written once.
Compaction does not erase earlier published boundaries or guarantee disk savings.

Four stages limit metadata traversal and overwritten recurrent history while
amortizing complete writes across five extensions. This default follows the
chain-depth study's tuning direction; it is not a measured optimum. The CPU
benchmark accepts `--stage-limit 4` and `--stage-limit 0` for comparison.

## CPU publication profile

Status: **research-only**. The fixture in
`tools/benchmark_flat_history_restore.py` models the 32 stored boundaries,
34 attention layers and 12 recurrent layers above. Its final publication
extends 182,272 tokens to 186,368 tokens: 4,096 added tokens and 31 delta stages.

The comparison uses publication code at `3fc3bfa` and the implementation that
retains verified snapshot proofs, performs one base lookup, and reuses at most
64 macro-object file handles during reconstruction. No persistent payload
cache is added. Handles close on both successful and failed reconstruction.

Conditions: Windows, Python 3.12.10, AMD Ryzen 9 9950X3D, 16 cores/32 threads,
scale 64, interval 0, stage limit 0, warm process header cache. Three serial
before/after pairs ran without the test suite running concurrently. Wall times
are medians; payload byte counts below are multiplied by 64, not actual I/O.

| Work for the final publication | `3fc3bfa` | Proof and handle reuse |
|---|---:|---:|
| Commit wall time, fixture | 108 ms | 93 ms |
| All 31 extensions, fixture | 2.217 s | 1.989 s |
| SHA-256 input, full-size equivalent | 5,982,063,040 B | 4,557,457,216 B |
| Object bytes read, full-size equivalent | 1,379,712,640 B | 1,379,712,640 B |
| Object bytes written, full-size equivalent | 104,254,208 B | 104,254,208 B |
| Extent file opens | 1,033 | 31 |
| Manifest written, actual fixture bytes | 24,927 B | 24,927 B |

The final commit ranged from 107 to 123 ms before and 92 to 94 ms with proof
and handle reuse. This is about 14% lower median CPU fixture time and 24%
fewer hashed bytes. These percentages do not establish Spark latency gains.

| Final-publication phase, inclusive median | `3fc3bfa` | Proof and handle reuse |
|---|---:|---:|
| Base reconstruction | 73.619 ms | 61.481 ms |
| Manifest lookup | 2.399 ms (two calls) | 1.773 ms (one call) |
| Header authentication | 0.050 ms | 0.050 ms |
| Sparse delta encoding | 16.632 ms | 9.179 ms |
| Manifest serialization | 0.685 ms | 0.600 ms |
| Manifest publication | 2.024 ms | 1.960 ms |

Phase times overlap: header authentication is included in reconstruction.
Warm header authentication read no objects. Reconstruction dominates; manifest
rewrite and warm header access do not explain the bulk of the growth.

Single-run checks: scale 16 reduced the final commit from 267 to 205 ms;
clearing authenticated headers before each publication at scale 64 reduced
it from 138 to 120 ms. Cold publication still read 3,392,978,560 equivalent
bytes. These checks do not clear the OS page cache.

With limit four at scale 64, the final entry has one stage. The complete
session's 31 extensions stage 7,696,832,128 equivalent object bytes, versus
3,590,341,632 without the limit: **2.14 times the staged writes**. The initial
snapshot is excluded from both totals. These are not SSD/NAND write counters.

## Remaining publication scaling

Status: **not implemented**: publication cost proportional only to added tokens.
Proof reuse removes a redundant base hash, not base reconstruction. Each
publication still reads retained state and hashes the whole composed result.

The delta header binds a SHA-256 of the canonical, layer-ordered snapshot.
The snapshot header changes with the block counts, and attention extensions
insert bytes between retained layer prefixes. A prior SHA-256 digest does not
provide the internal hash state needed to update that serialization from its
tail alone. Substituting the original root digest would break the chain.

Strict added-token scaling requires a separately specified incremental digest
format or authenticated persistent hash structure, including a distinct cache
identity and reader support. This implementation preserves existing identities
and formats and does not claim that requirement is met. The four-stage limit
bounds metadata and overwritten-state history, not retained-state byte work.

Hardware validation remains open: repeated capped and uncapped sessions,
independent bit-for-bit boundary checks, capture-slot fallback above 512 MiB,
publication CPU/memory pressure, commit and restore latency, cold starts,
and NVMe write amplification. No hardware qualification is inferred from CPU
timings or from the separate restore-cost hardware study.
