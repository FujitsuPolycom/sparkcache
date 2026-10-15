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
