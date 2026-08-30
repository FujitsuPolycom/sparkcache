# DeepSeek-V4 deployment tooling

This directory contains fail-closed deployment and validation tools for
DeepSeek-V4 SparkCache profiles.

## TP4/DCP1 qualified profile

`tp4_profile.json` defines the qualified
`deepseek-ai/DeepSeek-V4-Flash-0731` profile:

- TP4, PP1, DCP1;
- DSpark speculation with five draft tokens per step (`K5`) and the b12x MoE
  backend;
- `fp8_ds_mla`, 32 GiB KV per rank, and hybrid-memory-allocator (HMA) block
  size 256;
- 524,288-token request limit and 32 sequences;
- 200 GiB high / 180 GiB low rank-local NVMe watermarks; and
- Python verified restore with streaming and SparkCache CUDA placement disabled.

The launcher rejects DCP2/DCP4 because neither DSpark nor the opaque five-group
HMA page format defines safe DCP ownership. See `DCP_SUPPORT.md`.

Use `TP4_RUNBOOK.md` for checkpoint manifests, cluster preflight, overlay
generation, create-only inspection, cutover, and restart/restore gates. The
qualified measurements and exact runtime identifiers are in
`../../DEEPSEEK_V4_TP4_LIVE_VALIDATION.md`.

## Tool interfaces

| Tool | Purpose |
|---|---|
| `checkpoint_manifest.py` | hash every checkpoint artifact into one immutable identity; top-level download metadata is excluded |
| `tp4_profile.py` | validate and transform one cache-off Docker inspection |
| `tp4_cluster_preflight.py` | validate ranks 0–3, homogeneous image identity, rendezvous, and collective ports before Docker mutation |
| `tp4_prepare_vllm_overlays.py` | apply exact scheduler/config patch chains and produce a source-bound receipt |
| `tp4_launch.py` | create one guarded rank with read-only model/source/overlay binds and disjoint writable cache/JIT roots |
| `semantic_gate.py` | create or verify deterministic semantic references with a short request that publishes every physical rank's manifest inventory (a quorum prime) and an immediate echo sentinel |
| `capacity_gate.py` | exercise physical-byte accounting, high/low eviction, open-transaction exclusion, and orphan collection in a disposable root |
| `corruption_gate.py` | copy one serving entry, corrupt one copied chunk, require a fail-closed lookup, and invalidate only the copy |

Every output path used by checkpoint, overlay, and corruption tools is created
exclusively. Existing paths are not overwritten.

## Exact vLLM overlays

The qualified source uses the SparkRing serving-image lineage identified as
`R7`, with vLLM revision `e2666d9a6`. Its overlay chain is:

1. stock scheduler `1ea341...` plus patch 011 -> `d4ebec...`;
2. stock-lineage HMA failure patch 031 -> `2f34aa...`; and
3. vLLM config `fbc581...` plus patch 020 -> `71c4f9...`.

Overlay generation verifies every preimage and postimage. It also records the
deployable `sparkcache/` tree digest; `tp4_launch.py` rehashes the source,
receipt, and overlay files before invoking Docker.

Run the builder as a module from the repository mount so sibling deployment
packages are importable:

```bash
docker run --rm --entrypoint python3 --workdir /src \
  --volume /var/tmp/sparkcache-src:/src:ro \
  --volume /var/tmp:/host-output \
  <exact-image-id> \
  -m deploy.deepseek_v4.tp4_prepare_vllm_overlays \
  --vllm-root /opt/venv/lib/python3.12/site-packages \
  --repository /src \
  --output /host-output/sparkcache-deepseek0731-overlays
```

## Capacity semantics

The high watermark triggers post-commit and periodic reclamation; it is not a
preallocation limit. Maintenance removes invalid/expired manifests, applies
manifest-recency LRU pressure to the low watermark, and then collects chunks
not referenced by any surviving manifest. A failed pass can leave excess
bytes but does not block inference.

One publisher per rank-local root is qualified. Multiple publisher processes
can transiently exceed the configured high watermark until maintenance
reconciles physical use.

## Validation commands

```bash
python -m pytest deploy/deepseek_v4 -q
python -m pytest sparkcache -q
python -m ruff check deploy/deepseek_v4 sparkcache
```

The default tests are GPU-free. Live qualification additionally requires
four-rank logs and vLLM external-prefix-cache metrics; matching model text
alone is not evidence of an external restore.
