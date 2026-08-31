# DeepSeek-V4-Flash-0731 TP4/DCP1 runbook

This runbook converts a cache-disabled DeepSeek four-rank inspection that
passes the pinned profile and homogeneous-image preflight into a bounded
SparkCache deployment. It is intended to work on
any four-host DGX Spark ring whose source inspection satisfies the pinned
profile; it does not embed site-specific host names, addresses, model paths,
or transport ports.

## Qualification boundary

- Model: `deepseek-ai/DeepSeek-V4-Flash-0731`
- Runtime: the fixed-MTP4 SparkRing serving-image lineage `R7` at vLLM revision
  `e2666d9a6`, named by `tp4_profile.json`
- Parallelism: TP4, PP1, DCP1
- Speculation: DSpark with five draft tokens per step (`K5`) and the b12x MoE
  backend
- KV: 32 GiB per rank, `fp8_ds_mla`, block size 256
- SparkCache: Python hybrid-memory-allocator (HMA) block-page codec, restore,
  and bounded LRU enabled
- Capacity: 200 GiB high and 180 GiB low watermark per rank-local root
- Not qualified: DCP2/DCP4, shared writers, streaming snapshots, SparkCache
  CUDA restore, expert parallelism, or an unpinned checkpoint/runtime

The launcher fails closed when any of these inputs drift. Do not weaken a
check to make an unfamiliar image start; derive and test a profile for that
exact runtime.
The published base image inherits GLM CUDA-graph exact-state attestation
environment names for capture sizes 1 through 40 (`Q1` through `Q40`); the
DeepSeek transformation removes them and records them in
`SPARKRING_EXPLICITLY_UNSET`. Actual LMCache configuration is still rejected.

## Keep every stack in its own namespace

Assign each simultaneously installed model stack a unique set of:

| Resource | Example for this profile |
|---|---|
| container names | `deepseek0731-sc-r0` through `-r3` |
| API port | `8100` on rank 0 |
| rendezvous/master port | `29600` |
| collective/transport ports | a disjoint range in the source site profile |
| rank-local cache host directory | `/var/tmp/sparkcache-deepseek0731-tp4` |
| rank-local JIT host directory | `/var/tmp/jit-deepseek0731-tp4` |
| in-container SparkCache root | `/cache/sparkcache-deepseek0731-tp4-dcp1` |

The API and master port must differ. SparkCache opens no network ports, but
vLLM rendezvous, collectives, and any custom transport do. The first
qualification run should use the source runtime's stock NCCL transport.

## 1. Freeze the checkpoint identity

Build a complete content manifest on one rank. The output must be outside the
model tree and is created exclusively:

```bash
python3 -m deploy.deepseek_v4.checkpoint_manifest build \
  --root /var/tmp/deepseek-v4-flash-0731 \
  --repository deepseek-ai/DeepSeek-V4-Flash-0731 \
  --revision 7872f01b1d1fe23eabc4c98b48bffcef5a386062 \
  --output /var/tmp/deepseek0731-checkpoint-manifest.json \
  --workers 4
```

Copy that small manifest to the other ranks and verify each local model tree:

```bash
python3 -m deploy.deepseek_v4.checkpoint_manifest verify \
  --root /var/tmp/deepseek-v4-flash-0731 \
  --manifest /var/tmp/deepseek0731-checkpoint-manifest.json \
  --workers 4
```

All four commands must print the same `checkpoint_sha256`, file count, and
byte count. Top-level `.cache` and `.git` metadata are excluded; every actual
model/tokenizer/config file is included, and symlinks are rejected.

## 2. Preserve the cache-disabled source inspections

Save one `docker inspect` document per rank before modifying or stopping the
accepted DeepSeek containers:

```bash
docker inspect <accepted-rank-container> \
  > /var/tmp/deepseek0731-rank-inspect.json
```

The four inspections are the launch source of truth. Preserve them with the
checkpoint manifest and record the exact image ID. First prove this homogeneous
four-rank image can serve with SparkCache disabled. Evidence collected from
ranks with non-identical image IDs does not satisfy this gate.

## 3. Stage and verify the overlays

Stage the same clean SparkCache commit on every rank. Do not copy a dirty
working tree. The tree digest must equal `sparkcache.source_sha256` in
`tp4_profile.json`.

From the exact source image on each rank, create the scheduler and vLLM config
overlays:

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

The preparation chain is exact and atomic:

1. stock scheduler `1ea341...` plus patch 011 -> `d4ebec...`;
2. the stock-lineage HMA failure patch 031 -> `2f34aa...`; and
3. vLLM config `fbc581...` plus patch 020 -> `71c4f9...`.

Any preimage or postimage mismatch is a stop condition.

## 4. Create all four candidates before cutover

Before creating containers, validate the four inspections together. This
catches duplicate physical ranks, mixed image IDs, inconsistent master
addresses, and inconsistent collective port assignments:

```bash
python3 -m deploy.deepseek_v4.tp4_cluster_preflight \
  --inspect /var/tmp/deepseek0731-r0-inspect.json \
  --inspect /var/tmp/deepseek0731-r1-inspect.json \
  --inspect /var/tmp/deepseek0731-r2-inspect.json \
  --inspect /var/tmp/deepseek0731-r3-inspect.json \
  --checkpoint-sha256 <checkpoint_sha256> \
  --api-port 8100 \
  --master-port 29600 \
  > /var/tmp/deepseek0731-cluster-plan.json
```

Create separate cache and JIT directories on each rank. They must be disjoint:

```bash
mkdir -p /var/tmp/sparkcache-deepseek0731-tp4 \
  /var/tmp/jit-deepseek0731-tp4
```

Then run the guarded launcher once per rank, using that rank's inspection:

```bash
python3 -m deploy.deepseek_v4.tp4_launch \
  --inspect /var/tmp/deepseek0731-rank-inspect.json \
  --image <exact-image-id-from-that-inspection> \
  --name deepseek0731-sc-r<RANK> \
  --checkpoint-sha256 <checkpoint_sha256> \
  --cache-host-path /var/tmp/sparkcache-deepseek0731-tp4 \
  --jit-host-path /var/tmp/jit-deepseek0731-tp4 \
  --sparkcache-source-host-path /var/tmp/sparkcache-src/sparkcache \
  --scheduler-overlay-host-path \
    /var/tmp/sparkcache-deepseek0731-overlays/scheduler.py \
  --vllm-config-overlay-host-path \
    /var/tmp/sparkcache-deepseek0731-overlays/vllm.py \
  --vllm-overlay-receipt-host-path \
    /var/tmp/sparkcache-deepseek0731-overlays/receipt.json \
  --api-port 8100 \
  --master-port 29600 \
  --create-only
```

Inspect all four created containers. Require the same image, checkpoint,
profile, command, master address/port, model bind, and read-only overlay
receipts; require physical node ranks 0, 1, 2, and 3 exactly once. Only rank 0
may publish the API port.

## 5. Coordinated cutover and persistence gate

Archive the rollback stack's inspections, logs, and single-writer receipts.
Stop or start all four serving ranks as one coordinated operation; a partial
collective is not a healthy fallback. Start the four created candidates close
together and wait for rank 0 `/health` plus `/v1/models`.

Run the existing semantic gate on a miss, wait until all four physical ranks
publish the same digest for all five HMA groups, and save logs:

```bash
python3 -m deploy.deepseek_v4.semantic_gate miss \
  --endpoint http://<rank0-management-address>:8100 \
  --reference /var/tmp/deepseek0731-reference.json
```

Restart all four containers without clearing the rank-local cache roots. Run
the hit phase:

```bash
python3 -m deploy.deepseek_v4.semantic_gate hit \
  --endpoint http://<rank0-management-address>:8100 \
  --reference /var/tmp/deepseek0731-reference.json
```

A response with an empty assistant body or `finish_reason` equal to `length`
prints a structured `INCONCLUSIVE` result and exits with status 2. An
inconclusive result does not satisfy the persistence gate.

Qualification requires all of the following, not merely matching output:

- all four ranks rediscover the same manifest digest after restart;
- the scheduler observes a current-generation four-rank quorum;
- every rank logs an external-cache restore for the request;
- miss and hit semantic output are byte-identical; and
- the fresh post-restore canary is correct.

The hit gate sends a short canary before the long request. That first worker
round trip transports each restarted rank's discovery report to the scheduler;
without it, the scheduler must conservatively treat the first long request as
a quorum miss even though every worker has already discovered its manifest.
The canary is an exact echo sentinel rather than arithmetic: this DSpark build
can deterministically answer the same trivial arithmetic prompt differently
under async and synchronous scheduling, which makes arithmetic unsuitable as
a cache-corruption oracle.

## 6. Large-context and failure gates

Increase aligned prompt sizes in stages (32K, 64K, 128K, then the largest
safe size). Use deterministic content and a distant-fact answer so the
completion proves that early and late prompt regions survived. Record prompt
tokens, encoded bytes per rank, miss latency, restore latency, manifest
digest, and the five HMA group page counts.

The semantic gate scales deterministically with `--records`. Use a separate
reference per size; the hit phase reads and verifies the recorded size:

```bash
python3 -m deploy.deepseek_v4.semantic_gate miss \
  --endpoint http://<rank0-management-address>:8100 \
  --model deepseek-v4-flash-0731 \
  --records 4096 \
  --reference /var/tmp/deepseek0731-large-4096.json
```

Before calling the deployment durable, also test:

1. Capacity pressure in a separate empty root with `capacity_gate.py`.
2. A copied cache root with one chunk corrupted: restore must fail closed and
   recompute; never corrupt the only serving copy.
3. One-rank restart generation change: stale quorum reports must be withdrawn
   until the restarted rank reports the current generation.
4. A full four-rank power-cycle simulation, preserving NVMe cache roots.

## Rollback

Do not delete the rollback model containers, inspections, receipts, or cache
roots during qualification. On any failure, stop all four DeepSeek candidates
as a unit and restart all four rollback ranks from their recorded state. A
rollback is complete only after the rollback model passes health, model-list,
short semantic, and collective-rank checks.

## Known failure signatures

| Symptom | Meaning | Action |
|---|---|---|
| overlay preimage mismatch | image/runtime drift | stop and derive a pinned chain for the exact source |
| only some ranks remain alive | broken collective launch | capture logs, stop all ranks, relaunch together |
| correct answer without four restore logs | semantic coincidence or native prefix hit | do not qualify; prove external restore |
| quorum vanishes after one rank restarts | expected generation reset | wait for all four current-generation reports |
| DCP2/DCP4 requested | unsupported opaque-page layout | keep DCP1; design and test an explicit sharding codec |
| high watermark briefly exceeded | post-commit reclamation, not allocation reservation | inspect maintenance result and free space |
| corrupted chunk | integrity gate working | require recompute and no restored partial state |
