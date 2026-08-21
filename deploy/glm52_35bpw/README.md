# GLM-5.2 EXL3 3.5-bpw TP4/DCP4 deployment

This bundle adds standalone SparkCache to the fixed-MTP4 GLM-5.2 3.5-bpw
serving recipe identified by SparkRing as `R7` and documented by the public
[3.5-bpw quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM52_35BPW_QUICKSTART.md).
Setup and recovery signatures are collected in
[`TROUBLESHOOTING.md`](TROUBLESHOOTING.md).
It is specific to model revision
`9ab9579774cc432df91567a36f6e9e863e0d4c9f` and validates the inspected
engine before changing it: TP4/DCP4 all-gather/reduce-scatter (`ag_rs`),
interleave one, fixed MTP4,
dynamic `nvfp4_ds_mla` with FP8 RoPE, block size 64, 262,144-token model
limit, 9,250,000,000 GPU KV bytes per rank, eight sequences, CUDA graph capture
sizes 1 through 40 (`Q1` through `Q40`),
and served name `glm-5.2-exl3-r7-3.5bpw`.
It requires the public target-only exact-state receipt covering `Q1` through
`Q40` (`exact-Q40`) named by that quickstart. Operator variants specialized to
capture sizes 35 or 40 (`Q35`/`Q40`) are unsupported and rejected.

The transformation makes the accepted runtime's implicit block size 64
explicit and preserves its R7 speculation, Q40 compilation,
the Spark inter-rank collective layer (SIRCL), transport, rank, JIT, and
runtime settings. It removes an LMCache
connector and LMCache environment values, inserts one SparkCache connector,
and optionally changes only the API and rendezvous ports. The accepted image
and R7 entrypoint are retained, so its runtime verification, image-bound Q40
state, and `SPARKRING_EXPLICITLY_UNSET` handling still run.

## Prepare the SparkCache overlays

The overlay deployment keeps the exact image recorded by the qualified source
inspection. Stage three additional read-only inputs at rank-local host paths:

- this repository's `sparkcache/` directory;
- `scheduler.py` after exact application of the stock GLM-image patch
  `011-sparkcache-glm52-async-rollback.patch`; and
- `vllm.py` after exact application of
  `020-sparkcache-vmm-exemption.patch`.

The source files must come from that image's e2666d9a6 vLLM installation.
The published patch preimages reject any other source. Keeping these changes
as mounts preserves the image identity used by exact-Q40 while adding the
connector and fail-closed recompute behavior.

Prepare the two files on each rank from the exact inspected image (the output
directory must not already exist):

```bash
docker run --rm --entrypoint python3 \
  --volume /var/tmp/sparkcache-src:/src:ro \
  --volume /var/tmp:/host-output \
  <accepted-r7-image-id-or-tag> \
  /src/deploy/glm52_35bpw/prepare_vllm_overlays.py \
  --vllm-root /opt/venv/lib/python3.12/site-packages \
  --repository /src \
  --output /host-output/sparkcache-r7
```

The command produces `scheduler.py`, `vllm.py`, and `receipt.json` after
checking the exact pre- and postimage hashes. The serving command below uses
the generated Python files as read-only overlays. Before Docker creation, the
launcher rehashes both overlays and the deployable SparkCache source tree
against that receipt.

`Containerfile` is an optional artifact builder for those patched files or a
fully rebuilt image:

```bash
docker build \
  --file deploy/glm52_35bpw/Containerfile \
  --build-arg BASE_IMAGE=<accepted-r7-image-id-or-tag> \
  --tag sparkcache/glm52-exl3-r7-3.5bpw:local .
```

Do not serve that derived image under the accepted Q40 receipt without
regenerating the image-bound Q40 layer through the public quickstart. The
overlay launch below uses the accepted image instead.

## Launch one rank

Save `docker inspect` for the corresponding accepted R7 rank, then create a
replacement container. `--create-only` makes the Docker configuration
inspectable before it is started.

```bash
python -m deploy.glm52_35bpw.launch \
  --inspect glm52-r0-inspect.json \
  --image <exact-sha256-image-id-from-inspect> \
  --name glm52-r7-sparkcache-r0 \
  --checkpoint-sha256 \
    9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd \
  --cache-host-path /var/lib/sparkcache/glm52-r7-r0 \
  --sparkcache-source-host-path /var/tmp/sparkcache-src/sparkcache \
  --scheduler-overlay-host-path /var/tmp/sparkcache-r7/scheduler.py \
  --vllm-config-overlay-host-path /var/tmp/sparkcache-r7/vllm.py \
  --vllm-overlay-receipt-host-path /var/tmp/sparkcache-r7/receipt.json \
  --cache-root /cache/sparkcache-glm52-r7 \
  --api-port 8000 \
  --master-port 29500 \
  --create-only
```

Repeat from each rank's own inspection with a rank-unique container name.
`--image` must equal that inspection's immutable `Image` field; a tag or a
different derived image is rejected so the image-bound Q40 state cannot drift.
`--cache-host-path` is required and is bound read/write at `--cache-root`;
the accepted R7 inspection mounts only `/cache/jit` and
`/cache/exl3-online`, not a parent store for SparkCache. Every bind from the
source inspection is replayed with its inspected read/write mode, including
the model plus Q40/SIRCL files under `/opt/spark-vllm` and `/opt/venv`. Their
host sources must continue to exist at the inspected paths on each rank.
SparkCache source and the two compatibility files are mounted read-only at
their runtime locations; the cache directory is the only additional
read/write bind.

For the exact verified quickstart checkpoint, the documented namespace value
is the pinned model-index SHA-256
`9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd`.
This low-ceremony default is valid only for the exact quickstart checkpoint
after `scripts/download_exl3_r7.py verify` succeeds on that rank; the pinned
revision and LFS metadata are what bind its shard payloads. For any repack or
different artifact tree, pass `--checkpoint-sha256` from a complete immutable
artifact manifest so the cache namespace cannot be reused.
The cache profile alias `glm52-exl3-r7-3.5bpw` resolves to the defined
`glm52-nvfp4` layout object, so the alias does not change cache identity.

The default rank-local policy is:

- root `/cache/sparkcache-glm52-r7`;
- 200 GiB high and 180 GiB low watermarks;
- TTL zero;
- store and restore enabled;
- fail-closed load policy `recompute`;
- colocated-target MTP state, with no separate draft digest;
- scheduler probe `none`; and
- streaming snapshots and native direct restore disabled.

The complete `--kv-transfer-config` is the enable switch. The obsolete
`SPARK_CONTEXT_CACHE_ENABLE` image variable is removed and explicitly unset
before the inherited R7 entrypoint starts vLLM.

The accepted inspection has neither `--enable-prefix-caching` nor
`--disable-prefix-caching`; R7 supplies its native prefix-cache default. The
transformer preserves an absent flag or one explicit enable and rejects an
explicit disable.

## Native feature switches

Streaming publication and native direct restore use different libraries and
can be enabled independently. Each enable requires its library's container
path and SHA-256:

```bash
# Add to the launch command for streaming publication.
--streaming-snapshots \
--streaming-native-library \
  /opt/sparkcache-src/sparkcache/native/build/libspark_cache_snapshot.so \
--streaming-native-library-sha256 <64-lowercase-hex>

# Add independently for native direct restore.
--native-restore \
--native-restore-library \
  /opt/sparkcache-src/sparkcache/native/build/libspark_cache_placement.so \
--native-restore-library-sha256 <64-lowercase-hex>
```

Keep both switches off for the qualification baseline. Before enabling
either switch, build the corresponding `sparkcache/native` target for SM121
in the R7 toolchain. The read-only SparkCache source bind then carries the
resulting library into the container at the paths above.

The 200/180 GiB capacity policy remains active when streaming is enabled.
Use `--no-streaming-snapshots` and `--no-native-restore` for explicit
disabled settings.

## Concurrent deployment instances

Every model stack needs its own container names, rank-local cache host
directories, SparkCache root, API port, rendezvous/master port, and upstream
JIT directory. SparkRing collective and transport port ranges must also be
unique in the source site/profile. This transformer preserves those upstream
transport and JIT assignments and does not synthesize replacements. It rejects an API
port equal to the rendezvous port.

## Four-rank persistence gate

After all four ranks are healthy, create one deterministic reference on a
cache miss and wait for every rank to publish its manifest:

```bash
python -m deploy.glm52_35bpw.semantic_gate miss \
  --endpoint http://<rank0-management-address>:8000 \
  --reference glm52-sparkcache-reference.json \
  --records 12000
```

The exact-Q40 runtime publishes one rank-local receipt under the shared JIT
root and refuses to replace an existing receipt. Before every coordinated
restart, stop all four model containers, record each receipt's SHA-256, and
move these host files to unique backup names:

```text
/var/lib/sparkring/jit-cache/q40-exact-state-serving-v1-rank<RANK>.json
```

Do not remove the SparkCache roots. Start all four containers after the Q40
receipt paths are vacant, require fresh Q40 attestation on every rank, confirm
`/health`, then run the hit phase:

```bash
python -m deploy.glm52_35bpw.semantic_gate hit \
  --endpoint http://<rank0-management-address>:8000 \
  --reference glm52-sparkcache-reference.json \
  --records 12000
```

The hit phase sends a short request that publishes every physical rank's
manifest inventory (a quorum prime) so the scheduler receives each
worker's discovered-manifest stats before the long request. It requires the
exact long answer
`SPARKCACHE_OK:9540` and a post-restore `42` canary. Container logs must also
show the same digest offered by all four physical ranks and an external-cache
restore rather than a fresh store.

## Offline validation

```bash
python -m pytest sparkcache/test_glm52_35bpw_deploy.py \
  deploy/deepseek_v4/test_deploy.py -q
python -m pytest sparkcache -q
```

The fixed-MTP4 GLM-5.2 EXL3 3.5-bpw SparkCache composition (SparkRing recipe
identifier `R7`) is qualified at TP4/DCP4. The bundle is also GPU-free tested.
A rebuilt image must pass the public quickstart's four-rank qualification
checks against its immutable image ID.
