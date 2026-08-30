# GLM-5.3 Flash with seven-token DFlash2 TP4/DCP1 live validation

Status: **qualified** for persistent target-context store, coordinated engine
restart, all-rank manifest discovery, external restore, continued generation,
and DFlash2 speculation with seven draft tokens under the exact source
deployment recorded here. The shorter name `DFlash7` identifies that
seven-token deployment profile in commands and metrics.

## Artifact and serving contract

| Attribute | Qualified value |
|---|---|
| SparkCache source-tree SHA-256 | `6210f439c64e4079ed3304c9cc181174abb3e6045de740ba7b7c2546bcaf6ac2` |
| SparkCache profile | `glm53-flash-hybrid` |
| vLLM source revision | `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355` |
| vLLM lease contract | `sparkcache/runtime_patches/vllm-kv-block-lease-contract-da4d7be.json`, SHA-256 `2e3b17fd6a34f2dbb8e91a99b83dbf18629cf0e718f9f814236da4bbfc9ae3f1` |
| Target model | `local-inference-lab/GLM-5.3-Flash-NVFP4@520de24eabf507659eaef7c70f14fd584527facc` |
| DFlash model | `incoai/GLM-5.3-Flash-DFlash2@dc77ff1c99eeb2df044ee3d4f0094eb033fee410` |
| DFlash weights SHA-256 | `b33c03475ba7322cf398828f2d8d1be376df30dc05c6b40c28c8ea8da23e410b` |
| Parallelism | TP4, DCP1, PP1 over the four-node SparkRing RoCE ring |
| Serving geometry | 524,288 model limit, 8,192 scheduler tokens, 32 sequences, 12 GiB FP8 KV slab per rank |
| Speculation | DFlash2, seven draft tokens, BF16 draft weights, TP4 |
| Execution | B12X target attention/MoE/linear, FlashKDA prefill, async scheduling, chunked prefill, `FULL_AND_PIECEWISE` target graphs, DFlash FULL graphs |

## Provenance

The target checkpoint's published quantization configuration identifies
ModelOpt `0.39.0.dev290+gf9d9a71de.d20260407` as producer. Target routed
experts in layers 3–44 use NVFP4, while the embedded MTP expert layer 45 uses
MXFP8. Hugging Face attributes revision `520de24...` to uploader `lukealonso`.
The repository does not record the exact unquantized source revision, so this
record makes no claim about that missing lineage edge.

Every rank verified all 59 target-repository files against revision
`520de24...` with `hf cache verify --fail-on-missing-files`. Ranks 0, 2, and 3
also passed `--fail-on-extra-files`. Rank 1 contained 121 additional
`.cache/huggingface` metadata files; all 59 repository files matched, and the
extra metadata was not mounted as model content by vLLM.

Hugging Face attributes the public BF16 DFlash2 revision `dc77ff1...` to
uploader `zhijianliu`. Its configuration SHA-256 is
`c4aeac0101196a6e26705b34c45230bcd0c7c68ee2d2d1efdb242087f3712573`;
its weights are recorded in the serving-contract table. The checkpoint uses
the CC BY-NC-ND 4.0 license; SparkCache's Apache-2.0 repository license does not
replace the checkpoint license.
Every rank independently reproduced the published DFlash config and weight
SHA-256 values before launch.

The vLLM fork and pull-request lineage are documented in
[`deploy/glm53_flash/README.md`](deploy/glm53_flash/README.md). The runtime
image also pins `local-inference-lab/b12x@2fcf23a0ce269be27b2e03fece73d46e90e6aeea`.
The live NCCL 2.30.7 library had SHA-256
`ccd57342449c3f680befcb379329b935746e5299dc4de5f2516146e0411bd85f`.
This validation record has no source manifest that binds that exact binary
hash to an NCCL source commit, so NCCL source-commit provenance remains
unproven.

The four rank-local base and SparkCache image IDs were:

| Rank | Base image ID | SparkCache image ID |
|---:|---|---|
| 0 | `sha256:ddd13fb1ea8ca61aaf771715dc8c5a52dfe6860f0cc62c145d155916bf381fc9` | `sha256:56f051b1b1b6f9f858ea5d21b7933b64af81c22bee2c417a3f8b4466220e37e6` |
| 1 | `sha256:7fb81337ba088a6bf0bbce71b22a5881f812a21af9ac1d6deea9533a8e9eed92` | `sha256:8506935b369bd4f0d5d73495ded9a2fcb52bbe2f310ea093818e5d3d5366ae38` |
| 2 | `sha256:9bd97e3d77de969ee0788aaac31b2888fd4c6a3d893ac5fc544ca85363927935` | `sha256:b969a49ec091157c686a3bc3f52816b6aa910e495af0c92780a321ea5fbd5324` |
| 3 | `sha256:d592c83cc04106532adf7d8d410347062ac1b80fc1b6981deca414b5335efff4` | `sha256:c9f0be4dccfd8fdcec80a3edce1ad217604fa09afee0f14d13a2839fb97eed9f` |

Every image carried the same SparkCache source digest and verified the same
seven installed vLLM source files during its build.

## Cache geometry and cold store

Every rank registered 50 opaque hybrid-cache layers. The inventory covered the
target model's sparse-MLA/C4 pages and KDA/GDN recurrent checkpoints under
vLLM's 2,304-token resolved attention-page geometry. External DFlash state was
not persisted; its weight digest remained part of the cache identity.

An 8,192-token reusable span completed successfully and committed digest
`a6d78b05026f...` on every rank. Snapshot and durable-commit timings were:

| Rank | Snapshot | Durable commit |
|---:|---:|---:|
| 0 | 88.8 ms | 214.3 ms |
| 1 | 125.2 ms | 222.5 ms |
| 2 | 181.1 ms | 222.0 ms |
| 3 | 149.0 ms | 209.0 ms |

Three manifests and 77 immutable chunks occupied 284,880,209 bytes per rank.

## Coordinated restart and external restore

After all four containers were replaced, every rank reported
`checked=3 offered=3 rejected=0`. The first request after API startup occurred
before the scheduler had received a complete four-rank inventory checkpoint,
so it cleanly recomputed and refreshed the entry. The following identical
request formed quorum and restored exactly 8,192 external tokens:

| Rank | Restore time |
|---:|---:|
| 0 | 155.6 ms |
| 1 | 147.2 ms |
| 2 | 194.0 ms |
| 3 | 151.8 ms |

vLLM reported 16,457 external-prefix queries, 8,192 external-prefix hits, and
8,192 prompt tokens sourced from external KV transfer. The restored request
completed successfully in 1.509 seconds. An uncached semantic canary
finished with `stop` and final answer suffix `SPARKCACHE_GLM53_OK`. Its
suffix-only predicate did not prove that visible content contained no preceding
reasoning text or parser marker.

DFlash counters remained exact: 43 drafts produced 301 draft tokens
(`43 × 7`) and 112 accepted tokens. The service recorded zero preemptions.
Every rank retained 24 RTS `VLLM::Worker` RDMA QPs, zero container restarts,
zero OOMs, and zero fatal-log matches after the qualified restart.

## Reproduction commands and request identities

The image build used `deploy/glm53_flash/build_image.py` independently on each
rank with that rank's immutable base image ID and the SparkCache source digest
in the serving-contract table. SparkRing commit
[`6e9e3acef62886a71531310673463972944b2b84`](https://github.com/FujitsuPolycom/sparkring/tree/6e9e3acef62886a71531310673463972944b2b84)
publishes the sanitized TP4 launcher, runtime pins, cached and non-cached
quickstarts, and qualification receipts corresponding to the model mounts,
four-rank topology, and connector JSON.

The cold-store and post-restart replay use prompt SHA-256
`a8569c46a6cbf22bae4736c897023f7c552952440bf75e1ef6ebabe594f513cf`:

```bash
python deploy/glm53_flash/qualification_request.py \
  --endpoint http://<rank-0-host>:8015 \
  --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 \
  --kind persistent \
  --output <receipt-path>.json
```

The committed cold-store receipt is
[`evidence/glm53-flash-dflash7-bf16/cold.json`](evidence/glm53-flash-dflash7-bf16/cold.json).
Run the command before the coordinated four-container restart and again after
every rank logs `manifest discovery checked=3 offered=3 rejected=0`. Because
worker inventories reach the scheduler asynchronously, repeat the command once
if the first post-startup request records a quorum miss. The committed prime
and restore receipts are
[`post-restart-prime.json`](evidence/glm53-flash-dflash7-bf16/post-restart-prime.json)
and
[`post-restart-restore.json`](evidence/glm53-flash-dflash7-bf16/post-restart-restore.json).
Qualification requires the restore receipt to finish with `stop` or `length`,
vLLM to report 8,192 external-hit tokens, and every worker log to report an
8,192-token restore.

The continued-generation canary uses a separate short prompt:

```bash
python deploy/glm53_flash/qualification_request.py \
  --endpoint http://<rank-0-host>:8015 \
  --model glm-5.3-flash-nvfp4-dflash7-bf16-tp4 \
  --kind semantic \
  --output <semantic-receipt-path>.json
```

Its receipt,
[`post-restore-semantic.json`](evidence/glm53-flash-dflash7-bf16/post-restore-semantic.json),
records `finish_reason: stop` and `semantic_match: true` under the historical
suffix-only predicate. That receipt proves continued generation and the marker
suffix, not exact visible output. Exact-output semantic qualification requires
a new receipt from the equality validator in `qualification_request.py`; the
visible `message.content` must equal `SPARKCACHE_GLM53_OK` byte for byte.

## Limitations

This evidence does not qualify another SparkCache source tree, vLLM source
contract, target or draft checkpoint, parallel topology, scheduler budget,
cache geometry, SparkCache direct CUDA restore, streaming snapshots, or MTP profile.
It does not establish throughput neutrality or restore performance for spans
larger than 8,192 tokens. Full reasoning-trace equality is not a semantic
oracle for this GLM runtime. The historical canary establishes successful
continued generation and a final-answer suffix only. It does not establish
exact visible output.
