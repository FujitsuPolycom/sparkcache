# GLM-5.3 SparkCache CUDA full-snapshot validation

Date: 2026-08-30

## Status

Full `snapshot-v1` opaque-page restore is **qualified** at 131,072 tokens and C1
for the exact GLM-5.3 Flash DFlash7 TP4/DCP1 artifacts identified below. The
qualification covers verified all-rank placement, multi-group recovery, and
continued generation.

The flat `sparkcache-page-snapshot-manifest/v2` artifact is also qualified for
one exact C1 case: 813,068,464 encoded bytes in 13 authenticated objects,
sequential object read/hash of 1.35–1.50 seconds, an all-rank SparkCache CUDA
restore of 1.55–1.70 seconds, and the exact expected codeword before and after
restart. This result is bound to local image
`sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e`,
SparkCache source `a1511d26a1fe2b17b24561bc52e376bf7f54b06a`, and source tree
`4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb`. SparkCache `229d7d6` is the
implementation origin of the flat-v2 schema and header-inclusive statistics
fix, not the complete qualified runtime identity. No public OCI digest is
qualified by this record.

Tail-only opaque-page deltas, 16-member host-base read coalescing, and
multi-root concurrent restore are **research-only**. In the exact DFlash7 C2
case, both restored page-delta responses failed the codeword oracle while the
same requests succeeded through recomputation. Completion, timing, and
structural verification do not override that semantic result.

Earlier C2/C8/C16 receipts used a suffix-only predicate. They remain useful
performance diagnostics, but they do not qualify semantic concurrency or
shared GPU-prefix attachment. Arbitrary earlier opaque-page aliases remain
**unsupported**.

## Qualified runtime identity

| Attribute | Value |
|---|---|
| SparkCache source revision | `2b86fb9d02fa3595cca5caa864b81aedce44b8bb` |
| SparkCache source-tree SHA-256 | `b3e84d220e215bdad99455a7eefb431b9aea248e0edb6ff417319c420433f55a` |
| vLLM source revision | `local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355` |
| Serving topology | GLM-5.3 Flash, TP4/DCP1, one rank on each of `spark-r0` through `spark-r3` |
| Scheduler capacity | `--max-num-seqs 32` |
| Restore concurrency | Two host restore workers and two SparkCache CUDA placement lanes per rank |
| SparkCache CUDA staging | Two 256 MiB mapped-host arenas per rank |
| Persistent prefix | 131,072 tokens and 813,068,464 encoded bytes per rank |
| Runtime receipt | `evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-runtime.json` |

The runtime receipt records one immutable image ID per rank, CUDA placement library
SHA-256 `683cb9e0420da9c68e3263093077fdbcaa400913ff0fb1d18639771213220605`,
scheduler SHA-256
`4f8793c4ac4bf356a89c829b6e75b189e6bc4a74c97135208952d0bad1678f15`,
and KV-cache-manager SHA-256
`ee03dc9ce2b720c0be6e9f572d23580ba96eff68fe3406250557e83071654af0`.

The direct page-placement implementation is identified by these commits:

| Responsibility | Revision |
|---|---|
| Restore phase timing | `175f9401984a03744d7fe1a985d7c2ef6035f949` |
| SparkCache CUDA page placement | `71f367be07788d611698a251fe866d678b0034ae` |
| Multi-slab restore and exact-prefix discovery | `8e7f5fc62fd4fffdd661aca9ea634cf130c45d1a` |
| Direct pipelined slab restore | `94c44930a13df5c668d777e0270e7d8203069d7c` |
| Authenticated span-table bound | `9dbf73c0caab89b24346567e2769752ac746e114` |

The 131,072-token single-request measurement used source-tree SHA-256
`368cc18dbccc262a1f2a1f1eef5aced90690031abd1f2fedf3d192e60a67012b`
and parent/runtime image
`sha256:7c007cf673c35f5818da7fea8faa343304baed00f489efdcbd027d6616b8a290`.
The historical shared-prefix diagnostics used source revision `2b86fb9d...`
and source-tree SHA-256 `b3e84d...` identified in the table above.

### Qualified flat-v2 artifact

| Attribute | Value |
|---|---|
| SparkCache source revision | `a1511d26a1fe2b17b24561bc52e376bf7f54b06a` |
| SparkCache source tree | `4d5b8eb8c5c13793ee7a1e67b2b34bd38fcf4ddb` |
| SparkCache source-tree SHA-256 | `6651f2823c816fac93779cbca54a8f19c0ed262830953149f3a87d189d1f833b` |
| Flat-v2/header-fix implementation origin | `229d7d6158261e9510ab99d7e82d532abb9ade01` |
| Local image ID | `sha256:35b58a7bf414059c65b8f74e4e4b17ee6a81b7008e1bffbc9bd298b5e08c739e` |
| Parent image ID | `sha256:cc2c0e2f812f4b78d5b91f863aaf46fd8e8e505844245aa50911af1fb8e061c0` |
| Publication identity | `snapshot-v1` |
| Root schema | `sparkcache-page-snapshot-manifest/v2` |
| Persistent prefix | 131,072 tokens and 813,068,464 encoded bytes per rank |
| Physical objects | 13 authenticated objects per rank, each at most 64 MiB |
| Serving topology | GLM-5.3 Flash DFlash7, TP4/DCP1, one rank per DGX Spark |
| Qualified workload | C1 publication, restart, all-rank restore, and exact codeword |

The measured all-rank SparkCache CUDA restore was 1.55–1.70 seconds, of which
sequential macro-object read/hash consumed 1.35–1.50 seconds. The codeword
oracle matched exactly before restart and after restoration. The result
qualifies this artifact and workload only; it does not qualify a registry
artifact, tail-delta roots, or concurrent restored roots.

The sequential read/hash phase is a measured performance limitation of the
flat-v2 CUDA path. The identified legacy 512-object path used bounded parallel
reads within each arena slab. Existing legacy roots remain readable and follow
that restore branch, but connector publication writes flat-v2 roots and exposes
no operator setting for legacy publication.

### Bounded parallel-read implementation after the qualified artifact

SparkCache PR43 head `b553c487bc273ad3efefa4052dc06376543dcd9d` replaces the
sequential remaining-object loop with bounded two-arena reads. It authenticates the
first object before parsing the snapshot header, then reads and authenticates
at most two remaining objects concurrently into the two placement-owned arenas.
Complete-snapshot SHA-256 accumulation and CUDA submission remain in manifest
order and begin only after every read in the pair succeeds. The configured I/O
worker count may reduce concurrency to one; values above two remain capped by
arena ownership.

This scheduling is **implemented** and GPU-free tested. It is not present in
image `35b58a7…` and has no live semantic or performance qualification. The
safe retained qualification remains the sequential 1.55–1.70-second C1 restore
described above.

## Implemented restore path

SparkCache CUDA restore reads immutable `.spcc` objects directly into alternating
mapped-host arenas with `pread`, hashes every complete file in place, validates
its authenticated extent table, and submits only validated spans to the CUDA
page-placement kernel. Read work and CUDA submission overlap across slabs.

This path avoids Python `ContextChunk` reconstruction and an 813 MiB
intermediate join/copy. The adapter accepts at most 4,096 authenticated spans,
matching the validated SparkCache CUDA ABI. These changes do not alter `CacheIdentity`,
digest values, 256-token logical geometry, or the on-disk exact-manifest and
chunk formats.

## Practical restore gains

### One 131,072-token prefix

Each rank restored 813,068,464 bytes through four slabs:

| Rank | Cache service | Read and hash | CUDA placement submit | CUDA finish |
|---:|---:|---:|---:|---:|
| 0 | 141.9 ms | 77.9 ms | 25.4 ms | 3.4 ms |
| 1 | 131.3 ms | 84.6 ms | 13.4 ms | 3.4 ms |
| 2 | 139.0 ms | 92.7 ms | 14.1 ms | 3.4 ms |
| 3 | 250.1 ms | 140.7 ms | 20.1 ms | 3.4 ms |

The Python reconstruction pipeline measured 1.29--1.46 seconds per rank for
the same stored prefix. SparkCache CUDA restore therefore reduced the slowest-rank
time to 250.1 ms. End-to-end client latency was 0.907 seconds, including
scheduler work, live-token execution, and DFlash generation. A separate
historical canary found the expected marker suffix, and HTTP health remained
200.

### Eight concurrent 16K prefixes: diagnostic timing

The Python/Torch placement path produced 1.54--1.57 second submission spikes;
eight clients completed in 9.45--10.64 seconds. SparkCache CUDA placement submitted in
6--15 ms, and two restore lanes completed eight clients in approximately
1.2--2.1 seconds. This diagnostic isolates page placement as the dominant
serialized cost in that workload. The timing is diagnostic and does not
qualify semantic concurrency.

## Historical concurrency diagnostics

The independent-restore baseline issued a complete external restore for every
request: two per rank for C2, eight for C8, and sixteen for C16.

| Cohort | Client min | Client p50 | Client max |
|---:|---:|---:|---:|
| C2 | 1.282 s | 1.282 s | 1.377 s |
| C8 | 0.947 s | 2.017 s | 3.129 s |
| C16 | 1.063 s | 3.363 s | 5.335 s |

Bounded shared-prefix leases changed the observed rank-local work from sixteen complete
813 MiB restores to one. The standard chat C16 measurement completed every
request at 2.980 seconds p50 and 5.064 seconds maximum. This measurement
includes sixteen large prompt-tokenization operations.

Pretokenized requests preserve the exact chat-template token sequence while
moving tokenizer work outside the timed interval:

| Cohort | External restores per rank | Client min | Client p50 | Client max |
|---:|---:|---:|---:|---:|
| C2 | 1 | 0.541 s | 0.541 s | 0.737 s |
| C8, retained lease | 0 | 0.407 s | 1.211 s | 1.218 s |
| C16 | 1 | 0.735 s | 2.698 s | 2.701 s |
| C16, shared exact prefix and distinct tails | 1 | 2.407 s | 3.894 s | 3.896 s |

The pretokenized C16 restore used 104.2--165.3 ms of cache service per rank
and at most 3.2 ms of restore-queue wait. Every request completed, queues
drained to zero, no engine exited, and the historical post-run canary found the
expected marker suffix.

Diagnostic shared-prefix receipts:

- `evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-128k-c2-pretokenized.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-128k-c8-pretokenized.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-128k-c16.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-128k-c16-pretokenized.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-128k-c16-shared-trunk-pretokenized.json`;
- `evidence/glm53-flash-dflash7-bf16/post-hotlease-2b86fb9-semantic.json`.

Baseline receipts:

- `evidence/glm53-flash-dflash7-bf16/native-128k-c2-b00c6d4.json`;
- `evidence/glm53-flash-dflash7-bf16/native-128k-c8-hot-b00c6d4.json`;
- `evidence/glm53-flash-dflash7-bf16/native-128k-c16-hot-b00c6d4.json`;
- `evidence/glm53-flash-dflash7-bf16/post-b00c6d4-semantic.json`.

## Implemented shared-prefix ownership

One request leads restoration for a persistent digest. Followers wait without
allocating another external restore. A lease becomes visible only after every
worker reports successful restoration and the scheduler normalizes the
leader's multi-group HMA block tables.

The recurrent-cache manager's authenticated checkpoint slot replaces the
logical partial-boundary slot when required. Every physically partial
2,304-token page is copied into a dedicated immutable block before the lease
becomes attachable. Followers acquire ordinary vLLM block references and use
copy-on-write handling for private tails.

The implementation permits sixteen waiting followers per leader, retains at
most two reusable leases, and expires a lease after fifteen seconds. Allocation
pressure releases lease references before refusing ordinary serving blocks.
Failure to create or attach a lease skips sharing; it does not make unverified
state eligible. These bounds describe the implementation. They have no
exact-output semantic concurrency qualification in this record.

## Capacity observations and planned bounds

The exact DFlash7 runtime used a 20 GiB KV-cache setting. C2×128K is an observed
safe capacity candidate, not a qualified cached workload. C6×128K admitted only
one request at a time, used approximately 39–41% of GPU KV capacity for that
request, and serialized completion over 61–313 seconds. It therefore does not
establish C6 capacity or concurrency support.

C8×64K and C16×32K are planned qualification points and have no live evidence.
Sixteen independent 131,072-token requests are unsupported at 20 GiB unless
requests share a GPU-resident trunk or the deployment provides more KV
capacity.

## Verified-or-recompute recovery

The pinned vLLM scheduler patch handles hybrid requests whose restored state is
invalid in any KV-cache group. It discards the complete external prefix across
all groups and computes the request normally. Partially verified hybrid state
is never published.

A live recovery canary removed rank 3's manifest for one advertised
4,096-token entry. Ranks 0--2 verified their pages, rank 3 reported the missing
entry, and the scheduler discarded all external blocks before recomputation.
The request completed in 1.94 seconds, health remained 200, and no traceback or
engine exit occurred. Evidence:
`evidence/glm53-flash-dflash7-bf16/hotlease-2b86fb9-recovery-canary.json`.

The ten-file vLLM source contract and GPU-free patch tests cover single-group,
multi-group, disjoint-group, and asynchronous recovery behavior.

## Diagnostic records

Several receipts document safe optimization rejection while shared HMA block
ownership was being derived. They are historical diagnostics, not qualified
performance results:

- `evidence/glm53-flash-dflash7-bf16/singleflight-128k-c16-830a117.json`;
- `evidence/glm53-flash-dflash7-bf16/singleflight-retention18432-128k-c16.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-d30cdea-128k-c16.json`;
- `evidence/glm53-flash-dflash7-bf16/hotlease-599b65a-128k-c16.json`.

The retention-interval experiment did not produce all-group local-prefix hits
and is not part of the deployment contract. The implementation uses
explicit vLLM block references instead of depending on ordinary hash
rediscovery; this record does not qualify that path semantically.

## Repository validation

Source revision `2b86fb9d02fa3595cca5caa864b81aedce44b8bb` passed:

- `python -m pytest sparkcache -q`: 678 passed, 4 skipped;
- `python -m pytest deploy -q`: 99 passed, 1 skipped;
- `python -m ruff check sparkcache deploy`: all checks passed.

## Qualification limits

- The qualified workload is C1 full-snapshot restore at 131,072 tokens.
- PR43 bounded two-arena flat-v2 read is implemented and GPU-free tested, not
  live qualified. It does not inherit the `35b58a7…` result.
- Historical C2/C8/C16 timing and completion receipts are diagnostic because
  they do not carry exact-output semantic proof.
- Tail-delta publication, shared host-base reads, shared GPU-prefix attachment,
  and multi-root cached concurrency are research-only.
- At 20 GiB, C2×128K is an observed capacity candidate; C8×64K and C16×32K are
  planned and unqualified; C16×128K is unsupported without GPU trunk sharing
  or additional KV capacity.
- Sparse row-prefix aliases are implemented and GPU-free tested, but this GLM
  record does not qualify them because GLM uses opaque page storage.
- Neither the `da4d7be6` record nor local images `ed60…`, `cc2c0e2…`, and
  `35b58a7…` qualify tail-only publication or restored-delta concurrency.
- No cross-TP canonical shard format or network storage backend is implemented.
