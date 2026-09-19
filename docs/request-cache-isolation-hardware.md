# Request cache isolation hardware qualification

Status: **implemented; hardware qualification in progress**.

## Recorded candidate checks

The private candidate image
`sha256:c7a92265c93a4b9f18a487919c8fdb1e7372184431462f6bc836481d8a3d3095`
combines Qwen support with request-scope isolation from SparkCache commit
`677b172fed46845d55e8b3409622c4ec11acec7d`. The main-reconciled commit
`27ef4454af3d8bd729b10a4049cb9fcd628e55fd` has an identical serving package.
The [machine-readable record](request-cache-isolation-qualification.json)
binds conditions, per-rank counts and private evidence hashes. These observations
do not qualify the published SparkRing parent image for request-scope isolation.

| Conditions | TP2/DCP1 | TP4/DCP1 |
|---|---|---|
| Empty synthetic store, salt A | Correct answer, zero restore credit | Correct answer, zero restore credit |
| Repeat A after local prefix reset | 5,696 tokens restored on each rank | 7,200 tokens restored on each rank |
| Identical prompt with unseen salt B | Correct answer, zero restore credit | Correct answer, zero restore credit |
| Repeat B after local prefix reset | 5,696 tokens restored on each rank | 7,200 tokens restored on each rank |
| Omitted salt, then local reset and repeat | Cold miss, then 5,696 restored on each rank | Cold miss, then 7,200 restored on each rank |
| Concurrent A/A/B/B/C submissions | Four warm hits; unseen C misses; correct answers | Four warm hits; unseen C misses; correct answers |
| Restart all processes, replay A/B and unseen D | A/B each restore 5,696 on each rank; D misses | A/B each restore 7,200 on each rank; D misses |
| Genuine pre-scope entry presented to scoped connector | Pending hardware check | Intact legacy entry misses; scoped publication then restores 7,200 on every rank |

The prompt contains 8,194 tokens and an exact retrieval key. External-restore
claims above require worker logs from every physical rank, not merely API cache
credit. Local resets preserve the external store and use loopback-only developer
endpoints through SSH. Production cache directories are not used by the checks.
Concurrent submission does not establish shared-flight joining inside the engine;
that behavior remains covered by GPU-free tests rather than a hardware claim.
No throughput or long-duration stability qualification is implied.

The TP4 legacy-entry check uses the intact 8,194-token control from the
[Qwen TP4 corruption-recovery qualification](qwen38-tp4-cache-qualification.json).
Every payload object, aggregate snapshot and manifest was verified
before copying unchanged into the stopped candidate's isolated store. After
restart, the unsalted request recomputed with zero external restore credit;
after publication and a local prefix reset it restored 7,200 tokens on every
rank. The legacy manifest and payload hashes remained unchanged afterward.
This distinguishes namespace rejection from accepting a corrupted entry or
disabling all restoration.

## Protocol

This protocol qualifies `cache_salt` routing through the original vLLM Request,
the SparkCache scheduler and every physical worker. Run it independently on
TP2/DCP1 and TP4/DCP1 using the exact candidate image. GPU-free scope tests are
necessary but do not prove that an API server forwards the field correctly.

## Preconditions

- Record the immutable image, vLLM and SparkCache source revisions, checkpoint
  revision, source-bound native contract, native snapshot-library checksum,
  physical ranks, cache settings and complete launch configuration.
- Use an isolated synthetic cache root on every rank. Do not corrupt or delete
  production entries. Every worker must run the same connector implementation.
- Verify the snapshot library admits 64 manager groups for Qwen. Python's
  descriptor layout is unchanged, but an inherited 16-group library is invalid.
- Drain other requests before sequential cases. Preserve logs with timestamps,
  request IDs and per-rank capture/restore evidence. Do not log raw salt values.
- Select a deterministic exact-answer prompt longer than two physical manager
  pages, with its retrieval key inside the first page. For the Qwen geometry
  with 2,848-token pages, approximately 8,194 prompt tokens permits a 5,696-token
  retained prefix. Read actual geometry; do not hard-code this credit for GLM.
- Disable or explicitly flush local prefix reuse before external-restore cases,
  using the serving engine's supported idle-only API. A process restart also
  clears local reuse. Do not interpret aggregate `cached_tokens` as proof of
  external disk restore.

## Requests

Use the same token-identical prompt with salts A and B, two distinct generated
strings. Send `cache_salt` as a top-level field in `/v1/chat/completions` JSON
(or `extra_body={"cache_salt": ...}` with an OpenAI client). Use temperature zero,
bounded output and an exact expected retrieval answer. Save the request
fixture privately; public evidence should identify cases as A and B only.

| Case | Procedure | Required evidence |
|---|---|---|
| A cold | Submit A to empty synthetic roots; wait for bounded asynchronous publication to settle | Correct answer; all ranks publish the same scoped context digest under their own shard identities |
| A warm | Clear local prefix reuse; submit A again | Correct answer; positive external restore credit on every rank; no recomputation counted as restoration |
| B cold | Clear local reuse; submit the identical prompt with B | Correct answer; zero external credit from A; B publishes a different scoped digest |
| B warm | Clear local reuse; submit B | Correct answer; all ranks restore B's digest, never A's |
| Unsalted | Omit `cache_salt`, then repeat after clearing local reuse | First request misses A/B entries; repeat can restore its own namespace |
| Mixed batch | After restart, release A, A, B, B, and an unseen salt C from one client barrier | Correct answers; C gets no external hit; A/B restore only their own digests; no cross-scope flight follower association |
| Disk restore | Restart every serving process without removing synthetic roots; replay A and B | Correct answers and matching per-rank disk-restore evidence before accepting positive credit |
| Incompatible namespace | Make a pre-scope synthetic entry available without relabeling it, then send the unsalted prompt | Clean miss and correct recomputation; entry not used as a scoped hit |

Simultaneous client submission does not by itself prove concurrent engine work
or shared-flight joining. Record overlapping request intervals and scheduler
flight identities if available. If requests finish too quickly to exercise a
shared flight, report that hardware subcase as unobserved; retain the GPU-free
same-scope-join and different-scope-miss regression evidence separately. Never
claim hardware flight coverage solely from aggregate hit-rate counters.

## Failure conditions and reporting

Reject qualification on a wrong answer, worker exception, cross-scope digest,
missing rank, restore failure counted as a hit, malformed scope accepted for
placement, or serving stalled behind optional cache work. Client rejection of a
malformed API value proves API validation only; connector-level malformed and
missing metadata cases remain covered by `sparkcache/test_request_cache_salt.py`.

Report conditions, observed restored-token counts, per-rank digest matches,
answer results and unresolved coverage gaps. State that unsalted legacy entries
intentionally miss under `sparkcache-context-v3-request-scope`; no cache deletion
or `CacheIdentity` wire-format migration is required. A salt controls reuse,
not authentication, storage permissions or tenant quotas.
