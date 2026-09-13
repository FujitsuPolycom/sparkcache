# Request cache isolation hardware qualification

Status: **implemented test protocol; hardware results required**.

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
