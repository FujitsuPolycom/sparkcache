# Asynchronous capture through the generic KV connector API

Status: **implemented**. Capture retirement and persistent restore are
**qualified** for the TP4/DCP1 cases and source revisions in the
[GB10 validation record](evidence/connector-job-gb10-tp4-dcp1.md).
Other runtime compositions and failure conditions require separate GPU evidence.

The connector can persist immutable attention pages and retained recurrent
checkpoints using JJ's scheduler-local block-state snapshot. The adapter uses
`bind_gpu_block_pool`, `KVConnectorWorkerMetadata.aggregate`, and
`has_pending_push_work`; it does not require a SparkCache-specific scheduler hook.
The versioned compatibility contract is `jj-block-state-read-leases/v1`.

## Capture ownership

1. The scheduler supplies exact block IDs and recurrent boundary offers for one
   scheduled step. A recurrent offer names its group, physical page, and token
   boundary. Table position alone cannot establish a checkpoint source.
2. SparkCache selects complete immutable attention pages and exactly the offered
   recurrent page at the persisted boundary. It requires every source to be
   non-null and hashed, then takes one block-pool reference per unique physical
   page before dispatching connector metadata.
3. The worker's `wait_for_save` records a CUDA producer event after target and MTP
   state writes. It queues a capture job without invoking the native ring. The
   background submitter waits for that event on its CUDA stream and calls the
   native ring, including any native submission-error recovery.
4. When the native read-completion event is ready, the worker reports its physical
   TP rank for that job. The scheduler releases references only after all distinct
   physical ranks acknowledge the read. Request termination and file commit do
   not control this read lifetime.

The source pages remain protected across request cleanup. Preemption marks queued
or submitted capture work abandoned; it does not drain native work on the model
thread. A job cancelled before submission is acknowledged only after the progress
thread proves that no read started. A submitted job is acknowledged after its read
retires. Native errors with uncertain retirement disable optional capture and
retain a bounded set of source references; they do not fabricate completion or
keep an otherwise idle engine spinning.

Capture job IDs contain a lowercase UUID epoch and a positive 64-bit sequence.
Each worker retains one epoch and its highest accepted sequence. Active,
uncertain, and shutdown-draining ownership is checked before that watermark, so
a repeated active job cannot release its sources. Retired duplicates, late or
out-of-order offers, and a different epoch skip optional capture without launching
another read. The epoch cannot reset within a runtime instance; constructing a
runtime establishes a separate lifetime. This bounded replay guard deliberately
permits cache misses for reordered offers.

The native ring's `drain_context` scans native slots by context sequence, including
quarantined slots for which no valid Python ticket was returned. Submission,
recovery, and polling run outside the callback bookkeeping lock in job mode.
Explicit shutdown may wait for background work to retire.

## Restore and checkpoint boundaries

Stored recurrent bytes come only from an exact offer in the producing step.
Offers are not carried into a later step. If a required group, source hash, or
complete attention page is missing, the connector declines that optional store.
Zero-forward steps emit no stores because JJ's `no_forward` callback skips
`wait_for_save`.

The existing authenticated manifest and asynchronous restore paths remain in use.
Restore records a caller-stream event before background placement, waits for the
restore stream before reporting `finished_recving`, and retains all-rank
completion and failed-load recomputation. The generic adapter does not synthesize
the separate shared-prefix lease-publication acknowledgement used by the
request-lifetime adapter.

The four-checkpoint GLM configuration can offer several retained destinations
within one 8192-token forward. SparkCache selects the exact persisted boundary;
it does not require storing every offered destination. CPU tests with TP4/DCP4,
MTP3, physical/lookup blocks of 512 tokens, and scheduler blocks of 2048 tokens
select persisted spans of 6144, 8192, and 14336 tokens for prompts of 8192, 10240,
and 16384 tokens respectively. Recomputing the uncached tail is expected.

## Source compatibility

The source contracts below identify compositions reviewed with the actual CPU
scheduler, block pool, metadata types, and worker connector callback. A source
contract verifies compatibility; it does not grant GPU qualification by itself.

| vLLM composition | Source contract |
| --- | --- |
| JJ with four-checkpoint coalescing and token-sharded mHC, `abb715f132bdccb592a34b2596a3d3a8d757ffbc` | `sparkcache/runtime_patches/vllm-connector-jobs-jj-prefill-abb715f.json` |
| R27 with four-checkpoint coalescing and token-sharded mHC, `5dede5bb7fa04949a02823411f2fdf135e29b3dc` | `sparkcache/runtime_patches/vllm-connector-jobs-r27-prefill-5dede5b.json` |
| R27 composition with hybrid failed-restore recovery, `df62335d8248587f8d3fd1d9a234d1c162a9b84d` | `sparkcache/runtime_patches/vllm-connector-jobs-hybrid-recovery-df62335.json` |
| JJ with standalone hybrid failed-restore recovery, `9b87df5d47b9c7163d1105ac5ea8c0a088baafc9` | `sparkcache/runtime_patches/vllm-connector-jobs-hybrid-recovery-9b87df5.json` |
| Shared GLM prefill source with TP2 admission and TP4 recovery, `17bd258075f44dda8b405f384732f3c78d03f308` | `sparkcache/runtime_patches/vllm-connector-jobs-source-contract.json` |

Each contract identifies ownership semantics, required API symbols, and exact
SHA-256 values for ten source files. Hashes describe canonical LF source bytes
used in Linux images. The legacy request-finish source guard is unchanged.
Connector-job mode rejects a legacy contract even when its source hashes match.

Matching class names in another JJ or R27-derived release are insufficient.
Qualifying another composition requires source review of the listed semantics,
CPU conformance, an explicit source fingerprint, and the GPU gates below.

For a reusable upstream contract, the engine could advertise a versioned
capability set covering immutable boundary offers, pool ownership, post-MTP
producer readiness, and distinct-rank read retirement. A conformance suite would
exercise request cleanup and source reuse before/after rank acknowledgements,
then inject delayed CUDA reads and preemption. SparkCache could select this
adapter only for a supported capability version that passes that suite. This
engine-advertised contract is research-only; explicit reviewed source contracts
remain required until the engine supplies it. The maintained review entry points
are `REQUIRED_SEMANTICS`, `REQUIRED_SYMBOLS`, and
`verify_connector_job_contract` in
`sparkcache/runtime_patches/generic_connector_contract.py`.

## Configuration

Use a single `SparkContextCacheConnector` with its module path
`sparkcache.spark_context_cache_connector`, `kv_role="kv_both"`, and
`kv_load_failure_policy="recompute"`. The consumer role preserves JJ's deferred
free behavior during asynchronous scheduling. Configure these extra fields:

| Field | Value or requirement |
| --- | --- |
| `spark_cache_model_profile` | `glm53-flash-hybrid` |
| `spark_cache_publication_schema` | `tail-cow-v2` |
| `spark_cache_async_page_capture` | `1` |
| `spark_cache_async_page_capture_lease_mode` | `connector-jobs` |
| `spark_cache_async_page_capture_lease_contract` | Absolute installed path to one reviewed contract above |
| `spark_cache_async_page_capture_vllm_root` | Absolute installed vLLM source root |
| `spark_cache_async_page_capture_library` / `_sha256` | Absolute native snapshot-library path and its SHA-256 |
| `spark_cache_async_page_capture_slot_bytes` | Bounded byte capacity sufficient for an accepted page payload |
| `spark_cache_async_page_capture_slot_count` | `2` or `3` |
| `spark_cache_cuda_placement_library` / `_sha256` | Absolute native placement-library path and its SHA-256 when CUDA restore is enabled |
| `spark_cache_root` | Rank-local persistent storage root for the exact serving composition |

Keep row-oriented streaming snapshots and periodic full captures disabled for
this qualification. Capture backpressure is bounded by the scheduler's delayed
store limit, one queued or committing store per worker, and native ring capacity.
A busy or undersized ring skips optional work without a synchronous fallback.

Cache identities include checkpoint hashes, TP/DCP layout, group geometry, and
checkpoint capacity. These changes do not alter that wire identity. Use a separate
cache root for a different mHC reduction composition: arithmetic associations are
not encoded in the identity and cross-composition state equivalence is unqualified.

## Validation scope

The [integration CPU record](evidence/connector-job-integration-cpu.json)
identifies the package-source digest, 1305 passing tests, platform-specific
skips, source-contract verification, and isolated distribution checks.

GPU-free regressions cover exact boundary selection, refusal to reuse stale
offers, real block-pool references through request cleanup and allocator reuse,
four physical-rank acknowledgements, persistent CPU-byte roundtrips, and producer
event recording before queue admission. Failure tests use the actual Python
`NativeManagerPageRing` wrapper with a blocked backend to verify callback progress
during backend submission failure and invalid-ticket recovery. They do not run
the CUDA implementation.

The GB10 validation record covers four short cold requests, two explicit request
cancellations, and two exact-answer restores after all model processes restarted.
All four physical ranks completed native capture; pending work and retained
source references returned to zero after each retirement case. These results
apply to SparkCache `2bc05bc9e94a4344758e48db36f69a46dafa6946` and the listed
vLLM composition. The additional [whole-prefix restore ownership
guards](PRIVATE_RESTORE_SAFETY.md) have CPU regression coverage and require a
GPU check at the integrated revision.

For each additional serving composition, run the native CUDA ring tests and a model check
with cache misses, persistent hits, continuation, concurrent decode, preemption,
eviction, and forced capture backpressure. Confirm every rank completes its read
before source reuse and that failed restores recompute. Compare against a cold
run with the same model, CP/mHC flags, topology, and retained-state geometry.
CPU conformance establishes metadata and reference ownership; it does not
establish model-quality equivalence or a performance gain.
