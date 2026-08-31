# StreamingSnapshotRuntime integration boundary

**Status: research-only.** Scheduler/worker adapters and connector seams are
implemented behind the default-off `spark_cache_streaming_snapshots` option.
Explicit opt-in lazy-installs the builtin adapter for that connector role, then
is rejected unless the C++/CUDA library, its pinned SHA-256, the
runtime-pinned vLLM block-lease contract, and the declared GLM-5.2 cache
inventory all attest. Qualification requires a four-Spark cache-off versus
streaming-cache live-model comparison.

## What is implemented

`runtime.py` composes the implemented primitives without importing torch,
CUDA, vLLM, or the C++/CUDA ABI:

1. `begin_context()` admits one digest through duplicate/capacity suppression
   and opens an invisible transactional writer journal.
2. `accept_completed_prefill()` accepts a monotonic completed-token watermark,
   the request's current physical block table, and the actual producer CUDA
   stream handle.
3. The planner rounds the watermark to 256-token journal chunks and emits
   bounded macro-batches.
4. Each batch maps to only its physical block-table slice. The runtime leases
   those block IDs without waiting and expands the DCP-owned global positions
   into C++/CUDA per-token physical row slots.
5. `LeaseHandle.submit()` atomically submits the C++/CUDA gather and publishes a
   completion fence. C++/CUDA or lease backpressure aborts only caching.
6. `poll()` releases the source-block lease when the gather is GPU-complete,
   claims the READY ring view, and hands it to an injected writer transaction.
7. The claimed ring slot remains owned until the writer completion says it no
   longer reads the view.
8. The manifest is committed only after the planner covers the declared span,
   every batch is durably written, and no C++/CUDA ticket remains.
9. `take_committed()` exposes the digest together with its exact
   `CommitReceipt` only after that manifest visibility point. Unbounded
   connectors can advertise it immediately; bounded connectors enqueue the
   receipt to their background capacity worker and advertise only if the
   manifest survives serialized maintenance. Aborted or partial journals
   never enter this handoff.
10. `take_aborted()` exposes asynchronous writer/manifest failures to the
    worker adapter, which suppresses later offers for that request while
    serving continues. Preemption remains resumable: it aborts the preempted
    C++/CUDA context and releases its source blocks, but a later scheduler offer
    may begin a fresh transaction with the current block table.

`factory.py` supplies the model-serving facades around this state machine. The
scheduler facade tracks emitted offers so `request_finished()` delays physical
block reuse only while a gather may still read them. The worker facade is
constructed at connector startup but does not load C++/CUDA code or allocate the
ring until `register_kv_caches()` supplies the final canonical tensors. It
binds the exact 79 target-CKV plus 22 sparse-indexer GLM-5.2 inventory, retains
the contiguous row aliases for the ring lifetime, and assembles the frozen
mapped-host/depth-2/64-MiB profile.

## Stable publisher protocol

Publisher code imports only these protocols from
`sparkcache.streaming.runtime`:

```python
SnapshotJournalWriter.begin_context(...) -> SnapshotJournalTransaction
SnapshotJournalTransaction.submit_ready(batch, view) -> WriterCompletion
SnapshotJournalTransaction.commit_manifest() -> CommitReceipt
SnapshotJournalTransaction.abort()
WriterCompletion.query()
WriterCompletion.synchronize()
WriterCompletion.result()
```

The runtime is the exclusive owner of block leases and every ring operation:
submit, poll, claim, release, abandon, and shutdown. The publisher receives a
borrowed READY view but never a C++/CUDA ticket or ring handle.

Runtime invokes `submit_ready()` in increasing `batch_index` order. The
returned completion owns the borrowed view until `query()` becomes true (or
`synchronize()` returns). That boundary means the writer will never read the
view again and `result()` can return immediately without waiting for disk.
The writer may reach it by either:

- durably appending the chunk; or
- copying into immutable writer-owned storage whose transaction guarantees
  durability before `commit_manifest()`.

The runtime releases the ring slot only after that boundary and after calling
`result()`. A `result()` error releases the no-longer-borrowed slot and aborts
the invisible journal. `commit_manifest()` is invoked only after every
completion succeeded. `abort()` permanently suppresses visibility but may
return before already-owned background buffers finish; it never takes ring
ownership.

Cancellation, preemption, mapping failure, bounded-capacity pressure, C++/CUDA
`WOULD_BLOCK`, writer failure, and manifest failure abort the invisible
journal. A cancellation after GPU completion does not wait for disk: source KV
leases are already releasable, while claimed staging slots drain in later
`poll()` calls. Unknown CUDA, fence, or C++/CUDA ownership is fatal and retains the
lease instead of risking block reuse.

## Model-serving vLLM callback seam

The connector expresses the required completion edge with existing
KV-Connector-V1 callbacks:

1. `build_connector_meta()` emits a scheduler **promise** containing the
   request ID, digest/span, promised completed watermark, and complete current
   block table. It also gives the scheduler facade the same metadata for
   delayed-free ownership.
2. vLLM performs the corresponding target, sparse-indexer, and MTP forward.
3. The worker's post-forward `wait_for_save()` converts that promise into a
   completed offer and supplies
   `torch.cuda.current_stream().cuda_stream`. It then polls the streaming
   runtime even when the step has no offers, so writer completion and
   backpressure state continue to advance.

At request completion, vLLM gives worker `get_finished()` every newly finished
request ID, not just IDs retained for asynchronous sends. The worker adapter
therefore intersects that set with the offers it actually observed before
asking the lease registry which owned requests are releasable. This ownership
filter is mandatory: the registry intentionally treats an owned request with
no lease as ready so publication abandonment cannot strand scheduler blocks, but
an entirely unseen request must never be reported as an async-send completion.

This seam is valid only because `wait_for_save()` is invoked:

- after target CKV, sparse-indexer state, and MTP draft KV for every token
  below `completed_token_watermark` have been produced;
- on the worker rank that owns the registered cache tensors;
- with the request's still-owned, current block table;
- with the actual CUDA stream whose ordering proves those writes precede the
  gather;
- once per affected request in a mixed prefill batch;
- before any preemption/eviction path may recycle those blocks.

The watermark must be monotonic per request. It may advance by an arbitrary
amount; the runtime rounds down to complete 256-token chunks and suppresses
already offered ranges. Each emitted batch records its own supplied producer
stream; PIECEWISE, eager, and full-graph steps may legitimately use different
streams because the C++/CUDA ABI orders each submission independently.

The scheduler promise must never be handed to C++/CUDA code early. The
model-serving worker adapter accepts it only from `wait_for_save()`, after the
forward, and submits through `LeaseHandle.submit()` so no preemption path can
reuse a source block between C++/CUDA submission and completion-fence
publication.

## Opt-in assembly contract

Explicit opt-in assembles:

- the mapped/depth-2 `NativeSnapshotRing` selected by the standalone matrix;
- a bounded `BlockLeaseRegistry`;
- a `StreamingSnapshotCoordinator` with `chunk_tokens=256`;
- a journal writer adapter whose `submit_ready()` consumes the borrowed view
  until its returned ownership-transfer completion, never owns the ring, and
  whose `commit_manifest()` is the sole visibility point.

`ManifestSnapshotJournalWriter` supplies that writer adapter. It splits one
macro-batch READY view into the existing 256-token `ContextChunk` records,
derives DCP logical positions, and commits only exact full-span coverage.
Runtime owns every ring ticket and source-block lease; the publisher owns
neither.

## Deliberate non-goals

- The connector feature remains default-off and is not silently enabled.
- No C++/CUDA library is loaded, hashed, or allocated while the option is off, on
  the scheduler role, or before worker cache registration.
- No model, Spark, SSH, or CUDA probe is run.
- No inference callback waits on capacity maintenance or its retries.
- No manifest is published from a partial context.
- Standalone C++/CUDA probes and GPU-free connector/runtime/publisher tests do
  not claim live-model performance or zero regression; the four-Spark serving
  comparison is required for qualification.
