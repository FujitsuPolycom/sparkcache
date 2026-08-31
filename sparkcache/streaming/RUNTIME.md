# Streaming snapshot runtime

**Status: research-only.** The streaming runtime connects scheduler offers,
worker-owned KV blocks, a C++/CUDA gather ring, and an asynchronous manifest
writer.

Streaming is off by default. See [`OPT_IN.md`](OPT_IN.md) for the exact inputs
required to enable it.

## What the runtime does

1. `begin_context()` admits one digest and opens an invisible writer journal.
2. `accept_completed_prefill()` receives a monotonic token watermark, the
   request's block table, and the producer CUDA stream.
3. The planner rounds the watermark to complete 256-token chunks and creates
   bounded batches.
4. Each batch maps only the physical blocks it needs and reserves them without
   waiting.
5. `LeaseHandle.submit()` gathers the bytes and records completion as one
   synchronized operation.
6. `poll()` releases source blocks after CUDA completion and hands a read-only
   ring view to the writer.
7. The ring slot stays owned until the writer no longer reads that view.
8. The manifest commits after every planned batch is durable.

`take_committed()` exposes a digest only after the manifest exists. A bounded
connector runs capacity maintenance before advertising that digest.

`take_aborted()` reports writer and manifest failures to the adapter so later
offers for the request can be skipped. Serving continues.

Preemption aborts the gathered context and releases its source blocks. A later
offer may start a different transaction with the request's active block table.

## Writer interface

Publishers depend only on these protocols from
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

The runtime owns every block lease and ring operation. The publisher receives
a borrowed ready view, never a ring ticket or handle.

`submit_ready()` calls arrive in increasing batch order. Its completion owns
the borrowed view until `query()` succeeds or `synchronize()` returns.

At that point the writer has either appended the batch durably or copied it to
immutable writer-owned storage. The runtime can then release the ring slot.

`commit_manifest()` runs only after every completion succeeds. `abort()` keeps
the journal invisible, although already-owned writer buffers may finish in the
background.

Cancellation, preemption, mapping failure, capacity pressure, CUDA
backpressure, writer failure, and manifest failure abandon only the optional
publication.

Unknown CUDA or ownership state stops the worker and retains the lease. Reusing
those blocks without proof of completion could corrupt a different request.

## vLLM callbacks

The integration uses KV-Connector-V1 callbacks:

1. `build_connector_meta()` emits a scheduler promise with the request ID,
   digest, span, watermark, and complete block table.
2. vLLM computes every configured record family below that watermark.
3. Post-forward `wait_for_save()` supplies the producer CUDA stream, turns the
   promise into a completed offer, and polls the streaming runtime.

`wait_for_save()` must run while the request still owns the active block table
and before preemption or eviction can recycle those blocks.

At request completion, the worker intersects vLLM's finished IDs with requests
observed by the streaming adapter. This prevents ordinary requests from being
reported as completed asynchronous sends.

The watermark increases monotonically. It may advance by any amount; the
runtime rounds it down to complete chunks and ignores ranges already offered.

Each batch records its producer stream. Different execution modes may use
different streams because the CUDA ABI orders every submission independently.

## Assembly

Explicit opt-in creates:

- a bounded `NativeSnapshotRing`;
- a bounded `BlockLeaseRegistry`;
- a `StreamingSnapshotCoordinator` using 256-token chunks; and
- a journal writer whose manifest commit is the only visibility point.

`ManifestSnapshotJournalWriter` splits one ready batch into ordinary
`ContextChunk` records, derives logical positions, and commits only exact
full-span coverage.

## Limits

- No library loads or allocates while streaming is disabled.
- The scheduler never loads CUDA code.
- Registration must supply a complete profile-defined tensor inventory.
- Cache publication never waits for capacity maintenance.
- Partial contexts never receive manifests.
- GPU-free and synthetic tests do not establish live output or performance.
