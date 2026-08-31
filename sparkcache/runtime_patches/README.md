# SparkCache streaming-snapshot runtime contract

Streaming snapshots do **not** require a vLLM allocator patch. Official vLLM
at the runtime-pinned commit provides the ownership contract SparkCache needs:

1. scheduler-side `KVConnectorBase_V1.request_finished()` may return `True`;
2. the scheduler then retains the finished request and its KV block table;
3. worker-side `get_finished(finished_req_ids)` reports the request only after
   the last asynchronous CUDA gather event completes;
4. `Scheduler._update_from_kv_xfer_finished()` then calls `_free_blocks()`.

`vllm-kv-block-lease-contract.json` pins the complete official source files
that implement this behavior, including `KVOutputAggregator`, whose
world-size completion countdown requires every worker to report an owned
request exactly once. Runtime installation must verify those hashes or
explicitly port and re-pin the same semantics. A fuzzy patch or symbol-only
match is not sufficient.

Verify an unpacked vLLM source tree before installing the streaming connector:

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /path/to/vllm/source
```

The GLM-5.3 Flash integration pins the ten-file serving safety contract
`vllm-kv-block-lease-contract-da4d7be.json` to
`local-inference-lab/vllm@da4d7be6c97434f6942292ed8abbf4b32dc44355`.
The accepted hashes cover both the repository checkout and
the Python sources installed by the GLM-5.3 serving image, and named-state
coherence prevents a mixed source/runtime tree from passing. The contract
includes the Mamba manager and cache-spec files that define recurrent
block-table semantics. That fork includes
deferred block reclamation for overlapping consumer batches and preserves the
connector completion interfaces. Verify it explicitly; the default contract
names a different upstream lineage:

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /opt/spark-vllm \
  --contract sparkcache/runtime_patches/vllm-kv-block-lease-contract-da4d7be.json
```

The source-built GLM-5.3 runtime contract for
`local-inference-lab/vllm@e10536aadf02a18fccddda7ec939c33147e8b0b3`
uses `vllm-kv-block-lease-contract-e10536a.json` and the exact-input overlays
under `patches/vllm-e10536a`. The contract covers the same ten SparkCache-owned
interfaces after the fork added internal MTP5 and opt-in acceptance-length
adaptation. It has **implemented** status and requires four-rank qualification
before replacing a qualified runtime.

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /path/to/e10536a/source \
  --contract sparkcache/runtime_patches/vllm-kv-block-lease-contract-e10536a.json
```

The GLM-5.3 runtime with live-tensor B12X KDA binding pins
`local-inference-lab/vllm@0b67266a0f37d6146a8403fb8482403c62f412d5`.
Its SparkCache integration uses
`vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json` and the
exact-input overlays under `patches/vllm-glm53-b12x-kda-adaptive-mtp`.
The three KDA commits after `e10536a` change only the KDA implementation and
its model tests. Ten SparkCache ownership files remain byte-identical, while
the eleventh contract file binds the live-tensor B12X KDA implementation.
The four SparkCache patches produce the exact preimages consumed by the
recurrent-boundary producer. The contract names one coherent final runtime
state across all eleven files and is valid only after that producer creates its
four recurrent postimages. It has **implemented** status. Four-rank TP4/DCP1
serving remains unqualified until a receipt names an immutable image built from
this revision.

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /path/to/0b67266a/source \
  --contract sparkcache/runtime_patches/vllm-kv-block-lease-contract-glm53-b12x-kda-adaptive-mtp.json
```

This runtime contract does not change SparkCache wire values, digest salts,
256-token chunk geometry, or cache-identity fields. Static embedded MTP and
adaptive embedded MTP retain distinct draft-state identities. Incompatible or
unverified stored state is rejected and recomputed.

## Narrow integration

Use `sparkcache.streaming.BlockLeaseRegistry` in the worker connector:

```python
handle = leases.try_reserve(request_id, completed_chunk_block_ids)
if handle is None:
    abandon_cache_publication(request_id)  # never wait for a staging slot
else:
    try:
        handle.submit(lambda: submit_gather_and_record_event(...))
    except LeaseFenceError:
        # A callback failure after possible submission is fatal and retains
        # the lease. Only a failure proven to occur before handle.submit()
        # may use cancel_before_submit().
        raise
```

Never enqueue CUDA work and then call a separate `arm(event)` method. That
creates a reserve/submit/preemption race in which vLLM can recycle the blocks
after GPU work begins but before the event becomes visible. `handle.submit()`
serializes the submission callback and fence publication against preemption.
If the callback raises, submission status is unknown, the lease remains held,
and the worker stops rather than releasing blocks whose GPU use is uncertain.

Map every logical macro batch to only its relevant physical block-table slice:

```python
block_map = BlockTableRangeMapper(
    block_ids=tuple(request_block_ids),
    logical_tokens_per_block=block_size * dcp_degree,
)
offer = planner.offer_completed(request_id, completed_tokens, block_map)
```

The planner calls `blocks_for_range(logical_start, logical_end)` separately for
each emitted batch. Mapping failure aborts the optional cache transaction.
Adjacent aligned batches therefore lease disjoint physical blocks instead of
all contending on the request's complete block table.

Poll `leases.poll()` once per worker step.  Implement worker-side
`get_finished()` as:

```python
owned_finished = finished_req_ids & streaming_seen_request_ids
ready = leases.take_finished(owned_finished)
return ready or None, None
```

vLLM's `finished_req_ids` contains every request newly finished by that
scheduler output, including ordinary requests that never emitted a streaming
offer.  Never pass that unfiltered set to the lease registry: its intentional
"unknown means ready" rule would echo an ordinary ID as a completed async send
after the scheduler had already removed it.  Scheduler-side
`request_finished()` returns `True` only for a request admitted to a streaming
publication transaction.  Returning `True` when an admitted worker transaction
abandoned before submission is safe: after the ownership intersection,
`take_finished()` immediately reports that seen request without an active
lease, adding at most one scheduler round.

During normal prefill, vLLM already owns all blocks for the live request.  Most
chunk gathers finish while that ownership is still active.  The delayed-free
contract is needed only for the final tail that remains in flight when the
request finishes or is aborted.

## Preemption and eviction boundary

Normal request completion is not the only reclamation path. The pinned
`GPUModelRunner.execute_model()` calls:

```python
get_kv_transfer_group().handle_preemptions(kv_connector_metadata)
```

before persistent-batch updates and before the next forward can overwrite
reallocated blocks. Extend `SparkCacheConnectorMetadata` with:

```python
preempted_request_ids: tuple[str, ...] = ()
```

and populate it scheduler-side with:

```python
tuple(sorted(scheduler_output.preempted_req_ids or ()))
```

Worker-side `handle_preemptions()` delegates synchronously to
`WorkerStreamingSnapshotAdapter.handle_preemptions()`. The adapter validates
the sorted unique request IDs and asks `StreamingSnapshotRuntime.preempt()` to
cancel reservations or drain armed work. It releases leases only after their
completion fences prove that no gather still reads the blocks. If fence
synchronization fails, the runtime retains the lease and raises; the worker
must terminate rather than return to a forward that could overwrite blocks
whose read status is unknown.

Same-stream ordering alone is **not** the safety contract. Model runners,
connector progress threads, graph replay, and separate low-priority copy
streams can establish different stream relationships, while allocator
reclamation is host-side. A recorded completion event plus explicit
query/synchronize is the portable proof that no DMA or kernel still reads the
physical block.

## Required failure behavior

- Capacity exhaustion and block overlap abandon caching without waiting.
- A launch failure before CUDA submission cancels the reservation immediately.
- Once submitted, an armed lease is never cancelled.  Abort synchronizes its
  fence before releasing it.
- A failed fence query/synchronize retains the lease and is fatal to that
  worker; recycling blocks with unknown CUDA-read status would corrupt a later
  request.
- Connector shutdown calls `abort_all()` before destroying staging arenas.
- DCP ranks decide cache publication independently, but the manifest remains
  invisible until the existing all-rank quorum contract succeeds.

## Why not refcount `BlockPool` directly?

Adding an external refcount beneath `KVCacheManager` would touch allocation,
prefix caching, eviction order, hybrid cache groups, preemption, and reset
paths.  It would also duplicate a supported connector lifecycle already used
for asynchronous KV sends.  Delaying the finished request's existing block
table is the narrowest safe seam and is sufficient because streaming copies
occur while the request still owns its blocks.
