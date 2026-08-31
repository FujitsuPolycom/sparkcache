# vLLM ownership contracts

SparkCache pins the vLLM source files that define asynchronous connector
ownership. A contract is a JSON inventory of exact file digests, not a fuzzy
patch or symbol check.

The required lifecycle is:

1. scheduler-side `KVConnectorBase_V1.request_finished()` retains a finished
   request whose cache publication still reads its blocks;
2. each worker reports the request through `get_finished()` only after its
   asynchronous CUDA reads complete; and
3. the scheduler frees the request's blocks after every worker reports
   completion.

Verify an unpacked vLLM source tree before enabling streaming publication:

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /path/to/vllm/source \
  --contract /path/to/profile-lease-contract.json
```

Deployment profiles own the contract selection, source revision, overlays, and
qualification evidence. The generic verifier does not infer a runtime from a
model name.

## Worker integration

Reserve only the physical blocks read by one completed range:

```python
block_map = BlockTableRangeMapper(
    block_ids=tuple(request_block_ids),
    logical_tokens_per_block=block_size * dcp_degree,
)
offer = planner.offer_completed(request_id, completed_tokens, block_map)

handle = leases.try_reserve(request_id, offer.block_ids)
if handle is None:
    abandon_cache_publication(request_id)
else:
    handle.submit(lambda: submit_gather_and_record_event(...))
```

`handle.submit()` serializes CUDA submission and completion-fence publication
against preemption. Never enqueue work and arm its event in a separate call.
That split creates a window in which vLLM may recycle blocks already being read.

If submission status becomes unknown, retain the lease and stop the worker.
Releasing blocks without proving CUDA completion can corrupt another request.

## Completion reporting

Poll `leases.poll()` once per worker step. Intersect vLLM's finished request IDs
with requests actually owned by the streaming adapter:

```python
owned_finished = finished_req_ids & streaming_seen_request_ids
ready = leases.take_finished(owned_finished)
return ready or None, None
```

The ownership filter prevents ordinary requests from being echoed as completed
asynchronous sends. Scheduler-side `request_finished()` returns `True` only for
a request admitted to a streaming transaction.

## Preemption

Worker-side preemption handling must cancel unsubmitted reservations or drain
submitted work before block reuse. A completion event and explicit
query/synchronize provide the ownership proof; same-stream assumptions do not.

The connector metadata carries sorted, unique preempted request IDs to the
worker adapter. Fence failure is fatal and retains the lease.

## Failure behavior

- Capacity exhaustion and block overlap abandon caching without waiting.
- Failure proven to occur before CUDA submission releases the reservation.
- A submitted lease remains owned until its completion fence succeeds.
- Fence uncertainty stops the worker instead of recycling blocks.
- Shutdown aborts or drains every lease before staging arenas are destroyed.
- A partial publication remains invisible.

This contract changes no cache-identity field, digest salt, or persisted
geometry. Runtime-specific contract files and exact integration procedures live
with the deployment profiles that use them.
