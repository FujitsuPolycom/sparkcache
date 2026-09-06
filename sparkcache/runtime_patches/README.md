# vLLM ownership contracts

SparkCache can read KV blocks after a request's forward pass. vLLM must keep
those blocks alive until the asynchronous read is complete.

An ownership contract is a JSON inventory of the exact vLLM source files that
provide this behavior. It uses whole-file digests rather than a fuzzy symbol
or patch check.

## Required behavior

1. `KVConnectorBase_V1.request_finished()` retains a finished request while
   cache publication may still read its blocks.
2. Each worker reports the request through `get_finished()` after its CUDA reads
   complete.
3. The scheduler frees the blocks after every worker reports completion.

Verify an unpacked vLLM tree before enabling streaming publication:

```bash
python -m sparkcache.runtime_patches.verify_lease_contract \
  --vllm-root /path/to/vllm/source \
  --contract /path/to/profile-lease-contract.json
```

Deployment profiles choose the contract and source revision. The generic
verifier does not select a runtime from a model name.

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

`handle.submit()` performs CUDA submission and completion-event publication as
one synchronized operation. Splitting them could let vLLM recycle blocks while
SparkCache is still reading them.

If submission status is unknown, the worker keeps the lease and stops. It must
not release blocks without proof that CUDA has finished.

## Completion and preemption

Poll `leases.poll()` once per worker step. Report only finished request IDs that
belong to the streaming adapter:

```python
owned_finished = finished_req_ids & streaming_seen_request_ids
ready = leases.take_finished(owned_finished)
return ready or None, None
```

The ownership filter prevents ordinary requests from being reported as
completed asynchronous sends.

Preemption cancels an unsubmitted reservation or drains submitted work before
block reuse. A completion event plus `query()` or `synchronize()` provides the
ownership proof.

## Failure behavior

- Capacity exhaustion and block overlap abandon caching without waiting.
- A proven pre-submission failure releases the reservation.
- A submitted lease remains owned until its completion event succeeds.
- Uncertain event state stops the worker instead of recycling blocks.
- Shutdown aborts or drains every lease before destroying staging arenas.
- A partial publication remains invisible.

These contracts do not change cache identity, digest salts, or stored geometry.
Runtime-specific files and integration steps live with their deployment
profiles.
