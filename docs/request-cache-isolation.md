# Request cache isolation

Status: **implemented; GPU-free regression coverage**. This behavior belongs to
the source checkout, not previously published images or hardware qualifications.

The connector includes an opaque fingerprint of the original vLLM Request's
`cache_salt` in every persistent context digest. Equal salts can reuse matching
prefixes; different salts cannot share external entries, publication bases,
prefix aliases, or scheduler restore flights. `None`, an empty string, and each
distinct Unicode string identify different scopes. Strings are not normalized.
Raw salts are not copied into connector metadata, cache filenames, or logs.

The original Request must expose `cache_salt`, and the field must remain
immutable before and after restore dispatch; changing a salt is not a revocation
mechanism for work already dispatched. The allocation callback records it even
for a zero-external-token allocation, so store-only deployments and native local
prefix hits retain the same scope. Scheduler output that omits this field must
have an already recorded original Request. Missing, invalid, or conflicting
identity bypasses caching rather than assuming an unsalted request. If an
external hit was allocated before rejection, a failed-load completion returns
its blocks for recomputation instead of silently dropping the load plan.

Chunked and resumed prefills retain the scope in request bookkeeping. Store
plans and streaming offers carry a version-tagged fingerprint. Worker checks
reject missing/malformed scope metadata and inconsistent scope/digest fields.
The constant-size consistency checksum is not authentication: the scheduler,
connector metadata channel, and worker processes remain trusted. A process that
can forge complete connector metadata is outside this isolation boundary.
No full prompt is added to restore metadata or rehashed on every worker.

## Compatibility

The context-digest namespace is `sparkcache-context-v3-request-scope`. Every
request, including unsalted requests, cleanly misses entries created with the
preceding namespace: those entries cannot prove whether a salt was ignored when
they were written. They are not deleted or relabeled and remain subject to
ordinary capacity management. New prefills repopulate the cache.

`CacheIdentity` wire fields, shard geometry, native ABI and chunk encodings are
unchanged. Both scheduler and workers must run the same connector implementation;
mixed-version connector metadata is unsupported. A salt separates cache reuse,
not storage access permissions, quotas, or authentication between clients.

## Validation

`sparkcache/test_request_cache_salt.py` exercises scheduler admission, mixed-scope
batches, shared-prefix flights, original-request allocation, chunked/resumed
publication, streaming metadata, row aliases, tail publication, fresh-worker
restore, metadata rejection and failed-load completion on CPU tensors. The
full GPU-free suite covers the unchanged integrity and cache-recovery paths.
Hardware model qualification must be recorded separately for the exact image.
