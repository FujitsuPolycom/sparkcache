# Streaming snapshot opt-in

**Status: research-only.** Streaming publication is off by default. A
deployment profile may enable it only for an exact cache layout and vLLM
ownership contract.

Set `spark_cache_streaming_snapshots` or
`SPARK_CONTEXT_CACHE_STREAMING_SNAPSHOTS` to `1` or `true` to opt in. Both also
accept explicit `0` or `false` values.

When disabled, SparkCache does not load the snapshot library, allocate a ring,
reserve KV blocks, change scheduler metadata, or alter end-of-prefill
publication.

## Required inputs

Streaming starts only when:

- the snapshot library path is absolute;
- the configured SHA-256 matches the library;
- the installed vLLM files match the profile's ownership contract;
- worker cache registration matches the declared source inventory;
- the profile defines record order, strides, draft policy, and position
  mapping; and
- the profile defines bounded arena size and ring depth.

The scheduler remains CUDA-free. A worker loads the library and allocates the
ring only after `register_kv_caches()` supplies its complete tensor inventory.

## Request flow

1. The scheduler offers a request ID, digest, span, completed watermark, and
   block table.
2. Post-forward worker code supplies the producer CUDA stream and the largest
   completed aligned prefix.
3. The worker maps and reserves only the physical blocks in that range.
4. Lease reservation and CUDA submission happen as one operation.
5. CUDA completion releases the source blocks and exposes a read-only ring
   view to the writer.
6. The ring slot stays owned until the writer has finished reading the view.
7. A manifest becomes visible after every expected range is durable.

Backpressure, cancellation, preemption, writer failure, or shutdown abandons
the optional publication. It does not delay unrelated serving or expose a
partial manifest.

## Ownership

The C++/CUDA payload is a profile-defined, layer-major batch. A translator
converts it into canonical SparkCache records and derives any position record
that is not present in the CUDA payload.

`ContextChunk` owns immutable bytes. A ring-backed view remains borrowed until
the writer appends it durably or copies it into writer-owned storage.

The runtime owns ring tickets and block leases. The writer owns neither. The
manifest commit is the only visibility point.

## Live testing

GPU-free tests cover the state machine and byte comparison. A deployment
profile must separately record live output correctness, exact source and
artifact identities, and cache-off versus cache-on interference.

Profile-specific procedures live under [`../../deploy/`](../../deploy/).
