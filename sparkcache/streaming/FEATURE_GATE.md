# Streaming snapshot opt-in contract

**Status: research-only.** Streaming publication is default-off and available
only through a deployment profile with an attested cache inventory and vLLM
ownership contract.

`spark_cache_streaming_snapshots` and
`SPARK_CONTEXT_CACHE_STREAMING_SNAPSHOTS` accept explicit `0/1` or
`false/true` values.

When disabled, SparkCache does not construct a ring, import the snapshot CUDA
binding, reserve KV blocks, change scheduler metadata, or alter synchronous
end-of-prefill publication.

## Required attestations

Enabling streaming rejects connector construction unless:

- the snapshot library path is absolute;
- the configured library SHA-256 matches the file;
- the installed vLLM sources match the profile's block-lease contract;
- worker cache registration exactly matches the profile's source inventory;
- the profile declares record ordering, strides, draft policy, and logical
  position mapping; and
- the profile declares a bounded arena size and ring depth.

The scheduler role remains CUDA-free. The worker loads no library and allocates
no arena until `register_kv_caches()` supplies the complete tensor inventory.

## Serving lifecycle

1. The scheduler offers a stable request ID, digest, span, completed watermark,
   and block table.
2. Post-forward worker code supplies the actual producer CUDA stream and the
   largest completed aligned prefix.
3. The worker maps only the physical blocks needed by each offered range.
4. Lease reservation and CUDA submission occur in one synchronized operation.
5. GPU completion releases source-block leases and exposes a read-only ring
   view to the journal writer.
6. The ring slot remains owned until the writer no longer reads that view.
7. The manifest becomes visible only after every expected range is durable.

Backpressure, cancellation, preemption, writer failure, or shutdown abandons
the optional publication. It does not delay unrelated serving or expose a
partial manifest.

## Ownership boundary

The C++/CUDA payload is a model-profile-defined, layer-major macro batch. The
profile translator converts it into canonical SparkCache records and derives
any logical-position record absent from the CUDA payload.

`ContextChunk` owns immutable bytes. A ring-backed view therefore remains
borrowed until the writer either appends it durably or copies it into its own
immutable buffer.

The runtime owns every ring ticket and block lease. The writer owns neither.
The manifest commit is the only visibility edge.

## Qualification boundary

GPU-free runtime, connector, and byte-comparison tests establish state-machine
behavior. They do not establish live-model correctness, performance, or lack
of interference.

Each model-specific streaming profile must record exact source and artifact
identities plus a cache-off versus cache-on live comparison. Registered
profiles are documented under [`../../deploy/`](../../deploy/).
