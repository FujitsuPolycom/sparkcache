# CUDA manager-page capture contract

Status: **research-only**. The descriptor, C++/CUDA submission source,
connector integration, exact-hash vLLM ownership contract, and GPU-free tests
are implemented. No compiled artifact or serving result supports enabling it
in an operator profile.

## Purpose

Opaque manager-page storage preserves every byte owned by vLLM's hybrid memory
allocator, including split kernel rows and bytes outside a tensor's logical
shape. The ordinary publication path copies those pages in
`SparkContextCacheConnector._snapshot_hybrid_store()` before handing an
immutable snapshot to the background commit thread.

The copy is synchronous from the inference callback's perspective:

```text
select request-owned manager pages
  -> gather each layer with PyTorch
  -> copy each gathered tensor to CPU memory
  -> concatenate the encoded snapshot
  -> queue the filesystem commit
```

Only the filesystem commit is asynchronous. The GPU gathers, device-to-host
copies, CPU allocations, and snapshot encoding happen before the log reports
that the background commit was queued.

[SparkCache issue 45](https://github.com/FujitsuPolycom/sparkcache/issues/45)
records a GLM-5.3 TP4/DCP1 publication of 942,592 tokens. Under image
`sha256:77da063d1d51fa181eb39e519dda7c5ae4eb59a47e169cb4c33bd2cd42120225`,
each rank spent 16.6–17.9 seconds in capture while unrelated generation fell
to approximately 0–2 tokens per second. The result establishes interference in
the pre-commit capture phase; it does not isolate one hardware resource as the
sole cause.

## Implemented CPU contract

[`include/spark_cache_page_capture.h`](include/spark_cache_page_capture.h)
defines fixed-width descriptors for:

- stable layer allocations viewed as physical manager pages;
- one request's group-qualified physical page tables;
- one output span per layer; and
- the exact byte count required from a bounded ring slot.

[`src/spark_cache_page_capture_layout.cpp`](src/spark_cache_page_capture_layout.cpp)
validates that:

- groups and layers are dense and ordered;
- every layer in a group has the same physical-page capacity;
- physical page sizes fit their source strides;
- request groups cover their flattened physical-page table exactly;
- every selected page exists in its source group;
- every arithmetic operation fits its declared integer range; and
- the raw payload fits the configured ring slot.

The result is group-major, then layer-major. Each layer span contains request
pages in block-table order. Concatenating the spans produces the exact opaque
body consumed by `encode_page_snapshot()` after its deterministic header.
Physical page IDs and CUDA pointers never enter persistent data.

[`../streaming/manager_page_capture.py`](../streaming/manager_page_capture.py)
provides the same planning seam for GPU-free adapter tests. The connector
preserves synchronous capture unless `spark_cache_async_page_capture` is
explicitly enabled. Row-oriented streaming snapshots remain separate and
continue to reject `block_pages_v1`.

## C++/CUDA ownership sequence

`spark_cache_snapshot_try_submit_pages()` uses the bounded snapshot-ring states
with a separate producer-readiness edge:

```text
request owns every selected page in every group
  -> record readiness event on the vLLM producer stream
  -> low-priority capture stream waits for readiness
  -> copy selected opaque pages into one pinned ring slot
  -> record completion event on the capture stream
  -> completion releases all source-page leases
  -> background progress queues a small header plus claimed ring view
  -> hash and write bounded extents from that scatter view
  -> durable objects precede the manifest
  -> release the ring slot
```

Recording a readiness event avoids placing a multi-gigabyte gather directly on
the producer stream. It does not guarantee negligible interference: GPU copy
engines, unified-memory bandwidth, host memory, and CPU encoding remain shared
resources and require live measurement.

Mapped host arenas are pinned and receive the gather kernel's output. The
connector requires an explicit slot size and allocates two or three slots. It
does not wait for free space; saturation skips publication.

The exact source contract
[`../runtime_patches/vllm-manager-page-async-contract-55969c16.json`](../runtime_patches/vllm-manager-page-async-contract-55969c16.json)
binds enablement to vLLM commit
`55969c16d4da57da76ee5729f3102d4b2003833c`. That source supplies all-group
finished-request ownership and hash-proven, pinned recurrent boundary blocks.

Preemption calls `spark_cache_snapshot_drain_context()`. It synchronizes only
the affected capture before vLLM may reuse those pages. Ordinary completion is
polled on a background thread and never synchronizes the inference thread.

### Producer-stream proof

The attested vLLM model-runner mixin calls `wait_for_save()` from the forward
context manager's `finally` block on the same worker thread that launched the
model. Speculative decoding defers the same callback until its draft forward
has run, still on that model-runner thread. The attested GPU runner does not
enter another PyTorch stream context across that edge.

`wait_for_save()` reads `torch.cuda.current_stream().cuda_stream` and passes
that handle to `spark_cache_snapshot_try_submit_pages()`. The native function
records `producer_ready` on that exact handle. The low-priority capture stream
waits for `producer_ready` before reading any manager page.

## Safety invariants

1. The model allocation and its source strides remain stable for the lifetime
   of the capture handle.
2. The request retains every selected page across all KV-cache groups until
   the capture completion event succeeds.
3. The flattened page table is copied into ring-owned control memory before a
   submission returns; CUDA never reads borrowed Python storage.
4. A streaming capture includes only pages proven immutable at its watermark.
   A mutable terminal attention page or recurrent state is captured only after
   the request stops writing it.
5. Ring pressure returns an immediate publication skip. Serving never waits
   for a free capture slot.
6. Capture failure, cancellation, or uncertain CUDA completion leaves no
   visible manifest. Uncertain source-page ownership stops the worker rather
   than permitting page reuse.
7. Ring bytes remain immutable until the durable worker finishes every bounded
   extent read and releases the slot. No whole-snapshot Python copy is made.
8. Persistent block-page bytes and cache identity remain unchanged. Enabling a
   different physical format requires a distinct cache namespace.

## Explicit opt-in

The feature flag and every artifact setting are required:

```text
spark_cache_async_page_capture=1
spark_cache_async_page_capture_library=/absolute/libspark_cache_snapshot.so
spark_cache_async_page_capture_library_sha256=<64 lowercase hex characters>
spark_cache_async_page_capture_slot_bytes=<complete per-rank snapshot bytes>
spark_cache_async_page_capture_slot_count=2
spark_cache_async_page_capture_vllm_root=/absolute/vllm/source
spark_cache_async_page_capture_lease_contract=/absolute/ownership-contract.json
```

The environment equivalents use the
`SPARK_CONTEXT_CACHE_ASYNC_PAGE_CAPTURE_` prefix. Invalid or incomplete opt-in
configuration stops connector initialization. When the feature flag is absent,
the connector uses synchronous capture without loading the page-capture ABI.

## Qualification conditions

The implementation must remain absent from default profiles until all of these
conditions are recorded for an exact library and model-serving artifact:

- native bytes match the Python snapshot byte-for-byte for split kernel rows,
  physical page tails, attention pages, and recurrent pages;
- cancellation, preemption, stale tickets, event errors, writer errors, and
  shutdown preserve the ownership invariants above;
- a bounded ring returns immediately under saturation; and
- live cache-on/cache-off tests establish the effect on decode throughput,
  prefill throughput, latency, unified-memory headroom, and Python heartbeat
  delay during each bounded extent copy.

Until every condition is recorded, operator profiles must leave
`spark_cache_async_page_capture=0` and use the synchronous Python/PyTorch
capture path.

## GPU-free validation

```bash
python -m pytest -q \
  sparkcache/streaming/test_manager_page_capture.py \
  sparkcache/streaming/test_manager_page_native_ring.py \
  sparkcache/streaming/test_manager_page_runtime.py \
  sparkcache/native/tests/test_page_capture_contract.py

cmake -S sparkcache/native -B build/native \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=OFF
cmake --build build/native
ctest --test-dir build/native --output-on-failure
```
