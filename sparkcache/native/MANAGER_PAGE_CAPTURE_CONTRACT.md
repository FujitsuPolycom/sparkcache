# CUDA manager-page capture contract

Status: **implemented**. The descriptor, C++/CUDA submission source, connector
integration, exact-hash vLLM ownership contract, and GPU-free tests are part of
SparkCache. Live qualification remains specific to the compiled artifact,
model layout, vLLM source contract, and topology named in an evidence record.

## Purpose

Opaque manager-page storage preserves every byte owned by vLLM's hybrid memory
allocator, including split kernel rows and bytes outside a tensor's logical
shape. Asynchronous capture moves eligible request-owned pages to a bounded
mapped-host ring on a low-priority CUDA stream. A background publisher hashes
and writes the claimed view before releasing its ring slot.

Complete snapshots capture every request page. A `page-tail-cow-v2` extension
captures only pages whose bytes cannot be reused from its authenticated base.
SparkCache conservatively captures partial terminal attention pages and
mutable recurrent or sliding-window state again.

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

The result is group-major, then layer-major. Each layer span contains selected
request pages in block-table order. Complete capture produces the opaque body
consumed by `encode_page_snapshot()` after its deterministic header. Sparse
extension capture carries base-reuse geometry with the claimed ring view so
the background publisher can encode a page delta directly. Physical page IDs
and CUDA pointers never enter persistent data.

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
  -> verify the base once when the publication references one
  -> construct and hash bounded snapshot or delta extents from the ring view
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
8. Direct sparse-delta encoding consumes the verified base once. It does not
   reconstruct a complete result snapshot or compare all result pages.
9. `page-tail-cow-v2` has its own cache-identity wire value. Entries from
   `snapshot-v1` and `page-tail-cow-v1` therefore miss cleanly rather than
   being treated as compatible state.

## Explicit opt-in

The feature flag and every artifact setting are required:

```text
spark_cache_async_page_capture=1
spark_cache_publication_schema=tail-cow-v2
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

## Evidence and limitations

[`../../evidence/glm53-flash-dcp4-page-tail-v2/qualification.json`](../../evidence/glm53-flash-dcp4-page-tail-v2/qualification.json)
records byte-exact chained growth, restart restore, corruption rejection, and
foreground-interference measurements for one GLM-5.3 TP4/DCP4 artifact. The
result qualifies only the identities in that record.

The following limits remain:

- `page-tail-cow-v2` supports opaque manager-page profiles only.
- A flat manifest retains one ordered descriptor stage per published
  extension. Metadata and restore work therefore grow with retained stages.
- Partial terminal attention pages and recurrent or sliding-window state are
  captured again because their earlier bytes cannot be proven immutable from
  token boundaries alone.
- Two or three snapshot-ring slots are supported. Every additional slot
  consumes its full configured size in pinned unified memory.
- Ring saturation skips publication. It never waits for storage capacity.
- A vLLM revision without the attested producer-stream and page-lease contract
  cannot enable asynchronous capture.

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
