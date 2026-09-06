# SparkCache CUDA libraries

SparkCache includes two optional C++/CUDA libraries:

| Library | What it does | Status |
|---|---|---|
| `libspark_cache_placement.so` | Places authenticated cache records into request-owned GPU blocks. | **implemented** |
| `libspark_cache_snapshot.so` | Gathers completed cache rows or selected manager pages into a bounded write-behind ring. | **implemented** |

A deployment profile pins the library SHA-256, describes the memory layout,
and records the live tests performed with that exact artifact.

## CUDA placement

The placement library replaces Python object creation, host-side record
transposition, repeated slot-vector uploads, and layer-by-layer scatter calls
during large restores.

```text
read objects into a mapped host arena
  -> verify each complete object
  -> validate headers, offsets, and positions
  -> submit bounded slabs to a CUDA scatter
  -> synchronize and resume the request
```

If any step fails, SparkCache discards the request's private blocks and lets
vLLM compute the prompt normally.

## Python interface

[`../spark_cache_cuda.py`](../spark_cache_cuda.py) defines the Python ABI.
`sparkcache.native.python` re-exports it for compatibility.

`load_library()` calls `spark_cache_placement_query_abi()` before allocating
memory. It rejects incompatible versions, structure sizes, arena constants,
record kinds, and required capabilities.

Serving code uses
[`../spark_context_cache_cuda_placement.py`](../spark_context_cache_cuda_placement.py):

```python
from sparkcache.spark_context_cache_cuda_placement import (
    ArenaMode,
    CudaPlacementAdapter,
    CudaPlacementLibrary,
)

library = CudaPlacementLibrary.load(path, expected_sha256=digest)
adapter = CudaPlacementAdapter.create(
    library,
    arena_mode=ArenaMode.MAPPED_HOST,
    arena_bytes=arena_bytes,
    max_destinations=destination_count,
    max_slots=slot_count,
    max_chunks_per_slab=object_count,
    device_ordinal=device_ordinal,
)
```

`execute_cuda_restore()` owns object packing, full-object verification, arena
lifetime, slab submission, and request completion.
`ParkedRestore.finish()` is the only success path that resumes the request.

Symbols containing `native` remain compatibility interfaces. The canonical
interfaces are the `spark_cache_cuda` and `spark_context_cache_cuda_*`
modules.

## Placement ABI

[`include/spark_cache_placement.h`](include/spark_cache_placement.h) defines a
fixed C ABI with destination descriptors, verified-object descriptors, one
slot vector, two bounded arenas, and explicit transaction states.

Destination descriptors contain runtime pointers, capacities, strides, record
kinds, byte widths, and source ordinals. Persistent metadata contains no
runtime pointers or physical slot coordinates.

[`src/spark_cache_placement_parse.cpp`](src/spark_cache_placement_parse.cpp)
parses objects in place. It checks the format, headers, record offsets, record
kinds, digest syntax, and logical-position ownership.

The caller authenticates each complete encoded object before parsing it.
Per-record hashing would reread the same bytes without strengthening the
manifest's integrity boundary.

Manager-page delta restore uses two mapped host arenas. A bounded read pool
authenticates the following object batch while CUDA places the preceding
batch, then the arenas alternate.

The placement ABI fixes the arena count at two.
`spark_cache_cuda_placement_arena_bytes` changes each arena's bounded size,
and `spark_cache_cuda_restore_io_workers` changes storage-read parallelism.

A third placement arena would consume another arena's worth of pinned unified
memory without increasing storage-read parallelism.

Deployment evidence must show sustained arena waits before changing the ABI
and accepting that memory cost. Snapshot-ring slots are a separate resource
and do not change placement arena count.

## Arena modes

| Mode | Behavior |
|---|---|
| Mapped host | CUDA reads authenticated host pages directly. |
| Managed | CUDA uses a managed-memory arena. |
| Staged device | Pinned host input is copied through device staging. |

Arena support is reported by the ABI. A deployment profile chooses a mode and
size that fit its object geometry and memory budget.

For aligned reads, load the object prefix first, place the complete object at
an aligned arena offset, and hash only the encoded bytes. Padding is not part
of the digest.

## Memory rules

1. An arena moves `FREE -> FILLING -> VERIFIED -> IN_FLIGHT -> FREE`.
2. Reading, hashing, parsing, and position checks finish before submission.
3. Cross-thread producers publish readiness with release/acquire ordering.
4. The CPU never writes an `IN_FLIGHT` mapped arena.
5. The slot vector stays immutable until finish or abort.
6. Direct slabs cover destination rows exactly once and in order.
7. Concurrent streams write only disjoint rows or destinations.
8. A request resumes only after every event and device check succeeds.

An error may occur after an earlier slab wrote private request blocks. No
forward pass can use those blocks; SparkCache frees them before recomputation.

## Snapshot gather ring

`libspark_cache_snapshot.so` gathers completed rows or selected manager pages
into bounded staging arenas for optional write-behind publication. See
[`SNAPSHOT_RING_STATE_MODEL.md`](SNAPSHOT_RING_STATE_MODEL.md).

Backpressure or failure may cancel publication. It must not delay serving or
expose a partial manifest.

Opaque hybrid-memory-allocator pages require group-qualified page tables and
all-group block leases.

The descriptor, sparse manager-page selection, payload layout, and ownership
contract are documented in
[`MANAGER_PAGE_CAPTURE_CONTRACT.md`](MANAGER_PAGE_CAPTURE_CONTRACT.md).

The C++/CUDA path requires explicit opt-in and an attested vLLM page-ownership
contract. Deployment profiles link live evidence for exact compiled artifacts,
model layouts, runtime revisions, and topologies.

## Build and test

Run the GPU-free ABI, layout, and reference tests:

```bash
python -m pytest -q sparkcache/native/tests
```

Build with the same CUDA toolchain as the serving runtime:

```bash
cmake -S sparkcache/native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES='<target-architecture>' \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=ON
cmake --build build/native -j"$(nproc)"
ctest --test-dir build/native --output-on-failure
```

Before enabling CUDA placement for a model, record the exact library and
source identities, reference-path byte equality, continued-generation result,
remapping behavior, cancellation behavior, and memory bounds.

Model names, checkpoints, topologies, measurements, and launch commands belong
in deployment profiles and evidence records.
