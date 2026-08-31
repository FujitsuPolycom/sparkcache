# SparkCache CUDA libraries

SparkCache provides two independent C++/CUDA libraries:

| Library | Status | Purpose |
|---|---|---|
| `libspark_cache_placement.so` | **implemented** | Place authenticated cache records directly into request-owned GPU destinations |
| `libspark_cache_snapshot.so` | **research-only** | Gather completed cache rows into a bounded write-behind ring |

Both libraries are opt-in. A deployment profile must pin the library SHA-256,
declare a compatible layout, and provide its own GPU qualification evidence.

## CUDA placement

The placement path removes per-record Python objects, host-side record
transposition, repeated slot-vector uploads, and layer-by-layer scatter calls
from large restores.

```text
read immutable objects into a mapped host arena
  -> verify each complete object digest
  -> validate headers, offsets, and logical positions
  -> submit bounded slabs to a fused CUDA scatter
  -> synchronize and release the parked request
```

No restored request can run until the placement transaction succeeds. If
validation, submission, or synchronization fails, the request's blocks are
discarded and its context is recomputed.

## Canonical Python boundary

[`../spark_cache_cuda.py`](../spark_cache_cuda.py) owns the Python ABI
declarations. The `sparkcache.native.python` package re-exports that module and
does not maintain another ABI definition.

`load_library()` calls `spark_cache_placement_query_abi()` before allocation.
It rejects incompatible versions, structure sizes, arena constants, record
kinds, or required capabilities.

Model-serving orchestration uses the attested adapter in
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

`execute_cuda_restore()` owns object packing, complete-object verification,
arena lifetime, slab submission, and parked-request completion.
`ParkedRestore.finish()` is the only success edge that resumes the request.

Symbols containing `native` remain compatibility interfaces. Canonical imports
use `spark_cache_cuda`, `spark_context_cache_cuda_placement`, and the
`spark_context_cache_cuda_*_restore` modules.

## Placement ABI

[`include/spark_cache_placement.h`](include/spark_cache_placement.h) defines a
fixed C ABI with:

- one destination descriptor per registered tensor;
- one descriptor per verified encoded object;
- one uploaded physical-slot vector per restore;
- two reusable bounded arenas;
- direct-object and transposed-slab submission modes; and
- begin, acquire, submit, finish, and abort transaction states.

Destination descriptors contain runtime pointers, capacities, strides, record
kinds, byte widths, and source ordinals. Persistent metadata never contains
runtime pointers or physical slot coordinates.

[`src/spark_cache_placement_parse.cpp`](src/spark_cache_placement_parse.cpp)
parses canonical SparkCache objects in place. It validates the format version,
header shape, record offsets, record kinds, digest syntax, and logical position
ownership.

The caller must authenticate the complete encoded object before parsing it.
That outer digest is the integrity boundary; per-record rehashing would reread
the same bytes without strengthening the manifest contract.

## Arena modes

| Mode | Behavior | Intended use |
|---|---|---|
| Mapped host | CUDA reads authenticated host pages directly | Preferred on compatible unified-memory systems |
| Managed | CUDA-managed arena | Deployment-specific comparison |
| Staged device | Pinned host input plus device staging | Conservative fallback |

Supported arena sizes and modes are ABI capabilities. Deployment profiles must
choose values that fit their object geometry and memory budget.

To preserve aligned copies, read the object prefix first, place the complete
object at an aligned arena offset, then hash exactly the encoded bytes. Padding
is never part of the object digest.

## Memory ordering

These rules are mandatory:

1. An arena moves `FREE -> FILLING -> VERIFIED -> IN_FLIGHT -> FREE`.
2. Reading, hashing, parsing, and position validation finish before submission.
3. Cross-thread producers publish readiness with release/acquire ordering.
4. The CPU never writes an `IN_FLIGHT` mapped arena.
5. The slot vector stays immutable until finish or abort.
6. Direct slabs cover slot rows exactly once and in order.
7. Concurrent streams may write only disjoint rows or destinations.
8. The parked request resumes only after all completion events and device error
   state validate successfully.

A later slab can fail after earlier slabs wrote request-owned blocks. Those
blocks remain private to the parked request and are freed before recomputation;
no forward pass may consume them.

## Snapshot gather ring

`libspark_cache_snapshot.so` gathers completed rows into bounded staging arenas
for optional write-behind publication. Its ownership and cancellation rules are
specified in
[`SNAPSHOT_RING_STATE_MODEL.md`](SNAPSHOT_RING_STATE_MODEL.md).

The snapshot library uses a separate best-effort ABI. Backpressure or failure
may cancel publication but cannot delay serving or expose a partial manifest.

## Build and test

GPU-free ABI, layout, and reference tests:

```bash
python -m pytest -q sparkcache/native/tests
```

CUDA build in the same toolchain used by the serving runtime:

```bash
cmake -S sparkcache/native -B build/native \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES='<target-architecture>' \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=ON
cmake --build build/native -j"$(nproc)"
ctest --test-dir build/native --output-on-failure
```

The build artifact must contain the requested architecture and pass the
standalone byte-comparison probe before model integration. A successful build
or synthetic probe does not qualify live serving.

## Qualification boundary

A deployment profile may enable SparkCache CUDA placement only after proving:

- exact library and source identities;
- byte-identical output against the Python reference path;
- byte-identical continued generation;
- correct remapping to a different physical block allocation;
- corruption, cancellation, restart, and allocation-pressure behavior;
- bounded host and device memory; and
- no serving-wide stall behind a parked restore.

Exact models, checkpoints, topologies, measurements, and launch commands belong
in deployment profiles and evidence records under the repository root.
