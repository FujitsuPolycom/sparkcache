# Native placement and snapshot libraries

`libspark_cache_placement.so` implements checksum-gated direct restore into
registered cache tensors. It is **implemented** and opt-in; qualified
DeepSeek profiles keep it disabled. CPU/layout tests, CUDA 13 SM121
compilation, standalone mapped-host placement, ctypes ABI validation, and a
checksum-bound four-rank integration gate cover this library.

`libspark_cache_snapshot.so` implements the store-side gather ring documented
in [`SNAPSHOT_RING_PROTOTYPE.md`](SNAPSHOT_RING_PROTOTYPE.md). It is
**research-only** and uses a separate fail-open ABI so optional snapshot
publication cannot weaken fail-closed restore placement.

The native restore path removes Python object construction and host-side
record transposition from large restores.

## Why this path exists

The measured 392,960-token DCP4 rank-0 artifact contains:

- 1,535 complete chunks;
- 392,960 global tokens, or 98,240 DCP4-owned rows per rank;
- 3,142,449,596 encoded bytes on one rank;
- 79 target/CKV destinations at 368 bytes per local token;
- 22 sparse-indexer destinations at 132 bytes per local token.

The measured warm eight-thread read plus **one outer SHA-256 per encoded
chunk** is already only 205–216 ms per rank. Adding redundant per-record
hashes raises it to 363–375 ms. The measured 12–18 second requester wait is
therefore not an NVMe limit.

The per-record Python path decodes every chunk into immutable per-record
`bytes`, creates roughly 155,000 per-layer slices, joins each layer, uploads
the slot vector 101 times, performs 101 staging copies and launches 101
scatter operations. The vectorized Python fallback beside this directory
reduces assembly to about 0.4 seconds on the development CPU, but it still
transposes all 3.14 GB in host memory before placement.

The native direct path removes that entire middle representation:

```text
pread encoded .spcc files directly into cudaHostAllocMapped arena
  -> verify manifest outer SHA-256 once
  -> validate header, record offsets, and logical positions without copies
  -> one fused multi-destination scatter kernel for the arena
  -> final 101 registered cache tensors
```

No `ContextChunk`, record `bytes`, per-layer join, layer-wise H2D, or
layer-wise scatter exists on this path.

## Canonical Python/ctypes boundary

[`../spark_cache_native.py`](../spark_cache_native.py) is the single Python ABI
declaration. The `sparkcache.native.python` package re-exports that module; it
does not maintain a second copy. Do not independently declare these structures
in probes or connector modules. `load_library()` first calls
`spark_cache_placement_query_abi()` and fails before allocation if the ABI
version, arena/record constants, structure sizes, or required capabilities do
not match.

Model-serving restore orchestration uses the attested adapter in
[`../spark_context_cache_native_placement.py`](../spark_context_cache_native_placement.py):

```python
from sparkcache.spark_context_cache_native_placement import (
    ArenaMode,
    NativePlacementAdapter,
    NativePlacementLibrary,
)

library = NativePlacementLibrary.load(path, expected_sha256=digest)
adapter = NativePlacementAdapter.create(
    library,
    arena_mode=ArenaMode.MAPPED_HOST,
    arena_bytes=arena_bytes,
    max_destinations=destination_count,
    max_slots=slot_count,
    max_chunks_per_slab=chunk_count,
    device_ordinal=device_ordinal,
)
```

`execute_native_restore()` owns multi-file slab packing, complete-file digest
verification, arena lifetime, direct-slab submission, and parked-request
completion. `ParkedRestore.finish()` is the only success edge that permits the
requester to resume; any exception aborts the transaction and recomputes the
parked KV blocks.

The binding also exposes copied handle errors, a thread-local runtime error
for failures before `create()` returns a handle, and an in-flight statistics
snapshot. Python therefore gets complete diagnostics without retaining a
borrowed C string.

## Implemented interfaces

[`include/spark_cache_placement.h`](include/spark_cache_placement.h) defines a
fixed C ABI:

- one 32-byte destination descriptor per registered tensor;
- one 64-byte descriptor per verified encoded chunk;
- one uploaded `uint32` physical-slot vector per restore;
- two reusable 64, 128, or 256 MiB arenas;
- direct encoded-chunk and transposed-slab submission modes;
- fail-closed begin/acquire/submit/finish transaction states.

The destination descriptor stores a current runtime pointer, capacity,
stride, record kind, byte width, and source layer ordinal. No physical slot
coordinate is persisted.

[`src/spark_cache_placement_parse.cpp`](src/spark_cache_placement_parse.cpp)
strictly parses the canonical SparkCache v1 header in place. It validates:

- `SPCKV001` magic and ABI 1;
- exact canonical header shape;
- contiguous and bounded record offsets;
- required data record kinds;
- lowercase SHA-256 descriptor syntax;
- the complete logical-position payload against DCP rank ownership.

The parser is intentionally named
`spark_cache_parse_verified_v1_chunk`: the caller must first compare the
outer SHA-256 of the complete encoded span with the immutable manifest. That
single outer digest is the integrity boundary; hashing every record again
would re-read the same 3.14 GB.

[`src/spark_cache_placement.cu`](src/spark_cache_placement.cu) implements:

- `cudaHostAllocMapped` arenas, preferred for measuring the GB10 unified-memory
  architecture;
- `cudaMallocManaged` as a measured A/B;
- pinned host plus staged device arenas as a conservative fallback;
- two arenas and two nonblocking, lowest-priority CUDA streams so a cache
  restore yields to foreground model work where the device honors priority;
- one static destination-table upload;
- one unique, bounds-checked slot-vector upload per restore;
- one kernel launch per input arena;
- a byte-exact fused kernel covering every destination in that arena;
- a transposed-slab kernel for the existing NumPy placement fallback.

For the direct kernel, one CTA handles one encoded chunk and one destination.
Eight warps copy the chunk's 64 local DCP4 rows. The measured entry therefore
requires about 155,000 CTAs total, spread across bounded arena launches—not
one launch or Python dispatch per layer. Four-byte copies are used only when
both addresses and the width are aligned; the byte fallback preserves exact
semantics for every legal layout.

## Arena choices

| Arena | Mapped host footprint | Staged host+device footprint | Use |
|---:|---:|---:|---|
| 64 MiB | 128 MiB | 256 MiB | direct-path low-memory A/B |
| 128 MiB | 256 MiB | 512 MiB | required qualification point |
| 256 MiB | 512 MiB | 1 GiB | required maximum-throughput A/B |

The adjacent Python transposition fallback needs 30 slabs at 128 MiB or 14
slabs at 256 MiB for the 393,216-token inventory. The direct path is bounded
by encoded-chunk size instead of layer size and performs no host transpose.

To preserve the four-byte kernel path, read the 16-byte prefix/header first,
then choose padding before the encoded chunk so
`arena + encoded_offset + payload_offset` is at least four-byte aligned
(256-byte alignment is preferred). Re-read the complete file into that
location and hash exactly those `encoded_bytes`; padding is not part of the
digest.

## Memory ordering and visibility contract

These rules are not optional:

1. An arena moves `FREE -> FILLING -> VERIFIED -> IN_FLIGHT -> FREE`.
2. The producer must finish `pread`, outer SHA, parse and positions validation
   before calling `submit_*`.
3. Cross-thread producers publish readiness with release/acquire
   synchronization. The API adds a release fence immediately before launch.
4. The CPU must not write an `IN_FLIGHT` mapped arena. `acquire_arena()` waits
   on its CUDA completion event before returning the address again.
5. The slot vector is immutable until `finish_restore()` or
   `abort_restore()`.
6. Direct slabs must cover slot-vector rows exactly once, contiguously.
   Transposed slabs must cover every destination exactly once.
7. Two CUDA streams may overlap only because direct slabs write disjoint slot
   rows and transposed slabs write disjoint destination tensors.
8. The restored request remains parked until `finish_restore()` validates
   both stream events and the device error word.

Earlier slabs may already have written parked KV blocks when a later slab
fails. This does not weaken the existing fail-closed contract: those blocks
belong exclusively to the parked request, no forward pass may consume them,
and the failure path frees/recomputes them. Making the 3.14 GB installation
physically transactional would require another full-size device copy and
would defeat the purpose.

## Offline tests

Python ABI/source guards and zero-copy buffer tests:

```powershell
cd sparkcache
python -m pytest -q native/tests
```

CPU C++ parser/layout/reference gate from WSL:

```powershell
cd sparkcache\native
wsl.exe g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror -UNDEBUG `
  -Iinclude `
  src/spark_cache_placement_parse.cpp `
  tests/layout_test.cpp `
  -o /tmp/spark_cache_native_layout_test
wsl.exe /tmp/spark_cache_native_layout_test
```

The CPU test parses a real canonical v1 byte layout, validates DCP-owned
positions, scatters two target layers plus one indexer layer into scrambled
physical slots, and demands byte identity.

## Exact DGX Spark compile gate

Build inside the same CUDA/container environment as vLLM:

```bash
cd sparkcache/native
cmake -S . -B build \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=121 \
  -DSPARK_CACHE_PLACEMENT_ENABLE_CUDA=ON
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
./build/spark_cache_native_probe
python3 app/spark_cache_ctypes_probe.py \
  build/libspark_cache_placement.so
sha256sum build/libspark_cache_placement.so \
  build/spark_cache_native_probe
```

Qualification evidence from 2026-08-20: identical ARM64 source compiled every
library and probe target with `CMAKE_CUDA_ARCHITECTURES=121` in two Ubuntu
24.04 DGX Spark build containers, one using CUDA 13.0.88 and one using CUDA
13.2.86. Both containers passed
`spark_cache_placement_layout_test` and `spark_cache_snapshot_ring_test`, and
`cuobjdump --list-elf` reported an `sm_121` cubin in both
`libspark_cache_placement.so` and `libspark_cache_snapshot.so` under both
toolchains. The CUDA probes were compiled but not executed because the hosts
were serving another workload; this evidence qualifies build portability and
CPU/layout behavior only, not GPU execution or performance.

The standalone CUDA probe uses mapped pinned input, one slot upload, one
destination-table upload and one fused direct scatter. It verifies scrambled
destination rows byte-for-byte. This is a syntax/correctness gate, not a
performance claim.

The build is rejected if:

- SM121 is not present in the resulting fatbin;
- the probe reports nonzero staged H2D bytes in mapped mode;
- slot uploads, destination uploads, slabs or kernel launches differ from
  `1/1/1/1`;
- any CUDA, bounds or byte comparison error occurs.

## Exact 392,960-token model-integrated microbenchmark gate

The initial execution must use throwaway KV destinations rather than tensors
owned by a serving model.

1. Freeze the existing rank-local 392,960-token manifest and its 1,535 chunk
   digests. Do not rewrite the artifact.
2. Resolve the 101 registered tensor layouts, but allocate equal-sized
   throwaway CUDA destinations.
3. Derive the physical slots from a deliberately different block allocation.
   Upload the 98,240-entry vector once.
4. Build the static destination table once and attest its serialized SHA-256.
5. For each arena:
   - prefix-read enough bytes to determine `payload_offset`;
   - pad so the payload is 256-byte aligned;
   - parallel `pread` complete files directly into the arena;
   - compute and compare each complete encoded-file outer SHA;
   - call `spark_cache_parse_verified_v1_chunk` with
     `logical_start = chunk_index * 256`, DCP degree 4 and the local rank;
   - submit the verified descriptors.
6. Run the synchronous Python loader into separate throwaway tensors using the same
   artifact and the same remapped slots.
7. Synchronize and compare all bytes of all 101 destinations. Require zero
   mismatch on every rank.
8. Repeat mapped-host 128 MiB, mapped-host 256 MiB, managed 128/256 MiB and
   staged-device 128 MiB five times warm and twice cold.
9. Report per rank and slowest-rank:
   - pread plus outer SHA;
   - parse/position validation;
   - arena wait time;
   - GPU scatter event time;
   - placement wall time;
   - complete requester restore wall time;
   - staged H2D bytes and kernel launches;
   - decode throughput impact on single-request decode (C1) and
     eight-request aggregate decode (C8).

Admission gates:

- zero byte mismatch and byte-identical continued generation;
- zero duplicate/out-of-range slots and zero device errors;
- exactly one slot upload and one destination-table upload;
- mapped mode reports zero staged H2D bytes;
- no memory-growth trend across ten restores;
- concurrent serving never stalls behind the parked requester;
- warm read+verify remains at or below 300 ms p95;
- native placement is at or below 500 ms p95 on the slowest rank;
- complete warm restore is at or below 1.2 seconds p95, with a 1.5-second
  initial acceptance ceiling.

The sub-250-ms placement and sub-one-second complete restore are stretch
targets. They are physically plausible on UMA but are not claimed until the
all-rank probe measures them.

## Integration boundary

The model-serving connector selects this path only behind an explicit
environment flag and an attested library hash. The validation order is:

1. apply the published asynchronous-restore scheduler compatibility patch;
2. build and hash-attest this library;
3. keep the Python vectorized transposition plus native transposed
   scatter as fallback;
4. park the requester until `finish_restore()` succeeds on every rank;
5. rerun the 392,960-token equivalence, corruption, single-request decode,
   eight-request aggregate decode, cancellation, and restart gates;
6. only then make direct mapped placement the default.

The native direct-placement path shortens the restored requester's own wait.
The asynchronous scheduler contract keeps that requester-specific wait from
blocking unrelated requests.
