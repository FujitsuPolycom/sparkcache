# C++/CUDA streaming-snapshot ring state model

Status: **research-only**. CPU contract and state-machine checks pass. The CUDA
source has compiled for the recorded toolchains and passed the CPU/layout tests
described in [`README.md`](README.md).

The evidence records hold exact compiler and architecture details. No GPU test
or performance result supports enabling this ring in a serving profile.

## Purpose

The ring is designed to replace an end-of-prefill Python snapshot with bounded
write-behind work that:

- gathers selected physical KV rows incrementally;
- preserves the model producer stream's ordering;
- tells the scheduler exactly when it may release KV-block leases;
- hands immutable mapped/managed bytes to a background writer;
- never blocks inference when cache publication falls behind; and
- safely abandons a partially published context?

The state machine and ABI implement these rules. GPU execution and interference
with model serving remain untested.

## Ownership pipeline

```text
vLLM owns KV blocks
        |
        | submit(logical chunk, physical slots, producer CUDA stream)
        v
GPU_FILLING -- completion event --> READY -- claim --> WRITING
     |                                |                 |
     | abandon: drain event           | abandon: free   | abandon: mark/drop
     v                                v                 v
    FREE <------------------------- release ---------- FREE
```

The producer call is nonblocking. If both or all three slots are occupied, it
returns `WOULD_BLOCK`; the caller continues inference and drops that cache
chunk. There is no wait-for-space operation.

`poll(ticket) == OK` is the block-lease boundary: the gather kernel has
finished reading the submitted physical KV rows, so vLLM may reuse them.
The arena bytes remain owned by SparkCache until `release(ticket)`.

Generation-tagged tickets reject use after release. Abandoning a
context:

- frees unclaimed `READY` slots immediately;
- lets `GPU_FILLING` slots reach their CUDA event before reuse; and
- preserves a claimed writer's bytes until that writer releases them.

Callers do not retain abandoned tickets merely to drain them. Later submit and
poll calls query abandoned `GPU_FILLING` events and reap completed work.

CUDA-incomplete slots remain quarantined. A saturated ring returns
`WOULD_BLOCK` instead of waiting or reusing bytes that CUDA may still read.

### Post-launch failure quarantine

After the gather kernel launch executes, a launch-status or event-record error
cannot return the slot directly to `FREE`.

The completion event may be absent or stale while the gather still reads the
physical-slot table, source table, and destination arena.

The C++/CUDA failure path therefore:

1. marks the context discarded while leaving the affected slot
   `GPU_FILLING`;
2. records the exact producer stream in the slot and marks
   `requires_stream_drain`;
3. synchronizes that producer stream on the exceptional path;
4. calls `reap_discarded` only after synchronization succeeds; and
5. returns the original CUDA failure without publishing a ticket.

If stream synchronization fails, the slot remains quarantined. Later submit,
abandon, and shutdown calls retry the drain and reject teardown until completion
can be proved.

`shutdown_complete` stays false, `destroy()` retains C++/CUDA memory, and the
slot generation cannot be reused.

An unrecoverable CUDA failure may therefore retain the handle. It cannot free
memory that GPU work might still reference.

Two build-only regression seams exist and default off:

```text
SPARK_CACHE_SNAPSHOT_TEST_FORCE_EVENT_RECORD_FAILURE
SPARK_CACHE_SNAPSHOT_TEST_FORCE_STREAM_DRAIN_FAILURE
```

The first proves a post-launch event failure drains before reuse. Enabling
both proves a failed drain retains quarantine across reuse and shutdown.

## Payload layout

The model registers one `SparkCacheSnapshotSource` per target CKV, sparse
indexer, MTP draft KV, or boundary-hidden tensor. Each record kind uses dense
layer ordinals and one row width.

For every submitted local chunk, the C++/CUDA code creates a 64-byte-aligned,
layer-major raw payload:

```text
target layer 0 rows
target layer 1 rows
...
[64-byte alignment]
indexer layer 0 rows
...
[64-byte alignment]
MTP layer 0 rows
...
```

The writer gets offsets and lengths in `SparkCacheSnapshotReadyView`. It may
add canonical headers and positions, persist the span locally, or pass it to a
replication carrier. Physical slot IDs never enter the stored artifact.

## ABI decisions

- Snapshot ABI is version 1 and separate from restore placement ABI 1.
- Ring depth is exactly two or three slots.
- Supported arenas are `cudaHostAllocMapped` and `cudaMallocManaged`.
- The caller supplies the producer CUDA stream as an integer handle.
- A gather and completion event are enqueued on that same stream.
- Source tables are static for the loaded model and uploaded once.
- Physical slots are copied into a per-slot mapped control array.
- The Python loader opens and hashes one regular-file descriptor. On Linux it
  calls `dlopen("/proc/self/fd/N")` for that inode.
- The loader verifies device, inode, size, and modification time before and
  after loading. It retains the descriptor for the CDLL object's lifetime.
- Replacing the original pathname cannot substitute different bytes between
  hashing and loading.
- Platforms without Linux `/proc/self/fd` are unsupported. There is deliberately
  no weaker hash-path-then-`dlopen(path)` fallback.
- After loading the exact inode, the binding checks every ABI constant,
  structure size, and required capability.
- `WOULD_BLOCK`, `NOT_READY`, and `DROPPED` are ordinary opportunistic-cache
  outcomes. They are not model failures.
- `shutdown()` refuses while any writer owns a `WRITING` view. After all
  writers release, shutdown abandons unclaimed data, synchronizes outstanding
  GPU gathers, and leaves every slot `FREE` before destruction.

The snapshot library is separate (`libspark_cache_snapshot.so`) so its ABI and
artifact identity can change without changing the restore-placement library.

## Offline checks

From `sparkcache`:

```powershell
python -m pytest -q native/tests
python -m ruff check native/python/spark_cache_snapshot_native.py `
  native/python/snapshot_ring_state_model.py `
  native/app/spark_cache_snapshot_ring_lab.py `
  native/tests/test_snapshot_ring_contract.py
```

CPU C++ state/layout check from WSL:

```powershell
cd native
wsl.exe bash -lc "g++ -std=c++17 -Wall -Wextra -Wpedantic -Werror \
  -UNDEBUG -Iinclude src/spark_cache_snapshot_layout.cpp \
  tests/snapshot_ring_test.cpp -o /tmp/snapshot_ring_test &&
  /tmp/snapshot_ring_test"
```

Run the interactive state-model lab:

```powershell
python native/app/spark_cache_snapshot_ring_lab.py
```

Useful sequence: submit three contexts, submit a fourth to observe
`WOULD_BLOCK`, complete/claim one, abandon its context, then release it.

## Before live use

1. Record the built library SHA-256 and retain its CUDA 13/SM121 build inputs.
2. Execute the throwaway-destination probe in mapped and managed modes.
3. Compare gathered bytes against the canonical Python snapshot for target CKV,
   indexer, and MTP records at scrambled physical slots.
4. Verify the real vLLM stream handle and scheduler block-lease boundary.
5. Decide whether CUDA graph capture requires an event/host-node adapter.
6. Measure 2-slot versus 3-slot, 16/32/64 MiB, mapped versus managed.
7. Run cancellation, writer-crash, stale-ticket, single-request decode,
   concurrent decode, and all-rank agreement checks.
8. Keep the implemented connector/runtime/publisher integration default-off
   until byte identity, GPU execution, and sustained model-serving checks pass.

The first GPU-execution check must target throwaway buffers, not active model
KV state.
