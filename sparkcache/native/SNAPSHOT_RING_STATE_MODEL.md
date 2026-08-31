# C++/CUDA streaming-snapshot ring state model

Status: **research-only**. CPU contract and state-machine checks pass. The CUDA
source compiled for SM121 with CUDA 13.0.88 and 13.2.86, and both builds passed
the CPU/layout tests described in [`README.md`](README.md). The CUDA probes
have not run on a GPU, this document makes no performance claim, and the
snapshot ring is not enabled in a model-serving connector.

## Research objective

Can SparkCache replace the multi-second end-of-prefill Python snapshot with a
bounded write-behind ring that:

- gathers selected physical KV rows incrementally;
- preserves the model producer stream's ordering;
- tells the scheduler exactly when it may release KV-block leases;
- hands immutable mapped/managed bytes to a background writer;
- never blocks inference when cache publication falls behind; and
- safely abandons a partially published context?

The implemented contract satisfies these requirements at the state-machine
and ABI levels. GPU execution correctness and model-serving interference are
unsupported until the qualification requirements below pass.

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
finished reading the current physical KV rows, so vLLM may reuse those rows.
The arena bytes remain owned by SparkCache until `release(ticket)`.

Generation-tagged tickets reject use after release. Abandoning a
context:

- frees unclaimed `READY` slots immediately;
- lets `GPU_FILLING` slots reach their CUDA event before reuse; and
- preserves a claimed writer's bytes until that writer releases them.

Callers do not retain abandoned tickets merely to drain them. Every later
submit and poll nonblockingly queries abandoned `GPU_FILLING` events and
reaps only those that have completed. CUDA-incomplete slots remain
quarantined, so saturation still returns `WOULD_BLOCK` rather than waiting or
reusing live bytes.

### Post-launch failure quarantine

Once the gather kernel launch expression has executed, neither a launch-status
error nor a completion-event-record error permits the slot to return directly
to `FREE`. The completion event may be absent or stale while the gather still
reads the physical-slot table, source table, and destination arena.

The C++/CUDA failure path therefore:

1. marks the context discarded while leaving the affected slot
   `GPU_FILLING`;
2. records the exact producer stream in the slot and marks
   `requires_stream_drain`;
3. synchronizes that producer stream on the exceptional path;
4. calls `reap_discarded` only after synchronization succeeds; and
5. returns the original CUDA failure without publishing a ticket.

If stream synchronization fails, the slot remains quarantined. Later submit,
abandon, and shutdown calls retry the drain and reject teardown while it remains
unproven. `shutdown_complete` is never set, `destroy()` does not release C++/CUDA
memory, and the slot generation cannot be reused. This may deliberately leak
the handle after an unrecoverable CUDA failure, but it cannot free memory still
referenced by GPU work.

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
then add canonical SPCC headers/positions, hash and persist locally, or send
the same immutable span on a diagonal 10GbE buddy link. Physical slot IDs
never enter the persistent artifact.

## ABI decisions

- Snapshot ABI is version 1 and separate from restore placement ABI 1.
- Ring depth is exactly two or three slots.
- Supported arenas are `cudaHostAllocMapped` and `cudaMallocManaged`.
- The caller supplies the producer CUDA stream as an integer handle.
- A gather and completion event are enqueued on that same stream.
- Source tables are static for the loaded model and uploaded once.
- Physical slots are copied into a per-slot mapped control array.
- The Python loader opens and hashes one regular-file descriptor, then on
  Linux calls `dlopen("/proc/self/fd/N")` for that same inode. It verifies the
  device/inode/size/mtime identity before and after loading and retains
  the descriptor for the CDLL object's lifetime. Atomic replacement of the
  original pathname therefore cannot substitute different bytes between hash
  and load.
- Platforms without Linux `/proc/self/fd` are unsupported. There is deliberately
  no weaker hash-path-then-`dlopen(path)` fallback.
- After loading the exact inode, the binding checks every ABI constant,
  structure size, and required capability.
- `WOULD_BLOCK`, `NOT_READY`, and `DROPPED` are ordinary opportunistic-cache
  outcomes. They are not model failures.
- `shutdown()` refuses while any writer owns a `WRITING` view. After all
  writers release, shutdown abandons unclaimed data, synchronizes outstanding
  GPU gathers, and leaves every slot `FREE` before destruction.

The snapshot library is intentionally separate (`libspark_cache_snapshot.so`) so
changing its checksum does not invalidate the proven restore-placement
library.

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

## Qualification requirements

1. Record the built library SHA-256 and retain its CUDA 13/SM121 build inputs.
2. Execute the throwaway-destination probe in mapped and managed modes.
3. Compare gathered bytes against the canonical Python snapshot for target CKV,
   indexer, and MTP records at scrambled physical slots.
4. Verify the real vLLM stream handle and scheduler block-lease boundary.
5. Decide whether CUDA graph capture requires an event/host-node adapter.
6. Measure 2-slot versus 3-slot, 16/32/64 MiB, mapped versus managed.
7. Run cancellation, writer-crash, stale-ticket, single-request decode (C1),
   eight-request aggregate decode (C8), and four-rank quorum checks.
8. Keep the implemented connector/runtime/publisher integration default-off
   until byte identity, GPU execution, and model-serving soak checks pass.

The first GPU-execution check must target throwaway buffers, not active model
KV state.
