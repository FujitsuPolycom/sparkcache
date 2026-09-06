# SparkCache buddy replication

**Status: research-only.** This package defines a GPU-free protocol and bounded
state machines for copying SparkCache objects between rank-local stores.

The package does not open sockets, bind network interfaces, change routes, or
touch CUDA. A deployment may carry the same transaction over TCP, RDMA, or
another reliable byte transport.

## Transaction

```text
BEGIN(transaction, generation, context, identity)
PUT_CHUNK(index, sha256, bytes) ...
COMMIT(sha256(commit-record), ordered chunk digests)

or:

ABORT(transaction, generation)
```

The receiver publishes a replica only after every ordered object and the
canonical commit record validate. Objects staged before `COMMIT` are invisible.

Repeated `BEGIN`, `PUT_CHUNK`, `COMMIT`, and `ABORT` frames are safe. A newer
generation supersedes an abandoned generation; stale traffic is rejected.

`ACK` names the exact input sequence it acknowledges. `CREDIT` reports absolute
free byte and frame capacity, so replaying a credit cannot enlarge the sender's
window.

Sequences belong to one transaction and generation. Retransmission reuses the
same encoded bytes and consumes no extra credit.

## Effect on serving

Replication is optional repair work. If the receiver, sender window, remote
validation, or publication callback cannot proceed, `BuddySender` keeps the
local commit and marks that transaction local-only.

The sender does not wait for capacity. It queues one best-effort `ABORT` frame
to release remote staging without interrupting inference.

The receiver bounds active transactions, staged bytes, chunk descriptors, and
remembered receipts. An integrity or capacity failure discards only the
incomplete remote transaction and returns its credits.

A carrier must run transaction expiry even when no frames arrive. After a
connection loss, it reconnects with a different generation.

In durable-publication mode, `on_chunk` publishes each content-addressed object
as it arrives. The receiver keeps only its digest and index until `COMMIT`, so a
large publication need not accumulate in memory.

## Test

```bash
python -m pytest sparkcache/replication -q
```

The tests cover fragmented frames, canonical headers, size bounds,
manifest-last visibility, SHA-256 checks, duplicate delivery, stale
generations, aborts, credits, retransmission, and local-only fallback.
