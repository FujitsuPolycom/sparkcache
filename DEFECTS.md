# Known correctness and scaling defects

Each entry states a trigger, mechanism, and consequence. Severity
`high` means silent wrong behavior or a stuck request is reachable; `medium`
means degraded service or recoverable corruption of cache state. Entries are
removed when the fix and its regression test land; identifiers are stable and
never reused, so gaps mark fixed entries. Regression tests for removed
entries live in `sparkcache/test_defect_regressions.py`.

## D-6 (medium): quorum reports retransmit complete held sets

Each worker ships its complete sorted `_held` digest set through the stats
channel on every scheduler step. Per-process generation identifiers correctly
withdraw stale confirmations after restarts, but report size and scheduler
merge work still grow linearly with the number of retained manifests. Large
cache inventories therefore consume unbounded control-plane bandwidth.

**Removal criterion:** replace full-set reports with generation-scoped,
sequence-checked deltas and prove missed, duplicated, reordered, and
post-restart reports converge to the same quorum state.
