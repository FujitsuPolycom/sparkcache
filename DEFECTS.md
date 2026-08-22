# Known correctness and scaling defects

Each entry states a trigger, mechanism, and consequence. Severity
`high` means silent wrong behavior or a stuck request is reachable; `medium`
means degraded service or recoverable corruption of cache state. Entries are
removed when the fix and its regression test land; identifiers are stable and
never reused, so gaps mark fixed entries. Regression tests for removed
entries live in `sparkcache/test_defect_regressions.py`.
