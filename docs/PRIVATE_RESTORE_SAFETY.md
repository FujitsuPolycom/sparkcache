# Whole-prefix restore ownership

Status: **implemented**, covered by GPU-free regression tests. Hardware
integration with the combined TP4 image remains unverified.

Whole-prefix placement writes the complete restored snapshot; its load plan
does not carry a suffix-only write mask. A request with already-computed local
tokens can reference pages shared with another request. The connector therefore
declines external restoration for that request and lets vLLM compute its suffix.
This can replace an external-cache offer with recomputation, but preserves the
local prefix and avoids writing through unproven page ownership.

If a restore already owns allocated blocks, repeated lookup waits for its
completion rather than offering another writer or allowing recomputation into
those blocks. An unallocated offer is retired if a local prefix appears.

Failed restores report only their nonzero block IDs. Block `0` is shared null
padding in vLLM, not evidence that another request's cache is damaged. The same
rule applies when shutdown rejects a queued load before execution.

These changes adapt the private-admission and failure-isolation safeguards from
SparkCache source `360f97dbd00b62b06fc5e7839b87514c12f2908f`. They do not import
that composition's vLLM patch, single-slot/managed capture options, or follower
scheduling policy. Cache identities, stored formats, and native ABIs are unchanged.

Tests: `sparkcache/test_restore_private_admission.py` and
`sparkcache/test_restore_failure_isolation.py`. The failure tests execute the
shipped HMA recovery method against failed and unrelated request block tables.
