# PR646 runtime source snapshot

Status: research-only. This branch publishes the deployable SparkCache source
used by the experimental ARM64 GLM-5.3 Flash TP2/DCP1 runtime, without private
site launch records or benchmark logs from its development history.

The deployable sparkcache/ tree has source_tree_sha256
`433a75f4f558aa7192eceba61dfae44e9b9823d5be714f054c3568efed88a0ce`.
It matches the deployed source tree at private development revision
`5b27132122e7c79fadbaa2aecfa0090d7a28f283`; that historical commit is not
published by this branch. Use this branch's public commit as the source pin,
and verify the content digest rather than assuming commit IDs are identical.

Behavior: bounded single-slot managed capture, commit-bound vLLM lease and
recurrent-publication hooks, corrected recurrent-state restore indexing,
failure-isolated restore, and full-prefix private-destination admission.
Restore paths that cannot prove ownership or validity miss and recompute.
This publication does not change CacheIdentity salts, wire values, or chunk
geometry relative to the deployed source, and does not create a new namespace.

Companion vLLM patch/preimages are under patches/vllm-pr646-815f839; they are
bound to upstream815f839060c2781f6bcc47c0d584358b400ea0ea. Newer upstream
successors require their own integration and qualification. This source
snapshot does not qualify DCP2 or every SparkRing deployment profile.

Validation: python -m pytest sparkcache -q:1049 passed,7 skipped on Windows
Python3.12. GPU-free tests validate contracts and behavior, not GPU execution.
The source digest was reproduced independently in the clean publication
worktree. Existing public deployment scaffolding is inherited from the public
base; only runtime code, regression tests and necessary patch support are added.
