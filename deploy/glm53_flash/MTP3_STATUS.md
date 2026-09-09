# GLM-5.3 MTP3 cache implementation and issue status

Status: **implemented**, with bounded live evidence. The four-Spark native-MTP3
deployment remains **research-only**. Open performance issues are not evidence
that their related implementation PRs are unmerged; merged PRs are not evidence
that every reported workload is fixed.

## Deployable composition

Use SparkRing's
[MTP3 cache/checkpoint quickstart](https://github.com/FujitsuPolycom/sparkring/blob/main/docs/GLM53_MTP3_CACHE_CHECKPOINTS_QUICKSTART.md).
It selects a compatible image, transport bundle, recurrent checkpoints,
native placement library, and cache namespace together. Installing the Python
package alone does not install the runtime's GPU prefix-retention fixes.

The
[published image contract](https://github.com/FujitsuPolycom/sparkring/blob/main/runtime/glm53-spark-mtp3-mesh/performance/public-image.json)
pins SparkCache `48bbd2be4a7b972e56632a2d7b934bac5460f272` and includes
[SparkCache #62](https://github.com/FujitsuPolycom/sparkcache/pull/62) and
[#63](https://github.com/FujitsuPolycom/sparkcache/pull/63).
[SparkRing #236](https://github.com/FujitsuPolycom/sparkring/pull/236) integrates
the compute, transport, and cache-runtime changes from SparkRing #219, #226,
and #227. Those component PR numbers are not separate installation steps.

[SparkRing #237](https://github.com/FujitsuPolycom/sparkring/pull/237) implements
admission control that rejects public inference until readiness warmup completes.
That source change is merged but is **not included** in the published image
identified above. It requires a rebuilt image and startup validation.

## What remains in the performance issues

Source builds include request-attribution accounting, same-pass maintenance
inventory reuse, and optional deletion pacing. The immutable image identified
above does not contain these additions. Attribution also requires the matching
instrumented SparkRing scheduler; a package update alone cannot supply its events.

| Issue | Implemented | Remaining work |
|---|---|---|
| [#60: sustained publication and eviction slowdown](https://github.com/FujitsuPolycom/sparkcache/issues/60) | Reduced restore work, publication-dependency protection, backlog gauges, capacity guidance, same-pass inventory reuse, and opt-in deletion-attempt limits and cooldown. | Full inventory scans and individual filesystem operations still have no hard time bound. Validate the combined implementation against near-full 40 GiB traffic and matched before/after probes. |
| [#61: growing conversations lose local prefix reuse](https://github.com/FujitsuPolycom/sparkcache/issues/61) | Runtime lease accounting, local-prefix preference, checkpoint retention, and opt-in request attribution for accepted target execution. | Deploy the matching scheduler instrumentation and validate attribution and local retention on the reported long-context conversations with occasional images. |

One saver admission per rank bounds concurrent optional work; it does not bound
the duration of an inventory scan. Deletion pacing is disabled by default.
Smaller deletion budgets can repeat scans and increase total work. See
[capacity and cleanup](../../sparkcache/README.md#capacity-and-cleanup).
Backlog reports distinguish pending publications, their oldest age, and active
maintenance, but reports may not refresh while the connector is idle.

Matching traces distinguish restore offers, worker completion, and GPU lease
attachment. Their scheduler prefix-token input and API `cached_tokens` alone
cannot establish exact attribution.

The [request-attribution record](../../sparkcache/README.md#request-reuse-attribution)
uses authoritative scheduler events to distinguish consumed local prefixes,
verified external reuse, and accepted prompt computation. Counts accumulate
across preemption attempts; incomplete observations are labeled explicitly.

The [maintenance inventory benchmark](../../docs/maintenance-inventory-validation.md)
measures reduced metadata work on CPU fixtures. It is not a DGX4 serving result
or confirmation that the original slowdown is resolved.

Both issues should remain open until their remaining implementation questions
and workload-specific results are recorded explicitly.

## Evidence and its limits

The
[cache-pressure validation record](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/glm53-flash/mtp3-cache-history-validation.md)
contains 551 successful responses on a related composition. It used text-only
traffic, a 2 GiB cache per rank, and opt-in periodic full captures. It is useful
regression evidence, but it does not reproduce the original 40 GiB workload.

The published image's
[source-equivalence record](https://github.com/FujitsuPolycom/sparkring/blob/main/performance/records/glm53-flash/mtp3-integrated-image-source-equivalence.md)
records 5,308 identical runtime files and excludes SparkCache source from that
equality claim. File equality does not constitute a serving soak.

The [exact-image serving record](https://github.com/FujitsuPolycom/sparkring/blob/86d2ebdff794da3d89f3ba1c6aca649ff6b052a0/performance/records/glm53-flash/mtp3-cache-checkpoints-serving-smoke-20260906.md)
qualifies eight short correctness checks, including a growing conversation,
streaming, reasoning-mode rejection, and repeated red/blue/red image responses.
All ranks published a 4,096-token context and were healthy afterward. The
40 GiB cache limit was configured, not filled; these checks do not establish
near-capacity performance or multimodal persistent restoration.

The published profile defaults to 40 GiB per rank, a 32 GiB low watermark, and
periodic full captures disabled. Setting a capacity is not a near-capacity test.
Keep the successful small-cache evidence; use short semantic and multimodal
checks after deployment, then collect near-capacity and local-reuse evidence
during representative operation before closing #60 or #61.

For an affected workload, `SPARKCACHE_ACCESS_MODE=restore-only` with
`SPARKCACHE_ASYNC_PAGE_CAPTURE=0` is a diagnostic workaround documented in the
issue reports. It stops persistence of newly encountered contexts. It is not
required merely because the issues remain open.
