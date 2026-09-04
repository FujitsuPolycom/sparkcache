# Heat-aware admission and SSD write control: design specification

Status labels follow `CONTRIBUTING.md`. This document is a present-state
specification, not a history.

| Component | Status |
|---|---|
| 8-bit bounded heat-hit ring (counters, saturation, epoch decay) | implemented (research prototype, `research/heat_ssd_control/`) |
| Verified-restore recomputation-token calculation | implemented (research prototype) |
| Chunk reference ledger, shared-trunk accounting, marginal-byte cost | implemented (research prototype) |
| W-TinyLFU-style shadow cache for admission experiments | implemented (research prototype) |
| Hourly and daily staged-write window reporting | implemented (research prototype) |
| Staged-write versus unique-object byte accounting | implemented (research prototype) |
| SMART/Health log-page parsing and Data Units Written deltas | implemented (research prototype) |
| GPU-free behavior and production-import isolation tests | implemented (`research/heat_ssd_control/test_prototype.py`) |
| Per-rank publication-byte telemetry from the serving connector | unsupported |
| Wiring heat metadata into the vLLM connector, `ManifestStore`, or any serving path | unsupported |
| Publication or eviction decisions that consume heat metadata | research-only |
| Budget enforcement (rejecting or delaying a publication when over budget) | unsupported |
| Physical (NAND-media) write-amplification attribution | unsupported |
| Acquisition of the SMART/Health log page on a specific platform | unsupported (prototype consumes 512 raw bytes; it does not issue ioctls) |

This is a research design. No serving behavior described here as
research-only may be assumed by deployment tooling. The implemented interface
of SparkCache remains as specified in `README.md`; in particular `README.md`
(`Operations` section) continues to state, correctly, that hourly write
budgets, daily write budgets, and a physical-write-amplification estimate are
**unsupported** in the deployed connector.

## 1. Production context

SparkCache persists rank-local KV context to local NVMe through
`sparkcache.persistent_context_cache.cache_manifest.ManifestStore`:

- Chunks are immutable, content-addressed files (`chunks/<sha256>.spcc`,
  magic `SPCKV001`, `FORMAT_ABI = 1`). Each chunk carries up to
  `CacheIdentity.chunk_tokens` tokens (default 256) split into record
  families such as `target_ckv`, `sparse_indexer`, and `mtp_draft_kv`.
- Exact manifests (`manifests/<storage_key>/<context_digest>.json`) are the
  only visibility edge; prefix alias files (`prefix-aliases/`) and descriptor
  segments (`prefix-index/`) reference existing chunks without copying them.
  Alias publication is bounded to 64 aliases per exact root with descriptor
  segments of at most 16 chunk descriptors.
- Publication is durable and write-visible: `_publish_immutable_batch` writes
  each object's complete bytes to a `.writing-<uuid>` temporary file, fsyncs
  the file data, hard-links the temporary into place, and issues one
  directory fsync per batch. Identical content at the destination is
  verified and **not** overwritten, but the temporary staging bytes are
  still written first, so re-committing an unchanged chunk costs its full
  encoded size of host writes while retaining zero additional bytes.
- Capacity is governed by `CapacityPolicy` (high watermark
  `spark_cache_max_bytes`, low watermark defaulting to 90% of the high
  watermark, optional TTL seconds). `ManifestStore.maintain` evicts whole
  roots in manifest-recency LRU order and counts shared chunks and alias
  segments once. It has no knowledge of request frequency beyond the
  best-effort recency touch, which restores may apply at most once per 60
  seconds (`ManifestStore.touch`, `minimum_interval_seconds=60`).
- Scheduler- and worker-side counters in
  `sparkcache.spark_context_cache_connector` (for example `restore_hit`,
  `load_verified`, `store_skipped_present`, `store_skipped_quorum`,
  `restore_skip_backlog`) already observe part of what a heat-aware policy
  would need, but they are exported as `SparkCacheStats` only; no retention
  or admission decision consumes them.

Consequences that motivate this design:

1. The default `snapshot-v1` publication schema republishes a complete
   snapshot for a grown conversation. The opt-in `tail-cow-v1` and
   `page-tail-cow-v1` identities publish immutable row tails or page deltas
   instead. All three paths can stage an object before discovering that its
   content-addressed destination already exists, so retained bytes alone do
   not describe SSD traffic.
2. There is no frequency signal: a context restored once per second and a
   context restored once a day are equally subject to recency-LRU eviction
   if their manifests age similarly.
3. The serving connector has no publication-byte ledger or write budget. It
   does not correlate publication activity with host-observed NVMe writes.

## 2. Goals, non-goals, and the heat-isolation contract

Goals:

- A bounded, cheap, self-decaying heat signal per stored context.
- A defensible measurement of what heat-aware admission would retain or
  reject, produced by a shadow instance that cannot affect serving.
- Accounting that separates logical retained bytes from host-observed
  written bytes and from staged (write-path) bytes.
- An operator-facing method to observe SSD wear contribution through the
  standard NVMe Data Units Written counter.

Non-goals:

- Any change to restore correctness, hit authentication, or verification.
- Any deployment configuration option: no
  `--kv-transfer-config` key or environment variable is defined by this
  design.

Heat-isolation contract (binding on any future integration):

1. Heat metadata is diagnostic-only state that lives outside every
   authenticated storage surface. It must never appear in manifests, chunks,
   alias or segment payloads, `CacheIdentity`, any digest, or any schema
   validated by `_validate_manifest_metadata`.
2. Heat values may influence only *storage admission* (whether a candidate
   publication proceeds, and which roots maintenance removes). Heat values
   must never influence whether `ManifestStore.lookup` reports a hit, whether
   restore accepts bytes, or how `restore()` verifies content. A damaged but
   hot entry must be rejected and recomputed exactly like a cold one.
3. Serving must never wait on heat work: evaluation, decay, and budget
   computation are off the serving path by construction and must remain so.
4. Losing all heat metadata must be benign. Process restarts, device
   replacement, and clear-once (`spark_cache_clear_once`) may destroy heat
   state wholesale; correctness is unaffected, and the only consequence is
   the same unobserved posture as an empty heat model.

## 3. Bounded 8-bit hit counters

### Semantics

Each stored context is represented by a slot in a fixed-size ring of
unsigned 8-bit saturating counters. The ring is process-local scratch state;
it is never written to disk, never synced, and never shared across ranks.

- Key: `HeatKey(storage_key, context_digest)` — both fields are 64-character
  lowercase SHA-256 hex digests, matching `EntryKey` semantics for exact
  manifests (`EntryKey.root_kind = "manifest"`). Prefix-alias entries use
  the same pair because an alias hit resolves through the source manifest's
  identity; alias-specific hit accounting is not modeled.
- Slot index: little-endian 64-bit BLAKE2b digest of
  `<storage_key>` `0x0A` `<context_digest>`, taken modulo the ring capacity.
  Capacity must be a power of two, so the modulo is byte masking: with
  `digest_size = 8`, take the low `log2(capacity)` bits.
- Increment trigger: one increment per **completed verified restore** of the
  pair — the worker counted `load_verified` for that digest — and, in
  admission experiments, one increment per accepted publication candidate.
  Scheduler-probe manifest matches (`restore_hit` before verification) must
  not increment production-consumable counters in any integrated form;
  unverified probe counts are advisory.
- Saturation: increments clamp at 255; no wrap.
- Decay: every `decay_window` total increments (default 8192), every nonzero
  counter is shifted right by `decay_shift` bits (default 1) in one pass and
  the in-flight increment applies after the sweep. This bounds per-key
  overestimation from hash collisions to the collision rate and bounds the
  memory to one byte per slot.

### Sized example

A ring of 131,072 slots is 128 KiB of process memory; every slot has 256
representable values. With `decay_window = 8192`, a key restored once per decay
window settles at a steady estimate of about 1; a key restored ten times per
window settles near 10 with 8-bit precision until saturation. Keys that
collide in one slot add their estimates (false sharing); the design accepts
this because admission decisions are comparative, not absolute.

### Rejection and bounded behavior

- Malformed keys (digest shape), non-power-of-two capacity, or a decay
  window below 1 raise `ResearchFormatError` at construction or use.
- The ring has no I/O and no locks: a single-process caller is assumed.
- Any byte-level snapshot round trip is schema-checked; a mismatch raises
  `ResearchFormatError` rather than guessing.
- Counters are exact-lossy by design. There is no transition in which a
  counter drives a correctness decision, because per the isolation contract
  no correctness decision ever reads it.

### Schemas

`HitRing.snapshot()` emits JSON with schema
`sparkcache-research-heat-ring/v1`:

```json
{
  "schema": "sparkcache-research-heat-ring/v1",
  "capacity": 131072,
  "decay_window": 8192,
  "decay_shift": 1,
  "increments_since_decay": 4112,
  "counts_hex": "<262144 hex characters; one byte per slot, index 0 first>"
}
```

`HitRing.from_json(payload)` accepts only this schema and exact-length count
arrays. A snapshot preserves counters but not key identities; slot lookup
after a reload is still deterministic, so estimates for known keys survive
restart. An unobserved key can inherit a nonzero estimate when it collides
with an occupied slot, which is the sketch's stated false-sharing tradeoff.

## 4. Recomputation tokens avoided

Every verified restore replaces prefill work over the restored span. For a
restored manifest with `committed_tokens = s` accepted at
`num_computed_tokens = c` already-scheduled prefix tokens, the avoided
recomputation for that request is:

```
recompute_tokens_avoided = s - c
```

`recomputation_tokens_avoided(s, c)` implements this calculation and rejects
negative, non-integral, or reversed spans.

Over a wall-clock window `W`, for restores `i` of a context:

```
tokens_avoided(W, context) = sum_i (s_i - c_i)
```

When `c_i` is unavailable for a historical trace, the design approximates
`c_i = 0` and labels the result `tokens_avoided_upper_bound` — the
approximation overcounts only by prefix overlap with concurrently scheduled
work, which the production path reports when present.

A storage value comparison per context over a retention horizon `T`:

```
value(context) = tokens_avoided(T, context)              # benefit side
cost(context)  = committed_tokens                         # publication tokens
                 x rank_shard_factor
                 x republish_factor(context growth in T)
```

- `rank_shard_factor` accounts for per-rank sharding: under DCP degree `d`,
  each rank stores its shard of the span, so the fleet writes approximately
  `d` shards' worth of the same logical context; a per-rank record counts
  only its shard.
- `republish_factor` is 1 for a stable context and grows with the number of
  distinct publications of the same conversation. `snapshot-v1` submits a
  complete snapshot; `tail-cow-v1` and `page-tail-cow-v1` submit only the
  immutable tail or page-semantic delta plus authenticated metadata.

The prototype does not compute `value` itself. It provides the per-key
counter that any trace-summation script combines with actual restore timing
records (`sparkcache-restore-timing/v1`, which record the selected span) to
derive it. Deriving it automatically inside the connector is research-only.

Status: per-request formula is defined; automatic in-connector attribution
**research-only**; any admission decision consuming it **research-only**.

## 5. Shared-trunk value

A **trunk** is a chunk-aligned token prefix shared by at least two stored
contexts under the same `storage_key`. Chunks are content-addressed, so two
contexts share trunk bytes exactly when the same chunk digests appear in
both manifests (or in an alias's descriptor chain rooted at the same
manifest). The prefix-alias machinery already produces this shape: one exact
manifest plus up to 64 alias roots whose descriptor chains reference the
identical chunk objects.

For a stored context `C` with chunk list `chunks(C)` (ordered by logical
range) and the ledger's reference count `refs(d)` for chunk digest `d`:

```
shared_chunks(C) = { d in chunks(C) : refs(d) >= 2 }
trunk_tokens(C)  = sum(token_count(d) for d in shared_chunks(C))
```

- The count of shared chunks equals the shared prefix length in the actual
  publication pattern (contexts only append), but the ledger does not
  enforce consecutiveness; it reports the reference-count fact and leaves
  prefix interpretation to the caller.
- The value of the trunk to admission/eviction pressure: a context whose
  chunks are widely referenced contributes low *marginal* bytes (section 6)
  and high *avoided tokens per retained byte* — evicting it costs
  `trunk_tokens` of recomputation for every referencing root, while evicting
  a fully exclusive context costs one root's span. Manifest-recency LRU has
  no such notion; `ROADMAP.md` ("Trunk-aware eviction") describes the
  same gap and names alias reference counts as the prerequisite. This design
  supplies those counts as the ledger.

The prototype models all of this in `ChunkLedger`: it records exact per-chunk
token spans and byte sizes, decrements references on removal, and reports
`ContextHeatReport(shared_chunk_count, shared_tokens, marginal_bytes,
retained_shared_bytes, chunk_count)` per context.

## 6. Exclusive physical-byte cost

The **marginal byte cost** of a stored root is the number of filesystem
bytes that would be reclaimed if that root were deleted and the orphan
collector ran — the quantity `MaintenanceReport.bytes_reclaimed` measures
after the fact and this design predicts before an admission decision:

```
marginal_bytes(C) = sum(|b_d| : d in chunks(C), refs(d) == 1)
                  + manifest_bytes(C)
                  + sum(|b_seg| : seg in segments(C), seg_refs(seg) == 1)
retained_shared_bytes(C) = sum(|b_d| : d in chunks(C), refs(d) >= 2)
```

Chunk byte counts come from chunk descriptors (`descriptor["bytes"]`, the
exact encoded length). Two cost layers sit on top of encoded bytes:

- **Allocation rounding.** Files occupy whole allocation blocks.
  `CommitReceipt.allocated_bytes_upper_bound` already computes the upper
  bound `sum(ceil(size / 4096) * 4096)` across the manifest and every chunk;
  maintenance instead measures `st_blocks * 512`. Small metadata appearing
  large under 4 KiB rounding is one reason small manifests and alias files
  cost more physical space than their encoded length suggests.
- **Write-path impossibility of "cheap" rewrite.** Because publication
  stages whole temporary files even when content exists, the exclusive
  write cost of touching an existing shared trunk chunk is its full encoded
  size (section 9). Admission policy that avoids re-staging unchanged
  content is therefore the cheapest SSD lever available; the connector's
  `store_skipped_present` and quorum counters already remove most of this
  before the write path, which is the implemented (non-heat) baseline.

First-publisher attribution: the publisher that first creates an object pays
its host-write cost; later referencing roots pay zero retained bytes for it.
The ledger attributes marginal bytes to each root as defined above, which is
the quantity an eviction decision needs.

## 7. TinyLFU shadow evaluation

The design evaluates frequency-aware admission *off the serving path* by
replaying traces through a shadow instance — a bounded, pure-Python cache
that mirrors the resident set under a candidate admission policy and reports
the decisions it would have made.

Structure (W-TinyLFU-style, simplified):

1. **Window** (`window_capacity`, default 1024 entries, plain LRU): every
   distinct-key miss passes through the window before touching the main
   cache, which separates one-shot scans from reusable contexts.
2. **Main cache** (`main_capacity`, default 65,536 entries, plain LRU in the
   prototype; the full design segments it into a protected/probationary SLRU
   pair — the prototype's single-band simplification is stated explicitly
   and only weakens discrimination among main-cache residents).
3. **Admission comparison**: when the window overflows, the evicted window entry
   is admitted to the main cache iff the main cache has spare capacity or
   the adversary comparison `estimate(candidate) > estimate(victim_lru)`
   holds, where both estimates come from the section 3 ring with the same
   decay discipline. On a loss, the victim stays and the candidate is
   dropped (no ghost-band admission in the prototype).
4. **Sketch discipline**: BLAKE2b-64bit-indexed 8-bit ring as in section 3,
   incremented on every access (resident or not), decayed by the shared
   window.

Deviations from textbook W-TinyLFU, deliberate and stated: single-band main
cache (no SLRU segmentation), no ghost admission on rejection, no Cuckoo
filter (a direct-mapped ring substitues). They bias the shadow toward
slightly *pessimistic* hit retention relative to full W-TinyLFU.

Decision output per access, as `ShadowDecision`:

| reason | meaning |
|---|---|
| `resident_window` | key was resident in the window |
| `resident_main` | key was resident in the main cache (a shadow hit) |
| `spare_capacity` | main cache had room; admitted |
| `admission_win` | evicted the main-cache victim; admitted |
| `admission_loss` | candidate estimate at or below the victim; rejected |

`evaluate_trace(keys)` consumes an iterable without retaining every decision
and summarizes the trace into a
`TraceReport` (`requests`, `window_hits`, `main_hits`, `misses`, `admitted`,
`rejected`, `hit_rate`). Comparing `hit_rate` for (a) unlimited-cache
replay and (b) shadow admission replay against a recorded production hit
rate is the experiment this design exists to run before any connector
wiring. A deployment-integrable policy additionally needs quorum-aware
cohort decisions across ranks: an admission must hold for all physical
ranks or none (research-only prerequisite listed in section 12).

Status: **implemented** as a research prototype; integration is
**research-only**; the shadow never affects production serving (**unsupported**
by design, per the isolation contract).

## 8. Hourly and daily staged-write budget simulation

A **logical retained byte** is a byte of encoded durable state that became
newly reachable under a cache root.
Events are recorded per publication with the fields of schema
`sparkcache-research-write-event/v1`:

```json
{
  "schema": "sparkcache-research-write-event/v1",
  "at_ns": 1756425600000000000,
  "kind": "commit",
  "storage_key": "<64 hex characters>",
  "context_digest": "<64 hex characters>",
  "unique_object_bytes": 183014,
  "staged_write_bytes": 183014
}
```

- `kind` is one of `commit` (exact chunks plus manifest), `alias_publication`
  (alias files plus added descriptor segments), `metadata_touch` (recency
  metadata), or `repair` (invalidation-driven republish).
- `unique_object_bytes` is the encoded size of objects that did not exist
  before publication and remain reachable afterward. `CommitReceipt` does
  not expose this quantity: `CommitReceipt.encoded_bytes` describes the
  complete root or delta represented by the receipt, including referenced
  objects that may already exist. Serving integration therefore requires
  explicit per-object publication instrumentation.
- `staged_write_bytes` counts payload bytes passed to temporary-file writes,
  including re-staging of identical content (section 9). The prototype
  requires the caller to supply this value because the publication helpers do
  not expose it.

Windows are UTC-aligned from absolute nanoseconds: the hourly window index
is `at_ns // 3_600_000_000_000` and the daily index is
`at_ns // 86_400_000_000_000`. `WriteLedger.hourly_reports(budget)` and
`.daily_reports(budget)` fold events into `BudgetReport` rows per window:

```json
{
  "window_start_ns": 1756425600000000000,
  "window_end_ns":   1756429200000000000,
  "unique_object_bytes": 1245583360,
  "staged_write_bytes":  2242054400,
  "limit_bytes":    2000000000,
  "exceeded": true,
  "over_bytes": 242054400,
  "events": 97
}
```

Budget limits are `WriteBudget(hourly_limit_bytes=None,
daily_limit_bytes=None)`; `None` means monitored-not-limited and sets
`exceeded = None`. A configured limit applies to `staged_write_bytes`, the
prototype's closest in-process measure of write-path pressure. `exceeded` is
a reported fact about a completed window.
Nothing in the prototype enforces anything: enforcement — declining or
deferring a publication whose projected commit would exceed a budget — is
**unsupported**. Enforcement would need: a pre-commit projection API on
`ManifestTransaction` (transactions expose no projected-total
query), a decision point that cannot block a serving thread (violating
"serve never waits" is the known risk), and a defined degradation (skip the
store cleanly, like `store_skipped_busy` does). Those prerequisites are
listed in section 11; until they exist, budgets are reports only.

## 9. Logical versus physical write amplification

Three byte quantities are distinct and all three are measured or modeled:

| Quantity | Meaning | Source |
|---|---|---|
| `unique_object_bytes` | bytes of newly retained durable state | caller-supplied ledger events; no serving receipt exposes this value |
| `staged_write_bytes` | payload bytes pushed through temporary-file writes, including staging of identical content | caller-supplied ledger events; publication helpers do not expose this value |
| `host_written_bytes` | device-side counter of host writes over an interval | NVMe Data Units Written delta (section 10) |

Derived ratios the prototype reports:

```
staging_ratio    = staged_write_bytes / unique_object_bytes
host_ratio       = host_written_bytes / unique_object_bytes     # the meaningful WAF proxy
```

Why they diverge, mechanically, in this codebase:

1. **Identical-content re-staging.** `_publish_immutable` and
   `_publish_immutable_batch` always write a `.writing-<uuid>` temporary
   before attempting the link. When the destination already exists with
   identical bytes (the common case when a quorum loser re-publishes, or
   when a grown conversation's earlier chunks are re-committed), the bytes
   are written, verified, discarded, and the link is a metadata op. The
   retained-log accounting shows zero; the device counter does not. The
   scheduler-side dedup (`store_skipped_present`, quorum short-circuits)
   keeps most of this off the write path already; a heat-independent
   improvement would be probing destination existence before staging —
   not proposed here, only named as the mechanism.
2. **Allocation rounding and metadata.** Each object rounds up to whole
   allocation blocks (4 KiB-programmed bound in
   `allocated_bytes_upper_bound`; filesystem-dependent in measurement), and
   directory fsyncs plus temp-file create/unlink cycles contribute
   bookkeeping blocks invisible to logical accounting.
3. **Alias publication.** Publishing up to 64 aliases plus descriptor
   segments over one exact root is small logically (segment files are ~16
   descriptors of small JSON) but multiplies file count; with block
   rounding each segment file dominates its content. Device amplification
   from alias publication concentrates in block-rounding, measured only
   through `host_ratio`.
4. **Off-cache writes share the device.** `host_ratio` computed against
   cache-ledger bytes is valid only within a controlled interval where the
   workload's non-cache writes are known to be zero or bounded. Otherwise
   the ratio is an upper bound and must be reported as such.

`write_amplification(unique_object_bytes, staged_write_bytes,
host_written_bytes)` returns a `WriteAmplificationEstimate` with the two
ratios and explains missing inputs as `None` rather than substituting
guesses. Physical (media-level) amplification — NAND writes versus host
writes, garbage collection effects — is **unsupported**: the standard Data
Units Written counter is a host-interface counter and deliberately does not
report media amplification; a device-specific endurance telemetry field
would be required and none is consumed here.

## 10. NVMe Data Units Written monitoring

The field layout and counter semantics below follow the
[NVM Express Base Specification 2.3](https://nvmexpress.org/wp-content/uploads/NVM-Express-Base-Specification-Revision-2.3-2025.08.01-Ratified.pdf),
SMART / Health Information log. The log is 512 bytes and may be obtained, for example,
as the output of `nvme smart-log /dev/nvme0` on a system with `nvme-cli`,
or the same log page read programmatically) contains at fixed offsets:

| Offset | Size | Field |
|---|---|---|
| 0x00 | 1 | Critical Warning (bit flags) |
| 0x02 | 2 | Composite Temperature (kelvin, little-endian) |
| 0x04 | 1 | Available Spare |
| 0x05 | 1 | Available Spare Threshold |
| 0x06 | 1 | Percentage Used |
| 0x20 | 16 | Data Units Read (128-bit little-endian) |
| 0x30 | 16 | Data Units Written (128-bit little-endian) |

Data Units Written counts host writes in units of 1,000 512-byte data
units, rounded up, metadata excluded; the count is converted to 512-byte
units regardless of the namespace LBA size. Therefore:

```
host_written_bytes_estimate = data_units_written_units * 512_000
```

The counter is quantized and rounded up, so a delta is an estimate rather than
an exact byte count (the unit constant is `DUW_UNIT_BYTES` in the prototype).
A reported value of
`0` means "not reported" per the specification, so `SmartHealthSample`
carries `data_units_written_reported = units != 0` and practitioners must
treat a zero as missing rather than as zero bytes written.

The prototype `parse_smart_log_page(page)` consumes the raw 512 bytes,
validates the length and the two 128-bit fields, and returns a
`SmartHealthSample` at an operator-supplied `at_ns`.
`DuwMonitor.delta(first, second)` yields `DuwDelta(units, bytes_est,
seconds, rate_bytes_per_second)` and raises `ResearchFormatError` naming
its cause when either counter is unreported, the counter decreases (device
replacement, counter reset, or samples from different devices), or the
samples carry conflicting non-empty device identifiers.

Monitoring procedure (operator-executed, no connector involvement):

1. Sample on a cadence with a stable clock (daily is adequate; hourly if
   correlating against `WriteLedger` windows).
2. Persist `sample_to_json(sample)` rows, schema
   `sparkcache-research-ssd-sample/v1`.
3. Compare each delta's `bytes_est` against the matching
   `daily_reports().unique_object_bytes` (and `staged_write_bytes`) —
   `host_ratio` from section 9 is the resulting write-amplification proxy.
4. Relate deltas to endurance: a delta's bytes over the device's rated
   TBW gives a consumed-endurance fraction; `Percentage Used` (offset
   0x06) is the vendor's own normalization and should be checked for
   drift against the computed fraction.

Limitations, stated: the counter covers the whole device, not a directory;
multi-tenant devices need controlled intervals; it excludes metadata
host-side yet includes file-system metadata writes, so device attribution
is always approximate; and acquiring the log page itself (ioctl plumbing,
out-of-band `smartctl` parsing) is intentionally outside the prototype.

## 11. Offline prototype

`research/heat_ssd_control/` holds the prototype. Contract:

- **Excluded from serving packages.** `pyproject.toml` includes only
  `sparkcache*` packages, so the research package is absent from wheels and
  runtime images. No production module imports it, no configuration option
  constructs it, and `import sparkcache` does not import it.
- **No serving-code dependency in either direction.** The modules import
  nothing from `sparkcache`, and the production-import isolation regression
  parses every Python module under `sparkcache/` to reject a reverse import.
  Chunk geometry (256 tokens) is a caller-supplied
  parameter whose value must match `CacheIdentity.chunk_tokens` for the
  modeled root; the modules do not read it from the codebase.
- **Side-effect free.** No filesystem, network, logging, thread, or
  subprocess use. All state is explicit constructor/method arguments.
- **No vllm or torch imports**, keeping the modules GPU-free-runnable.

Files:

| File | Contents |
|---|---|
| `research/heat_ssd_control/__init__.py` | Package marker restating the offline isolation contract |
| `research/heat_ssd_control/heat_model.py` | `ResearchFormatError`, `HeatKey`, `HitRing` (8-bit ring, saturation, epoch decay, snapshots), `recomputation_tokens_avoided`, `PublishedContext`, `ChunkLedger` (reference counts, shared-trunk reports, marginal bytes) |
| `research/heat_ssd_control/shadow_admission.py` | `ShadowConfig`, `TinyLFUShadow`, `ShadowDecision`, `TraceReport`, `evaluate_trace` |
| `research/heat_ssd_control/write_budget.py` | `DUW_UNIT_BYTES`, `WriteEvent`, `event_to_json`, `WriteBudget`, `WriteLedger`, `BudgetReport`, `write_amplification`, `parse_smart_log_page`, `SmartHealthSample`, `DuwMonitor`, `DuwDelta`, sample JSON schema `sparkcache-research-ssd-sample/v1` |
| `research/heat_ssd_control/test_prototype.py` | GPU-free behavior, malformed-input, atomic-ledger, and production-import isolation regressions |

Rejection behavior across all modules:

- Malformed external inputs — digest shape, negative counts, misaligned byte
  arrays, wrong schema strings, short log pages, or a counter reset in a
  delta — raise `ResearchFormatError` (a `ValueError` subclass) with a
  message naming the field. Nothing is retried, repaired, logged, or
  swallowed.
- Unknown-but-parseable inputs are still rejected when the schema demands
  exact keys; forward-compatibility guessing is deliberately absent so a
  modeling mistake surfaces immediately.
- Counter decrease, conflicting device identity, and duplicate-but-different
  publications raise rather than silently overwrite: a model that disagrees
  with its inputs must stop, not reconcile.

Example, end to end with no serving dependency:

```python
from research.heat_ssd_control.heat_model import HeatKey, HitRing
from research.heat_ssd_control.shadow_admission import TinyLFUShadow, ShadowConfig
from research.heat_ssd_control.write_budget import (
    DUW_UNIT_BYTES, parse_smart_log_page, DuwMonitor, WriteEvent, WriteLedger,
    WriteBudget,
)

key = HeatKey("a" * 64, "b" * 64)
ring = HitRing()
ring.record_hit(key)
assert ring.estimate(key) == 1

shadow = TinyLFUShadow(ShadowConfig(window_capacity=4, main_capacity=4))
trace = [key] * 4 + [HeatKey("a" * 64, "c" * 64)] * 2 + [key]
report = shadow.evaluate_trace(trace)
assert report.requests == 7 and report.hit_rate > 0

event = WriteEvent(at_ns=0, kind="commit", storage_key=key.storage_key,
                   context_digest=key.context_digest,
                   unique_object_bytes=1000, staged_write_bytes=1000)
ledger = WriteLedger(); ledger.add(event)
report = ledger.hourly_reports(WriteBudget(hourly_limit_bytes=500))
assert report[0].exceeded

# sample = operator-supplied 512-byte SMART/Health log page
# delta = DuwMonitor.delta(parse_smart_log_page(page_a, at_ns=t0),
#                          parse_smart_log_page(page_b, at_ns=t1))
# delta.bytes_est == delta.units * DUW_UNIT_BYTES
```

## 12. Integration prerequisites

Any change that makes serving consume this design requires, in addition to
the GPU-free prototype tests:

1. A quorum-consistent admission point: publication candidates must be
   accepted or rejected identically across every physical rank, or the
   verified-or-recompute contract must tolerate rank-divergent publications
   (partial publication is invisible until
   `commit_manifest`, so a divergent rejection simply produces no manifest —
   that direction is safe, divergent *acceptance* is not).
2. A pre-commit projection hook on `ManifestTransaction` for budget checks,
   plus a defined skip-publication degradation that keeps store failures
   non-fatal in the same way as other optional publication work.
3. A persistence decision for counters (or documented cold start), resolved
   against the isolation contract's prohibition of heat in authenticated
   surfaces: if heat is persisted, its files live outside
   `_CACHE_DATA_DIRECTORIES` and clear-once semantics must be extended
   explicitly.
4. A measured qualification showing cache-active time-to-first-token and
   decode throughput within 2% of the cache-off profile (the standing
   model-serving qualification requirement in `ROADMAP.md`).

## 13. GPU-free validation

The offline prototype was validated on CPython 3.12.10 without CUDA, vLLM,
or torch:

| Command | Result | Conclusion |
|---|---:|---|
| `python -m pytest research -q` | 22 passed | Prototype behavior and serving-package isolation are covered |
| `python -m pytest sparkcache -q` | 738 passed, 7 skipped | The prototype does not change the deployed SparkCache source contract |
| `python -m pytest deploy -q` | 108 passed, 1 skipped | Deployment profile and source-hash checks remain unchanged |
| `python -m ruff check .` | passed | Repository lint rules pass |
