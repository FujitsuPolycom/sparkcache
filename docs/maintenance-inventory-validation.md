# Maintenance inventory and offer reconciliation

Status: **implemented**. Serving-performance qualification is **research-only**.

`MaintenanceReport.surviving_entries` identifies metadata-qualified roots from
the exclusive inventory pass. Offer reconciliation reuses that result without
rereading roots or probing shared payload files once per reference.

The [CPU benchmark record](maintenance-inventory-validation.json) measures the
complete maintenance and reconciliation call on Windows 11 with NTFS. It
compares equivalent fixtures against source revision `7fc9f85509624a0f9c5bc6bd7c5f70f1a43d389e`.

| Fixture | Baseline median | Inventory median | Root reads | Chunk metadata probes |
|---|---:|---:|---:|---:|
| Eight branches, 32 flat page extensions each | 175.23 ms | 72.22 ms | 514 to 257 | 4,995 to 514 |
| 64 token-row roots sharing 32 chunks | 37.38 ms | 10.14 ms | 128 to 64 | 2,112 to 64 |

A repeat with combined source `11658f0a3b9f155af1d95ff78a807c9b374f6093`,
including canonical string ordering, measured 78.41 ms for the flat-page fixture
and retained the same reduced I/O counts. The JSON records that source and its
seven observations separately.

The page fixture includes attention and recurrent-state bytes. It exercises
257 roots and 4,481 chunk references with tiny payloads, not a loaded model.
The measurements do not qualify DGX serving throughput or NVMe behavior.

Each normal pass still reads the complete inventory and traverses its reference
graph. Root parsing falls from two passes to one, and chunk metadata probes
scale with unique files instead of the total number of root references.

Run the [benchmark tool](../tools/benchmark_maintenance_inventory.py) from the
checkout being measured. Use the same tool against each source checkout.
Install the repository's CPU test dependencies; the connector fixture stubs
its model-runtime interfaces.

```powershell
$env:BENCH_FILESYSTEM = 'NTFS'
$env:BENCH_KIND = 'pages'
python tools/benchmark_maintenance_inventory.py 8 32
$env:BENCH_KIND = 'rows'
python tools/benchmark_maintenance_inventory.py 64 32
```

Set `BENCH_FILESYSTEM` to the independently verified filesystem. Fixture
construction and counting instrumentation are outside the seven timed passes.
The artifact records normalized source hashes for both measured implementations.

Regression tests in `sparkcache/test_maintenance_survivors.py` cover shared
payload metadata, corrupt exact roots shadowing aliases, storage-mode
eligibility, concurrent publication, and repeated inventory freshness.

The snapshot does not authorize cache restoration. Payload integrity remains
mandatory at restore, and same-sized corruption becomes a verified miss.
No cache identity, namespace, or persisted format changes.
