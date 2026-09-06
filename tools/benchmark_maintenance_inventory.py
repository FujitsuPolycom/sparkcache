"""Measure complete maintenance and offer reconciliation using GPU-free fixtures.

Run from the checkout being measured. Positional arguments specify root/chunk
counts for BENCH_KIND=rows (default), or branch/extension counts for pages.
BENCH_FILESYSTEM records the independently verified filesystem type. Test
fixtures supply connector API stubs; CUDA and model execution are not used.
"""
from collections import Counter
import gc
import os
import platform
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch

sys.path.insert(0, str(Path.cwd()))
from sparkcache.persistent_context_cache.cache_manifest import CapacityPolicy
from sparkcache.persistent_context_cache.test_cache_manifest import _chunk, _identity
from sparkcache.test_spark_context_cache_connector import _make_connector

roots, chunks = map(int, sys.argv[1:3])
with tempfile.TemporaryDirectory(prefix="sparkcache-inventory-bench-") as directory:
    connector = _make_connector(Path(directory), 0)
    identity = _identity()
    connector._identity = lambda rank: identity
    connector._capacity_policy = CapacityPolicy(max_bytes=10**9, low_watermark_bytes=10**9)
    kind = os.environ.get("BENCH_KIND", "rows")
    if kind == "pages":
        from sparkcache.persistent_context_cache.cache_manifest import CacheIdentity
        from sparkcache.spark_context_cache_codec import context_prefix_digest
        from sparkcache.spark_context_cache_hybrid import PageGroup, PageLayer, PageLayout, encode_page_snapshot
        identity = CacheIdentity(target_checkpoint="1" * 64, draft_checkpoint="2" * 64,
                                 quantization_layout="benchmark-page", rope_layout="glm53-hybrid-v1",
                                 tp_degree=4, dcp_degree=4, chunk_tokens=256,
                                 record_schema=("target_ckv", "logical_positions"),
                                 publication_schema="page-tail-cow-v2")
        connector._identity = lambda rank: identity
        connector._storage_mode = "block_pages"
        layout = PageLayout((PageGroup(256, (PageLayer("attention", "u8", (64,), 64),)),
                             PageGroup(1, (PageLayer("recurrent", "u8", (32,), 32),))))
        tokens_base = tuple(range(512))
        salt = "shared-base-page-profile"
        base_digest = context_prefix_digest(tokens_base, salt, token_count=512)
        connector._store.commit_page_snapshot(identity=identity, context_digest=base_digest,
            span_tokens=512, snapshot=encode_page_snapshot(layout, (2, 1),
            {"attention": b"a" * 128, "recurrent": b"r" * 32}))
        for branch in range(roots):
            tokens = tokens_base + tuple(range(100000 * (branch + 1), 100000 * (branch + 1) + chunks * 256))
            prior = base_digest
            payload = b"a" * 128
            for stage in range(chunks):
                payload += bytes((stage + branch,)) * 64
                boundary = (stage + 3) * 256
                snapshot = encode_page_snapshot(layout, (stage + 3, 1), {
                    "attention": payload, "recurrent": bytes((stage + 64,)) * 32})
                connector._store.commit_page_extension(identity=identity, base_context_digest=prior,
                    token_ids=tokens, identity_salt=salt, layout=layout,
                    base_block_counts=(stage + 2, 1), result_block_counts=(stage + 3, 1),
                    base_boundary_tokens=(stage + 2) * 256,
                    result_boundary_tokens=boundary, result_snapshot=snapshot)
                prior = context_prefix_digest(tokens, salt, token_count=boundary)
        digests = {path.stem for path in (Path(directory) / "manifests" / identity.storage_key).glob("*.json")}
        references = sum(len(connector._store._capacity_entry(path).chunks)
                         for path in (Path(directory) / "manifests" / identity.storage_key).glob("*.json"))
    else:
        payloads = tuple(_chunk(i * 256, (i + 1) * 256) for i in range(chunks))
        digests = {f"{i + 1:064x}" for i in range(roots)}
        for digest in digests:
            connector._store.commit(identity=identity, context_digest=digest, chunks=payloads)
        references = roots * chunks
    unique_chunks = len(list((Path(directory) / "chunks").glob("*.spcc")))
    connector._held.update(digests)
    counts = Counter()
    read_bytes, stat = Path.read_bytes, Path.stat

    def read(path):
        if path.suffix == ".json":
            counts["root_reads"] += 1
        return read_bytes(path)

    def metadata(path, *args, **kwargs):
        if path.parent.name == "chunks" and path.suffix == ".spcc":
            counts["chunk_stats"] += 1
        return stat(path, *args, **kwargs)

    with patch.object(Path, "read_bytes", read), patch.object(Path, "stat", metadata):
        connector._maintain_capacity(force=True)
    elapsed = []
    for _ in range(7):
        gc.collect()
        start = time.perf_counter()
        report = connector._maintain_capacity(force=True)
        elapsed.append((time.perf_counter() - start) * 1000)
        assert set(connector._held) == digests
        assert report.manifests_evicted == report.chunks_deleted == 0
    connector.shutdown()
    print(json.dumps({"revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
                      "kind": kind, "roots": len(digests), "unique_chunks": unique_chunks, "references": references,
                      "platform": platform.platform(), "filesystem": os.environ.get("BENCH_FILESYSTEM", "unspecified"),
                      "manifest_source_sha256": hashlib.sha256(Path('sparkcache/persistent_context_cache/cache_manifest.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                      "connector_source_sha256": hashlib.sha256(Path('sparkcache/spark_context_cache_connector.py').read_bytes().replace(b'\r\n', b'\n')).hexdigest(),
                      "io_counts": counts, "elapsed_ms": elapsed, "median_ms": statistics.median(elapsed)}))
