"""Opt-in diagnostics distinguish GPU lease attachment from persistent restore."""

import json
from types import SimpleNamespace

import pytest

from sparkcache import test_spark_context_cache_connector as fixtures
from sparkcache import spark_context_cache_connector as connector_module


def _records(monkeypatch):
    records = []

    def info(message, *args, **kwargs):
        if message.startswith("spark-context-cache-reuse:"):
            records.append(json.loads((message % args).split(":", 1)[1]))

    monkeypatch.setattr(connector_module.logger, "info", info)
    return records


@pytest.mark.parametrize("enabled", [False, True])
def test_offer_trace_uses_caller_prefix_and_is_latched_at_construction(tmp_path, monkeypatch, enabled):
    monkeypatch.setenv("SPARK_CONTEXT_CACHE_TRACE_REUSE", "1" if enabled else "0")
    records = _records(monkeypatch)
    connector = fixtures.AsyncRestoreTests()._cohort_connector(tmp_path)
    monkeypatch.setenv("SPARK_CONTEXT_CACHE_TRACE_REUSE", "0" if enabled else "1")
    tokens = list(range(1100))
    digest = fixtures.AsyncRestoreTests._offer(connector, tokens)
    request = SimpleNamespace(request_id="offer", prompt_token_ids=tokens)
    try:
        assert connector.get_num_new_matched_tokens(request, 256) == (768, True)
        assert connector.counters["restore_hit"] == 1
        assert connector._need_load["offer"] == (digest, 1024)
        if not enabled:
            assert records == []
            return
        assert len(records) == 1
        record = records[0]
        assert record["event"] == "external_restore_offer"
        assert record["caller_block_aligned_prefix_tokens"] == 256
        assert record["offered_external_tokens"] == 768
        assert record["selected_span_tokens"] == 1024
        assert record["request_id"] == "offer"
        assert "local_hit_tokens" not in record
    finally:
        connector.shutdown()


def test_lease_trace_requires_matching_attachment_and_is_not_restore(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_CONTEXT_CACHE_TRACE_REUSE", "1")
    records = _records(monkeypatch)
    connector = fixtures._make_connector(tmp_path, 0)
    key = "a" * 64
    connector._restore_flight_followers["lease"] = connector_module._RestoreFollower(key, key, 1024)
    try:
        connector.shared_prefix_lease_attached("missing", key)
        connector.shared_prefix_lease_attached("lease", "b" * 64)
        assert records == []
        connector.shared_prefix_lease_attached("lease", key)
        assert connector.counters["shared_prefix_leases_attached"] == 1
        assert connector.counters["load_verified"] == 0
        assert len(records) == 1
        assert records[0]["event"] == "gpu_lease_attached"
        assert records[0]["lease_span_tokens"] == 1024
        assert records[0]["request_id"] == "lease"
    finally:
        connector.shutdown()


@pytest.mark.parametrize("corrupt", [False, True])
def test_worker_trace_follows_actual_verified_or_recompute_result(tmp_path, monkeypatch, corrupt):
    monkeypatch.setenv("SPARK_CONTEXT_CACHE_TRACE_REUSE", "1")
    records = _records(monkeypatch)
    fixture = fixtures.AsyncRestoreTests()
    connector = fixtures._make_connector(tmp_path, 2)
    connector.register_kv_caches(fixtures._make_pools(8, 64))
    digest = "a" * 64
    fixture._store_entry(connector, digest)
    if corrupt:
        path = next((tmp_path / "chunks").glob("*.spcc"))
        payload = bytearray(path.read_bytes())
        payload[-1] ^= 1
        path.write_bytes(payload)
    connector.bind_connector_metadata(connector_module.SparkCacheConnectorMetadata(plans=[
        connector_module._ReqPlan("restore", digest, 1024, fixture.BLOCKS, False),
    ]))
    try:
        connector.start_load_kv(None)
        assert connector.wait_for_pending_loads(timeout=5)
        assert connector.get_finished(set())[1] == {"restore"}
        assert connector.counters["load_failed" if corrupt else "load_verified"] == 1
        assert len(records) == 1
        record = records[0]
        assert record["schema"] == "sparkcache-reuse-trace/v1"
        assert record["event"] == "worker_restore_completed"
        assert record["outcome"] == ("recompute" if corrupt else "verified")
        assert record["request_id"] == "restore"
        assert record["rank"] == record["dcp_rank"] == 2
        assert record["requested_span_tokens"] == 1024
        assert record["verified_span_tokens"] == (0 if corrupt else 1024)
        assert record["end_to_end_ms"] >= record["service_ms"] >= 0
        assert "restore_read" in record["phase_ms"]
        assert bool(connector.get_block_ids_with_load_errors()) == corrupt
    finally:
        connector.shutdown()


def test_trace_logger_failure_does_not_change_lease_decision(tmp_path, monkeypatch):
    monkeypatch.setenv("SPARK_CONTEXT_CACHE_TRACE_REUSE", "1")
    connector = fixtures._make_connector(tmp_path, 0)
    key = "a" * 64
    connector._restore_flight_followers["lease"] = connector_module._RestoreFollower(key, key, 1024)

    def fail(*args, **kwargs):
        raise OSError("diagnostic sink unavailable")

    monkeypatch.setattr(connector_module.logger, "info", fail)
    try:
        connector.shared_prefix_lease_attached("lease", key)
        assert connector.counters["shared_prefix_leases_attached"] == 1
    finally:
        connector.shutdown()
