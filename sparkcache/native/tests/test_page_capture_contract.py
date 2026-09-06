from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = ROOT / "include" / "spark_cache_page_capture.h"
SNAPSHOT_HEADER = ROOT / "include" / "spark_cache_snapshot.h"
LAYOUT = ROOT / "src" / "spark_cache_page_capture_layout.cpp"
CUDA = ROOT / "src" / "spark_cache_snapshot.cu"
RUNTIME = ROOT.parents[0] / "streaming" / "manager_page_runtime.py"
CONNECTOR = ROOT.parents[0] / "spark_context_cache_connector.py"
CONFIG = ROOT.parents[0] / "spark_context_cache_config.py"
LEASE_CONTRACT = (
    ROOT.parents[0]
    / "runtime_patches"
    / "vllm-manager-page-async-contract-55969c16.json"
)


def test_page_capture_contract_is_bounded_and_pointer_free_on_disk() -> None:
    header = HEADER.read_text(encoding="utf-8")
    assert "SPARK_CACHE_PAGE_CAPTURE_CONTRACT_VERSION 1u" in header
    assert "SPARK_CACHE_PAGE_CAPTURE_MAX_GROUPS 16u" in header
    assert "SPARK_CACHE_PAGE_CAPTURE_MAX_SOURCES 256u" in header
    assert "physical" in header
    assert "page IDs are transient" in header
    assert "uint64_t destination_offset_bytes" in header
    assert "uint64_t length_bytes" in header


def test_page_capture_layout_checks_request_and_slot_bounds() -> None:
    layout = LAYOUT.read_text(encoding="utf-8")
    assert "physical_page_offset != expected_page_offset" in layout
    assert "expected_page_offset != physical_page_count" in layout
    assert "physical_pages[descriptor.physical_page_offset + page]" in layout
    assert "end > slot_bytes" in layout
    assert "planned_spans" in layout
    assert layout.index("planned_spans[index]") < layout.index("spans[index] =")


def test_page_capture_contract_does_not_enable_live_manager_page_streaming() -> None:
    connector = CONNECTOR.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert "block-page storage does not support" in config
    assert "streaming snapshots" in config
    assert "streaming snapshots do not support" in connector
    assert "multiple KV-cache groups" in connector


def test_page_capture_cuda_submission_has_distinct_stream_events() -> None:
    header = SNAPSHOT_HEADER.read_text(encoding="utf-8")
    cuda = CUDA.read_text(encoding="utf-8")
    assert "SPARK_CACHE_SNAPSHOT_CAP_MANAGER_PAGE_CAPTURE" in header
    assert "spark_cache_snapshot_configure_page_sources" in header
    assert "spark_cache_snapshot_try_submit_pages" in header
    assert "spark_cache_snapshot_drain_context" in header
    assert "cudaStreamCreateWithPriority" in cuda
    assert "cudaStreamNonBlocking" in cuda
    assert "least_priority" in cuda

    submit = cuda.index("spark_cache_snapshot_try_submit_pages")
    poll = cuda.index("spark_cache_snapshot_poll", submit)
    body = cuda[submit:poll]
    assert "cudaEventRecord(slot.producer_ready, producer)" in body
    assert "cudaStreamWaitEvent(slot.capture_stream, slot.producer_ready, 0)" in body
    assert "gather_manager_pages_kernel<<<" in body
    assert "slot.capture_stream" in body
    assert "record_snapshot_completion_event(slot.complete, slot.capture_stream)" in body
    assert "dim3(page_plan.span_count, max_group_pages)" in body
    assert "byte_index / source.bytes_per_page" not in body
    assert "cudaStreamSynchronize" not in body
    assert "cudaDeviceSynchronize" not in body


def test_page_capture_preemption_drain_is_explicit_and_synchronous() -> None:
    cuda = CUDA.read_text(encoding="utf-8")
    start = cuda.index("spark_cache_snapshot_drain_context")
    body = cuda[start:]
    assert "cudaEventSynchronize" in body
    assert "ring.reap_discarded" in body


def test_connector_async_page_capture_keeps_synchronous_fallback() -> None:
    connector = CONNECTOR.read_text(encoding="utf-8")
    assert "if self._async_page_capture_enabled:" in connector
    assert "runtime.submit(plan, producer_stream=producer_stream)" in connector
    assert "self._snapshot_store(plan)" in connector
    assert "runtime.preempt(request_id)" in connector
    assert "runtime.take_finished(finished_req_ids)" in connector
    assert "_complete_async_page_capture" in connector
    assert "manager-page capture cannot" in connector
    assert "start while persistent cache initialization is unavailable" in connector


def test_manager_page_async_contract_names_all_group_and_boundary_ownership() -> None:
    import json

    payload = json.loads(LEASE_CONTRACT.read_text(encoding="utf-8"))
    symbols = {
        symbol
        for entry in payload["files"]
        for symbol in entry["required_symbols"]
    }
    assert payload["vllm_commit"] == "55969c16d4da57da76ee5729f3102d4b2003833c"
    assert "SupportsHMA.request_finished_all_groups" in symbols
    assert "Scheduler._update_from_kv_xfer_finished" in symbols
    assert "KVOutputAggregator.aggregate" in symbols
    assert "KVConnectorModelRunnerMixin._get_kv_connector_output" in symbols
    assert "KVCacheManager._pin_recurrent_boundary" in symbols
    assert "KVCacheManager.pop_blocks_for_free" in symbols


def test_background_handoff_keeps_claimed_ring_bytes_scattered() -> None:
    runtime = RUNTIME.read_text(encoding="utf-8")
    assert "PageSnapshotScatter(" in runtime
    assert "bytes(claimed.payload)" not in runtime
    assert "header + body" not in runtime
    assert "self._mark_completed_locked(" in runtime
