"""Group inventories preserve order and remain bounded for hybrid models."""
import pytest
import ctypes
from types import SimpleNamespace
from sparkcache.streaming import manager_page_native_ring_ctypes as binding
from sparkcache.streaming.manager_page_capture import (
    ManagerPageSource, ManagerPageCaptureError, plan_manager_page_capture,
)


@pytest.mark.parametrize('count', [17, 48, 64])
def test_capture_hybrid_group_inventory(count):
    sources = tuple(ManagerPageSource(4096 + i * 1024, 8, 16, 16, i, 0) for i in range(count))
    plan = plan_manager_page_capture(sources, ((2, 5),) * count, slot_bytes=count * 32)
    assert plan.used_bytes == count * 32
    assert plan.group_offsets == tuple(2 * i for i in range(count))
    assert [span.group_index for span in plan.spans] == list(range(count))
    assert [span.destination_offset_bytes for span in plan.spans] == list(range(0, count * 32, 32))


def test_capture_rejects_group_inventory_over_capacity():
    sources = tuple(ManagerPageSource(4096 + i * 1024, 8, 16, 16, i, 0) for i in range(65))
    with pytest.raises(ManagerPageCaptureError, match='group count'):
        plan_manager_page_capture(sources, ((2, 5),) * 65, slot_bytes=65 * 32)


@pytest.mark.parametrize('max_groups', [16, 64])
def test_native_group_capacity_must_match_binding(max_groups):
    class Function:
        def __init__(self, callback=lambda *args: 0):
            self.callback = callback

        def __call__(self, *args):
            return self.callback(*args)

    def query(pointer):
        info = ctypes.cast(pointer, ctypes.POINTER(binding.PageCaptureAbiInfo)).contents
        info.contract_version = binding.CONTRACT_VERSION
        info.max_groups = max_groups
        info.max_sources = binding.MAX_SOURCES
        for name, typ in [('source', binding.PageCaptureSource), ('group', binding.PageCaptureGroup),
                          ('span', binding.PageCaptureSpan), ('plan', binding.PageCapturePlan),
                          ('submission', binding.PageCaptureSubmission)]:
            setattr(info, 'sizeof_' + name, ctypes.sizeof(typ))
        info.capability_flags = binding.abi.CAP_MANAGER_PAGE_CAPTURE | binding.abi.CAP_LOW_PRIORITY_CAPTURE_STREAM
        return binding.abi.STATUS_OK

    library = SimpleNamespace(
        spark_cache_snapshot_query_page_capture_abi=Function(query),
        spark_cache_snapshot_configure_page_sources=Function(),
        spark_cache_snapshot_try_submit_pages=Function(),
        spark_cache_snapshot_drain_context=Function(),
    )
    if max_groups == 64:
        assert binding._bind_page_api(library).max_groups == 64
    else:
        with pytest.raises(binding.abi.NativeSnapshotError, match='ABI differs'):
            binding._bind_page_api(library)
