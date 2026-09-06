from __future__ import annotations

import ctypes
from pathlib import Path

import pytest

from sparkcache.streaming.manager_page_capture import (
    ManagerPageExtensionCapturePlan,
    ManagerPageCaptureError,
    ManagerPageSource,
    plan_manager_page_extension_capture,
    plan_manager_page_capture,
)
from sparkcache.spark_context_cache_hybrid import (
    PageGroup,
    PageLayer,
    PageLayout,
    encode_page_snapshot,
    encode_page_snapshot_header,
)


class _NativeSource(ctypes.Structure):
    _fields_ = [
        ("source_base", ctypes.c_uint64),
        ("source_pages", ctypes.c_uint64),
        ("source_page_stride_bytes", ctypes.c_uint64),
        ("bytes_per_page", ctypes.c_uint32),
        ("group_index", ctypes.c_uint32),
        ("layer_ordinal", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
    ]


class _NativeGroup(ctypes.Structure):
    _fields_ = [
        ("physical_page_offset", ctypes.c_uint32),
        ("page_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32 * 2),
    ]


class _NativeSpan(ctypes.Structure):
    _fields_ = [
        ("destination_offset_bytes", ctypes.c_uint64),
        ("length_bytes", ctypes.c_uint64),
        ("source_index", ctypes.c_uint32),
        ("physical_page_offset", ctypes.c_uint32),
        ("page_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NativePlan(ctypes.Structure):
    _fields_ = [
        ("used_bytes", ctypes.c_uint64),
        ("span_count", ctypes.c_uint32),
        ("group_count", ctypes.c_uint32),
        ("source_count", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


class _NativeSubmission(ctypes.Structure):
    _fields_ = [
        ("context_sequence", ctypes.c_uint64),
        ("logical_start", ctypes.c_uint64),
        ("physical_page_count", ctypes.c_uint32),
        ("group_count", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("reserved", ctypes.c_uint32),
    ]


def _sources() -> tuple[ManagerPageSource, ...]:
    return (
        ManagerPageSource(0x1000, 1024, 4096, 4096, 0, 0),
        ManagerPageSource(0x2000, 1024, 8192, 6144, 0, 1),
        ManagerPageSource(0x3000, 1024, 256, 192, 1, 0),
    )


def test_manager_page_plan_is_group_then_layer_ordered() -> None:
    plan = plan_manager_page_capture(
        _sources(),
        ((17, 3, 91), (8, 5)),
        slot_bytes=64 * 1024,
    )

    assert plan.physical_pages == (17, 3, 91, 8, 5)
    assert plan.group_offsets == (0, 3)
    assert plan.group_page_counts == (3, 2)
    assert plan.used_bytes == 3 * 4096 + 3 * 6144 + 2 * 192
    assert [span.group_index for span in plan.spans] == [0, 0, 1]
    assert [span.destination_offset_bytes for span in plan.spans] == [
        0,
        3 * 4096,
        3 * 4096 + 3 * 6144,
    ]
    assert [span.physical_page_offset for span in plan.spans] == [0, 0, 3]


def test_extension_capture_reuses_complete_attention_pages_and_replaces_state() -> None:
    plan = plan_manager_page_extension_capture(
        _sources(),
        ((17, 3, 91, 44), (8,)),
        base_page_counts=(2, 1),
        logical_tokens_per_page=(256, 2304),
        reuse_policies=("full", "recurrent_align"),
        base_boundary_tokens=512,
        slot_bytes=64 * 1024,
    )

    assert isinstance(plan, ManagerPageExtensionCapturePlan)
    assert plan.base_page_counts == (2, 1)
    assert plan.result_page_counts == (4, 1)
    assert plan.reused_pages_by_group == (2, 0)
    assert plan.capture.physical_pages == (91, 44, 8)
    assert plan.capture.group_page_counts == (2, 1)
    assert plan.capture.used_bytes == 2 * 4096 + 2 * 6144 + 192


def test_extension_capture_replaces_a_partial_terminal_attention_page() -> None:
    plan = plan_manager_page_extension_capture(
        _sources(),
        ((17, 3, 91), (8, 5)),
        base_page_counts=(2, 1),
        logical_tokens_per_page=(256, 512),
        reuse_policies=("full", "sliding"),
        base_boundary_tokens=384,
        slot_bytes=64 * 1024,
    )

    assert plan.reused_pages_by_group == (1, 0)
    assert plan.capture.physical_pages == (3, 91, 8, 5)
    assert plan.capture.group_page_counts == (2, 2)
    assert plan.capture.used_bytes == 2 * 4096 + 2 * 6144 + 2 * 192


def test_manager_page_plan_matches_block_page_codec_body_order() -> None:
    layout = PageLayout(
        (
            PageGroup(
                256,
                (
                    PageLayer("attention.0", "u8", (4,), 4),
                    PageLayer("attention.1", "u8", (6,), 6),
                ),
            ),
            PageGroup(
                2304,
                (PageLayer("recurrent.0", "u8", (3,), 3),),
                reuse_policy="recurrent_align",
            ),
        )
    )
    sources = (
        ManagerPageSource(0x1000, 8, 4, 4, 0, 0),
        ManagerPageSource(0x2000, 8, 6, 6, 0, 1),
        ManagerPageSource(0x3000, 8, 3, 3, 1, 0),
    )
    groups = ((2, 5), (7,))
    payloads = {
        "attention.0": b"a002" + b"a005",
        "attention.1": b"b00002" + b"b00005",
        "recurrent.0": b"r07",
    }
    plan = plan_manager_page_capture(sources, groups, slot_bytes=64)
    encoded = encode_page_snapshot(layout, (2, 1), payloads)
    header = encode_page_snapshot_header(layout, (2, 1))

    assert encoded[len(header) :] == b"".join(payloads.values())
    assert plan.used_bytes == len(encoded) - len(header)
    assert [span.length_bytes for span in plan.spans] == [8, 12, 3]


@pytest.mark.parametrize(
    ("sources", "groups", "slot_bytes", "match"),
    (
        (
            (
                ManagerPageSource(0x1000, 4, 16, 16, 0, 1),
            ),
            ((0,),),
            16,
            "dense",
        ),
        (
            (
                ManagerPageSource(0x1000, 4, 16, 16, 0, 0),
                ManagerPageSource(0x2000, 5, 16, 16, 0, 1),
            ),
            ((0,),),
            32,
            "share physical-page capacity",
        ),
        (_sources(), ((17, 3, 1024), (8, 5)), 64 * 1024, "outside"),
        (_sources(), ((17, 3, 91), (8, 5)), 1024, "bounded ring slot"),
        (
            (
                ManagerPageSource((1 << 64) - 8, 4, 16, 16, 0, 0),
            ),
            ((0,),),
            16,
            "address range",
        ),
    ),
)
def test_manager_page_plan_rejects_unproven_geometry(
    sources: tuple[ManagerPageSource, ...],
    groups: tuple[tuple[int, ...], ...],
    slot_bytes: int,
    match: str,
) -> None:
    with pytest.raises(ManagerPageCaptureError, match=match):
        plan_manager_page_capture(sources, groups, slot_bytes=slot_bytes)


def test_native_page_capture_structure_sizes_are_fixed() -> None:
    assert ctypes.sizeof(_NativeSource) == 40
    assert ctypes.sizeof(_NativeGroup) == 16
    assert ctypes.sizeof(_NativeSpan) == 32
    assert ctypes.sizeof(_NativePlan) == 24
    assert ctypes.sizeof(_NativeSubmission) == 32


def test_live_block_page_streaming_remains_disabled() -> None:
    root = Path(__file__).resolve().parents[1]
    config = (root / "spark_context_cache_config.py").read_text(encoding="utf-8")
    connector = (root / "spark_context_cache_connector.py").read_text(
        encoding="utf-8"
    )
    assert "block-page storage does not support" in config
    assert "streaming snapshots" in config
    assert "streaming snapshots do not support" in connector
    assert "multiple KV-cache groups" in connector
