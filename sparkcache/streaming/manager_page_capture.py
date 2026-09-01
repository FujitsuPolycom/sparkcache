"""GPU-free planner for bounded opaque manager-page capture.

The planner mirrors the C++ layout contract without loading CUDA. It does not
enable streaming publication. A model-serving adapter may use the result only
after it has separately established all-group block leases and an attested
C++/CUDA implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


PAGE_CAPTURE_CONTRACT_VERSION = 1
MAX_CAPTURE_GROUPS = 16
MAX_CAPTURE_SOURCES = 256
_UINT32_MAX = (1 << 32) - 1
_UINT64_MAX = (1 << 64) - 1


class ManagerPageCaptureError(ValueError):
    """A manager-page inventory or request cannot be captured safely."""


@dataclass(frozen=True, slots=True)
class ManagerPageSource:
    """One stable layer allocation viewed as complete physical pages."""

    source_base: int
    source_pages: int
    source_page_stride_bytes: int
    bytes_per_page: int
    group_index: int
    layer_ordinal: int


@dataclass(frozen=True, slots=True)
class ManagerPageCaptureSpan:
    """One layer's exact destination extent in the raw page body."""

    source_index: int
    group_index: int
    physical_page_offset: int
    page_count: int
    destination_offset_bytes: int
    length_bytes: int


@dataclass(frozen=True, slots=True)
class ManagerPageCapturePlan:
    """Validated request tables and their bounded raw payload layout."""

    physical_pages: tuple[int, ...]
    group_offsets: tuple[int, ...]
    group_page_counts: tuple[int, ...]
    spans: tuple[ManagerPageCaptureSpan, ...]
    used_bytes: int


def _uint(value: object, *, name: str, maximum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ManagerPageCaptureError(f"{name} must be an integer")
    if not 0 <= value <= maximum:
        raise ManagerPageCaptureError(f"{name} is outside its unsigned range")
    return value


def _positive_uint(value: object, *, name: str, maximum: int) -> int:
    result = _uint(value, name=name, maximum=maximum)
    if result == 0:
        raise ManagerPageCaptureError(f"{name} must be positive")
    return result


def plan_manager_page_capture(
    sources: Sequence[ManagerPageSource],
    physical_pages_by_group: Sequence[Sequence[int]],
    *,
    slot_bytes: int,
) -> ManagerPageCapturePlan:
    """Plan exact multi-group page bytes without touching live tensors.

    Sources are ordered by group and layer. Request page IDs retain the order
    supplied by each manager group. The output spans are therefore already in
    the canonical block-page body order used by ``encode_page_snapshot``.
    """

    source_table = tuple(sources)
    groups = tuple(tuple(group) for group in physical_pages_by_group)
    capacity = _positive_uint(slot_bytes, name="slot_bytes", maximum=_UINT64_MAX)
    if not source_table or len(source_table) > MAX_CAPTURE_SOURCES:
        raise ManagerPageCaptureError("manager-page source count is invalid")
    if not groups or len(groups) > MAX_CAPTURE_GROUPS:
        raise ManagerPageCaptureError("manager-page group count is invalid")

    expected_layer = [0] * len(groups)
    pages_per_source_group: list[int | None] = [None] * len(groups)
    previous_group = 0
    for source_index, source in enumerate(source_table):
        source_base = _positive_uint(
            source.source_base,
            name=f"sources[{source_index}].source_base",
            maximum=_UINT64_MAX,
        )
        source_pages = _positive_uint(
            source.source_pages,
            name=f"sources[{source_index}].source_pages",
            maximum=_UINT64_MAX,
        )
        stride = _positive_uint(
            source.source_page_stride_bytes,
            name=f"sources[{source_index}].source_page_stride_bytes",
            maximum=_UINT64_MAX,
        )
        width = _positive_uint(
            source.bytes_per_page,
            name=f"sources[{source_index}].bytes_per_page",
            maximum=_UINT32_MAX,
        )
        group_index = _uint(
            source.group_index,
            name=f"sources[{source_index}].group_index",
            maximum=_UINT32_MAX,
        )
        layer_ordinal = _uint(
            source.layer_ordinal,
            name=f"sources[{source_index}].layer_ordinal",
            maximum=_UINT32_MAX,
        )
        if group_index >= len(groups):
            raise ManagerPageCaptureError("source group index is not declared")
        if source_index and group_index < previous_group:
            raise ManagerPageCaptureError("sources must be ordered by group")
        previous_group = group_index
        if layer_ordinal != expected_layer[group_index]:
            raise ManagerPageCaptureError(
                "layer ordinals must be dense within each group"
            )
        expected_layer[group_index] += 1
        known_pages = pages_per_source_group[group_index]
        if known_pages is None:
            pages_per_source_group[group_index] = source_pages
        elif known_pages != source_pages:
            raise ManagerPageCaptureError(
                "layers in one group must share physical-page capacity"
            )
        if stride < width:
            raise ManagerPageCaptureError(
                "source page stride is smaller than its opaque page bytes"
            )
        required_bytes = (source_pages - 1) * stride + width
        if required_bytes > _UINT64_MAX or source_base + required_bytes > _UINT64_MAX:
            raise ManagerPageCaptureError("source address range overflows uint64")

    if any(layer_count == 0 for layer_count in expected_layer):
        raise ManagerPageCaptureError("every manager-page group needs a source")

    flattened: list[int] = []
    group_offsets: list[int] = []
    group_counts: list[int] = []
    for group_index, page_ids in enumerate(groups):
        if not page_ids:
            raise ManagerPageCaptureError(
                "every manager-page group needs at least one request page"
            )
        group_offsets.append(len(flattened))
        page_capacity = pages_per_source_group[group_index]
        assert page_capacity is not None
        for page_index, page_id in enumerate(page_ids):
            value = _uint(
                page_id,
                name=f"physical_pages_by_group[{group_index}][{page_index}]",
                maximum=_UINT32_MAX,
            )
            if value >= page_capacity:
                raise ManagerPageCaptureError(
                    "request page is outside its manager group"
                )
            flattened.append(value)
        group_counts.append(len(page_ids))
    if len(flattened) > _UINT32_MAX:
        raise ManagerPageCaptureError("physical-page table exceeds uint32")

    spans: list[ManagerPageCaptureSpan] = []
    cursor = 0
    for source_index, source in enumerate(source_table):
        page_count = group_counts[source.group_index]
        length = page_count * source.bytes_per_page
        end = cursor + length
        if end > capacity:
            raise ManagerPageCaptureError(
                "manager-page payload exceeds the bounded ring slot"
            )
        spans.append(
            ManagerPageCaptureSpan(
                source_index=source_index,
                group_index=source.group_index,
                physical_page_offset=group_offsets[source.group_index],
                page_count=page_count,
                destination_offset_bytes=cursor,
                length_bytes=length,
            )
        )
        cursor = end

    return ManagerPageCapturePlan(
        physical_pages=tuple(flattened),
        group_offsets=tuple(group_offsets),
        group_page_counts=tuple(group_counts),
        spans=tuple(spans),
        used_bytes=cursor,
    )


__all__ = [
    "MAX_CAPTURE_GROUPS",
    "MAX_CAPTURE_SOURCES",
    "ManagerPageCaptureError",
    "ManagerPageCapturePlan",
    "ManagerPageCaptureSpan",
    "ManagerPageSource",
    "PAGE_CAPTURE_CONTRACT_VERSION",
    "plan_manager_page_capture",
]
