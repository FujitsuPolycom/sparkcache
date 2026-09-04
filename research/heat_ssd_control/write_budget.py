"""Logical and host-observed write accounting plus SMART/Health parsing.

Models the byte quantities the production publication path produces
(immutable-object bytes, staged write-path bytes) and the device-side
NVMe Data Units Written counter, and derives write-amplification ratios
between them. No enforcement: the prototype reports without affecting
publication.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Sequence

from research.heat_ssd_control.heat_model import ResearchFormatError, require_digest

DUW_UNIT_BYTES = 512_000
"""One reported Data Units Written unit is 1,000 512-byte units, rounded up."""

_SMART_LOG_PAGE_BYTES = 512
_DUW_OFFSET = 0x30
_DUW_FIELD_BYTES = 16
_EVENT_KINDS = frozenset(
    {"commit", "alias_publication", "metadata_touch", "repair"}
)
_EVENT_SCHEMA = "sparkcache-research-write-event/v1"
_SAMPLE_SCHEMA = "sparkcache-research-ssd-sample/v1"
_HOUR_NS = 3_600_000_000_000
_DAY_NS = 86_400_000_000_000


@dataclass(frozen=True)
class WriteEvent:
    """One publication's byte contribution, as the ledger records it."""

    at_ns: int
    kind: str
    storage_key: str
    context_digest: str
    unique_object_bytes: int
    staged_write_bytes: int

    def __post_init__(self) -> None:
        if type(self.at_ns) is not int or self.at_ns < 0:
            raise ResearchFormatError("at_ns must be a non-negative nanosecond timestamp")
        if not isinstance(self.kind, str) or self.kind not in _EVENT_KINDS:
            raise ResearchFormatError(
                f"kind must be one of {sorted(_EVENT_KINDS)}, got {self.kind!r}"
            )
        require_digest(self.storage_key, "storage_key")
        require_digest(self.context_digest, "context_digest")
        if (
            type(self.unique_object_bytes) is not int
            or type(self.staged_write_bytes) is not int
            or self.unique_object_bytes < 0
            or self.staged_write_bytes < 0
        ):
            raise ResearchFormatError("event byte counts must be non-negative integers")
        if self.unique_object_bytes > self.staged_write_bytes:
            raise ResearchFormatError(
                "unique_object_bytes cannot exceed staged_write_bytes"
            )


def event_to_json(event: WriteEvent) -> str:
    """Serialize one write event under the ``write-event/v1`` schema."""
    if not isinstance(event, WriteEvent):
        raise ResearchFormatError("event must be a WriteEvent")
    document = {
        "schema": _EVENT_SCHEMA,
        "at_ns": event.at_ns,
        "kind": event.kind,
        "storage_key": event.storage_key,
        "context_digest": event.context_digest,
        "unique_object_bytes": event.unique_object_bytes,
        "staged_write_bytes": event.staged_write_bytes,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True)
class WriteBudget:
    """Staged-write interval limits; ``None`` means monitored, not limited."""

    hourly_limit_bytes: int | None = None
    daily_limit_bytes: int | None = None

    def __post_init__(self) -> None:
        for field in ("hourly_limit_bytes", "daily_limit_bytes"):
            value = getattr(self, field)
            if value is not None and (type(value) is not int or value < 0):
                raise ResearchFormatError(f"{field} must be None or a non-negative integer")


@dataclass(frozen=True)
class BudgetReport:
    """One window's totals against its budget."""

    window_start_ns: int
    window_end_ns: int
    unique_object_bytes: int
    staged_write_bytes: int
    limit_bytes: int | None
    events: int

    @property
    def exceeded(self) -> bool | None:
        if self.limit_bytes is None:
            return None
        return self.staged_write_bytes > self.limit_bytes

    @property
    def over_bytes(self) -> int:
        if self.limit_bytes is None:
            return 0
        return max(0, self.staged_write_bytes - self.limit_bytes)


class WriteLedger:
    """Accumulates ``WriteEvent`` records and folds them into UTC windows."""

    def __init__(self, events: Sequence[WriteEvent] = ()) -> None:
        self._events: list[WriteEvent] = []
        for event in events:
            self.add(event)

    def add(self, event: WriteEvent) -> None:
        if not isinstance(event, WriteEvent):
            raise ResearchFormatError("event must be a WriteEvent")
        if self._events and event.at_ns < self._events[-1].at_ns:
            raise ResearchFormatError("events must arrive in non-decreasing at_ns order")
        self._events.append(event)

    def hourly_reports(self, budget: WriteBudget | None = None) -> list[BudgetReport]:
        if budget is not None and not isinstance(budget, WriteBudget):
            raise ResearchFormatError("budget must be a WriteBudget or None")
        limit = budget.hourly_limit_bytes if budget else None
        return self._fold(_HOUR_NS, limit)

    def daily_reports(self, budget: WriteBudget | None = None) -> list[BudgetReport]:
        if budget is not None and not isinstance(budget, WriteBudget):
            raise ResearchFormatError("budget must be a WriteBudget or None")
        limit = budget.daily_limit_bytes if budget else None
        return self._fold(_DAY_NS, limit)

    def _fold(self, window_ns: int, limit: int | None) -> list[BudgetReport]:
        reports: dict[int, list[WriteEvent]] = {}
        for event in self._events:
            reports.setdefault(event.at_ns // window_ns, []).append(event)
        return [
            BudgetReport(
                window_start_ns=index * window_ns,
                window_end_ns=(index + 1) * window_ns,
                unique_object_bytes=sum(item.unique_object_bytes for item in events),
                staged_write_bytes=sum(item.staged_write_bytes for item in events),
                limit_bytes=limit,
                events=len(events),
            )
            for index, events in sorted(reports.items())
        ]


@dataclass(frozen=True)
class WriteAmplificationEstimate:
    """Ratios between logical retention, staging, and device counters.

    ``host_ratio`` is valid only over a controlled interval; with any
    concurrent non-cache device writes it is an upper bound.
    """

    unique_object_bytes: int | None
    staged_write_bytes: int | None
    host_written_bytes: int | None
    staging_ratio: float | None
    host_ratio: float | None


def write_amplification(
    *,
    unique_object_bytes: int | None = None,
    staged_write_bytes: int | None = None,
    host_written_bytes: int | None = None,
) -> WriteAmplificationEstimate:
    for name, value in (
        ("unique_object_bytes", unique_object_bytes),
        ("staged_write_bytes", staged_write_bytes),
        ("host_written_bytes", host_written_bytes),
    ):
        if value is not None and (type(value) is not int or value < 0):
            raise ResearchFormatError(f"{name} must be None or non-negative")
    staging_ratio = (
        staged_write_bytes / unique_object_bytes
        if staged_write_bytes is not None and unique_object_bytes
        else None
    )
    host_ratio = (
        host_written_bytes / unique_object_bytes
        if host_written_bytes is not None and unique_object_bytes
        else None
    )
    return WriteAmplificationEstimate(
        unique_object_bytes=unique_object_bytes,
        staged_write_bytes=staged_write_bytes,
        host_written_bytes=host_written_bytes,
        staging_ratio=staging_ratio,
        host_ratio=host_ratio,
    )


@dataclass(frozen=True)
class SmartHealthSample:
    """Parsed SMART/Health log page plus capture metadata."""

    at_ns: int
    device: str
    critical_warning: int
    composite_temperature_kelvin: int
    available_spare: int
    available_spare_threshold: int
    percentage_used: int
    data_units_read_units: int
    data_units_written_units: int
    data_units_written_reported: bool

    def __post_init__(self) -> None:
        if type(self.at_ns) is not int or self.at_ns < 0:
            raise ResearchFormatError("at_ns must be a non-negative integer")
        if not isinstance(self.device, str):
            raise ResearchFormatError("device must be a string")
        for field in (
            "critical_warning",
            "available_spare",
            "available_spare_threshold",
            "percentage_used",
        ):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value <= 0xFF:
                raise ResearchFormatError(f"{field} must be an unsigned 8-bit integer")
        if (
            type(self.composite_temperature_kelvin) is not int
            or not 0 <= self.composite_temperature_kelvin <= 0xFFFF
        ):
            raise ResearchFormatError(
                "composite_temperature_kelvin must be an unsigned 16-bit integer"
            )
        for field in ("data_units_read_units", "data_units_written_units"):
            value = getattr(self, field)
            if type(value) is not int or not 0 <= value < 1 << 128:
                raise ResearchFormatError(f"{field} must be an unsigned 128-bit integer")
        if type(self.data_units_written_reported) is not bool or (
            self.data_units_written_reported
            != (self.data_units_written_units != 0)
        ):
            raise ResearchFormatError(
                "data_units_written_reported must match the nonzero counter"
            )

    @property
    def data_units_written_bytes(self) -> int | None:
        if not self.data_units_written_reported:
            return None
        return self.data_units_written_units * DUW_UNIT_BYTES


def parse_smart_log_page(page: bytes, *, at_ns: int, device: str = "") -> SmartHealthSample:
    """Parse one raw 512-byte SMART/Health log page.

    Acquiring the page (an ``nvme smart-log`` output capture or an ioctl) is
    the caller's job; this function interprets the fixed-layout bytes. A
    reported Data Units Written value of 0 means "not reported" per the
    NVMe specification and is surfaced as ``data_units_written_reported``.
    """
    if not isinstance(page, (bytes, bytearray)) or len(page) != _SMART_LOG_PAGE_BYTES:
        raise ResearchFormatError(
            f"SMART/Health log page must be exactly {_SMART_LOG_PAGE_BYTES} bytes"
        )
    if type(at_ns) is not int or at_ns < 0:
        raise ResearchFormatError("at_ns must be a non-negative nanosecond timestamp")
    if not isinstance(device, str):
        raise ResearchFormatError("device must be a string")

    def le128(offset: int) -> int:
        chunk = bytes(page[offset : offset + _DUW_FIELD_BYTES])
        return int.from_bytes(chunk, "little")

    return SmartHealthSample(
        at_ns=at_ns,
        device=device,
        critical_warning=page[0x00],
        composite_temperature_kelvin=int.from_bytes(page[0x02:0x04], "little"),
        available_spare=page[0x04],
        available_spare_threshold=page[0x05],
        percentage_used=page[0x06],
        data_units_read_units=le128(0x20),
        data_units_written_units=le128(_DUW_OFFSET),
        data_units_written_reported=le128(_DUW_OFFSET) != 0,
    )


@dataclass(frozen=True)
class DuwDelta:
    """Change of Data Units Written between two same-device samples."""

    units: int
    bytes_est: int
    seconds: float
    rate_bytes_per_second: float | None


class DuwMonitor:
    """Compares two parsed SMART/Health samples of one device."""

    @staticmethod
    def delta(first: SmartHealthSample, second: SmartHealthSample) -> DuwDelta:
        if not isinstance(first, SmartHealthSample) or not isinstance(second, SmartHealthSample):
            raise ResearchFormatError("delta requires two SmartHealthSample inputs")
        if first.device and second.device and first.device != second.device:
            raise ResearchFormatError(
                f"samples carry conflicting device identifiers: {first.device!r} vs {second.device!r}"
            )
        if not first.data_units_written_reported or not second.data_units_written_reported:
            raise ResearchFormatError(
                "Data Units Written is not reported by one or both samples"
            )
        if second.at_ns < first.at_ns:
            raise ResearchFormatError("second sample precedes the first")
        if second.data_units_written_units < first.data_units_written_units:
            raise ResearchFormatError(
                "Data Units Written decreased between samples: device replacement, "
                "counter reset, or samples from different devices"
            )
        units = second.data_units_written_units - first.data_units_written_units
        seconds = (second.at_ns - first.at_ns) / 1_000_000_000
        if not seconds and units:
            raise ResearchFormatError(
                "Data Units Written changed without elapsed sample time"
            )
        bytes_est = units * DUW_UNIT_BYTES
        rate = bytes_est / seconds if seconds > 0 else None
        return DuwDelta(
            units=units,
            bytes_est=bytes_est,
            seconds=seconds,
            rate_bytes_per_second=rate,
        )


def sample_to_json(sample: SmartHealthSample) -> str:
    """Serialize one sample under the ``ssd-sample/v1`` schema."""
    if not isinstance(sample, SmartHealthSample):
        raise ResearchFormatError("sample must be a SmartHealthSample")
    document = {
        "schema": _SAMPLE_SCHEMA,
        "at_ns": sample.at_ns,
        "device": sample.device,
        "critical_warning": sample.critical_warning,
        "composite_temperature_kelvin": sample.composite_temperature_kelvin,
        "available_spare": sample.available_spare,
        "available_spare_threshold": sample.available_spare_threshold,
        "percentage_used": sample.percentage_used,
        "data_units_read_units": sample.data_units_read_units,
        "data_units_written_units": sample.data_units_written_units,
        "data_units_written_reported": sample.data_units_written_reported,
    }
    return json.dumps(document, sort_keys=True, separators=(",", ":"))


def utc_window_bounds(window_start_ns: int, window_ns: int) -> tuple[str, str]:
    """ISO-8601 UTC strings for a window's start and end (reporting aid)."""
    if (
        type(window_start_ns) is not int
        or type(window_ns) is not int
        or window_start_ns < 0
        or window_ns <= 0
    ):
        raise ResearchFormatError(
            "window_start_ns must be non-negative and window_ns must be positive"
        )
    start = datetime.fromtimestamp(window_start_ns / 1_000_000_000, tz=timezone.utc)
    end = datetime.fromtimestamp((window_start_ns + window_ns) / 1_000_000_000, tz=timezone.utc)
    return start.isoformat(), end.isoformat()
