"""GPU-free regression coverage for the isolated heat and SSD prototype."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from research.heat_ssd_control.heat_model import (
    ChunkLedger,
    HeatKey,
    HitRing,
    HitRingConfig,
    PublishedContext,
    ResearchFormatError,
    context_reports,
    recomputation_tokens_avoided,
)
from research.heat_ssd_control.shadow_admission import ShadowConfig, TinyLFUShadow
from research.heat_ssd_control.write_budget import (
    DUW_UNIT_BYTES,
    DuwMonitor,
    WriteBudget,
    WriteEvent,
    WriteLedger,
    event_to_json,
    parse_smart_log_page,
    sample_to_json,
    write_amplification,
)


def _digest(character: str) -> str:
    return character * 64


def _key(context: str, storage: str = "a") -> HeatKey:
    return HeatKey(_digest(storage), _digest(context))


def _context(
    context: str,
    chunks: tuple[tuple[str, int], ...],
    *,
    manifest_bytes: int = 10,
    segments: tuple[tuple[str, int], ...] = (),
    storage: str = "a",
    chunk_token_counts: tuple[int, ...] | None = None,
) -> PublishedContext:
    return PublishedContext(
        key=_key(context, storage),
        chunk_digests=tuple(_digest(digest) for digest, _size in chunks),
        chunk_bytes=tuple(size for _digest_character, size in chunks),
        chunk_token_counts=(
            chunk_token_counts
            if chunk_token_counts is not None
            else (256,) * len(chunks)
        ),
        manifest_bytes=manifest_bytes,
        segment_digests=tuple(_digest(digest) for digest, _size in segments),
        segment_bytes=tuple(size for _digest_character, size in segments),
    )


def test_importing_sparkcache_does_not_load_offline_research_modules() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sparkcache,sys; "
                "assert not any(name == 'research.heat_ssd_control' or "
                "name.startswith('research.heat_ssd_control.') "
                "for name in sys.modules)"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_production_modules_do_not_import_research_package() -> None:
    package_root = Path(__file__).resolve().parents[2] / "sparkcache"
    offenders: list[str] = []
    for path in package_root.rglob("*.py"):
        relative = path.relative_to(package_root)
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) and any(
                alias.name == "research.heat_ssd_control"
                or alias.name.startswith("research.heat_ssd_control.")
                for alias in node.names
            ):
                offenders.append(str(relative))
            if isinstance(node, ast.ImportFrom) and node.module and (
                node.module == "research.heat_ssd_control"
                or node.module.startswith("research.heat_ssd_control.")
            ):
                offenders.append(str(relative))
    assert offenders == []


def test_hit_ring_saturates_and_decay_counts_saturated_accesses() -> None:
    key = _key("b")
    ring = HitRing(HitRingConfig(capacity=8, decay_window=256, decay_shift=1))
    for _ in range(255):
        ring.record_hit(key)
    assert ring.estimate(key) == 255

    # The 256th access starts a new decayed epoch even though the old counter
    # was saturated: 255 >> 1, followed by the in-flight increment.
    assert ring.record_hit(key) == 128
    assert json.loads(ring.snapshot())["increments_since_decay"] == 0


def test_recomputation_tokens_avoided_uses_the_verified_restore_span() -> None:
    assert recomputation_tokens_avoided(131_072, 4_096) == 126_976
    with pytest.raises(ResearchFormatError, match="num_computed_tokens"):
        recomputation_tokens_avoided(4_096, 4_097)


def test_hit_ring_snapshot_round_trip_and_schema_rejection() -> None:
    key = _key("b")
    ring = HitRing(HitRingConfig(capacity=8, decay_window=16, decay_shift=1))
    ring.record_hit(key)
    restored = HitRing.from_json(ring.snapshot())
    assert restored.estimate(key) == 1

    document = json.loads(ring.snapshot())
    document["counts_hex"] = "00"
    with pytest.raises(ResearchFormatError, match="counts_hex"):
        HitRing.from_json(json.dumps(document))
    with pytest.raises(ResearchFormatError, match="JSON"):
        HitRing.from_json(None)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: HeatKey("A" * 64, _digest("b")),
        lambda: HitRingConfig(capacity=3),
        lambda: HitRingConfig(capacity=True),
        lambda: HitRingConfig(decay_window=0),
        lambda: HitRingConfig(decay_window=2, decay_shift=2),
        lambda: ChunkLedger(chunk_tokens=True),
    ],
)
def test_heat_model_rejects_malformed_bounded_inputs(factory: object) -> None:
    with pytest.raises(ResearchFormatError):
        factory()  # type: ignore[operator]


def test_chunk_ledger_reports_shared_trunks_and_exclusive_bytes() -> None:
    first = _context("b", (("d", 100), ("e", 200)), segments=(("f", 20),))
    second = _context(
        "c",
        (("d", 100), ("0", 300)),
        manifest_bytes=11,
        segments=(("f", 20),),
    )
    ledger = ChunkLedger(chunk_tokens=256)
    ledger.publish(first)
    ledger.publish(second)

    first_report = ledger.report(first.key)
    assert first_report.shared_chunk_count == 1
    assert first_report.shared_tokens == 256
    assert first_report.retained_shared_bytes == 100
    assert first_report.marginal_bytes == 210
    assert first_report.encoded_bytes == 330
    assert set(context_reports(ledger)) == {first.key, second.key}

    ledger.remove(second.key)
    unshared = ledger.report(first.key)
    assert unshared.shared_chunk_count == 0
    assert unshared.marginal_bytes == 330


def test_chunk_ledger_uses_exact_shared_terminal_span() -> None:
    first = _context("b", (("d", 100),), chunk_token_counts=(128,))
    second = _context("c", (("d", 100),), chunk_token_counts=(128,))
    ledger = ChunkLedger(chunk_tokens=256)
    ledger.publish(first)
    ledger.publish(second)
    assert ledger.report(first.key).shared_tokens == 128


def test_chunk_ledger_rejected_publication_is_atomic() -> None:
    existing = _context("b", (("d", 100),))
    conflicting = _context("c", (("e", 50), ("d", 999)))
    independent = _context("0", (("e", 50),))
    ledger = ChunkLedger()
    ledger.publish(existing)

    with pytest.raises(ResearchFormatError, match="byte count changed"):
        ledger.publish(conflicting)

    ledger.publish(independent)
    assert ledger.report(independent.key).marginal_bytes == 60


def test_published_context_rejects_duplicate_objects_and_non_integer_bytes() -> None:
    with pytest.raises(ResearchFormatError, match="unique"):
        _context("b", (("d", 100), ("d", 100)))
    with pytest.raises(ResearchFormatError, match="integers"):
        _context("b", (("d", True),))


def test_tinylfu_shadow_replays_a_generator_with_bounded_rollup() -> None:
    first, second, third = _key("b"), _key("c"), _key("d")
    shadow = TinyLFUShadow(
        ShadowConfig(
            window_capacity=1,
            main_capacity=1,
            ring=HitRingConfig(capacity=32, decay_window=64, decay_shift=1),
        )
    )
    report = shadow.evaluate_trace(key for key in (first, second, first, third, first))
    assert report.requests == 5
    assert report.main_hits == 2
    assert report.window_hits == 0
    assert report.misses == 3
    assert report.admitted == 1
    assert report.rejected == 1
    assert report.final_resident == 2
    assert report.hit_rate == pytest.approx(0.4)


def test_write_windows_budget_staged_bytes_not_retained_bytes() -> None:
    first = WriteEvent(
        at_ns=0,
        kind="commit",
        storage_key=_digest("a"),
        context_digest=_digest("b"),
        unique_object_bytes=60,
        staged_write_bytes=120,
    )
    second = WriteEvent(
        at_ns=3_600_000_000_000,
        kind="alias_publication",
        storage_key=_digest("a"),
        context_digest=_digest("c"),
        unique_object_bytes=10,
        staged_write_bytes=10,
    )
    ledger = WriteLedger((first, second))
    reports = ledger.hourly_reports(WriteBudget(hourly_limit_bytes=100))
    assert len(reports) == 2
    assert reports[0].unique_object_bytes == 60
    assert reports[0].staged_write_bytes == 120
    assert reports[0].exceeded is True
    assert reports[0].over_bytes == 20
    assert reports[1].exceeded is False
    assert ledger.daily_reports()[0].exceeded is None
    assert json.loads(event_to_json(first))["schema"] == (
        "sparkcache-research-write-event/v1"
    )


def test_write_event_rejects_non_integral_and_impossible_byte_counts() -> None:
    common = {
        "at_ns": 0,
        "kind": "commit",
        "storage_key": _digest("a"),
        "context_digest": _digest("b"),
    }
    with pytest.raises(ResearchFormatError, match="integers"):
        WriteEvent(**common, unique_object_bytes=True, staged_write_bytes=1)
    with pytest.raises(ResearchFormatError, match="cannot exceed"):
        WriteEvent(**common, unique_object_bytes=2, staged_write_bytes=1)


def test_write_amplification_keeps_missing_denominators_explicit() -> None:
    estimate = write_amplification(
        unique_object_bytes=100,
        staged_write_bytes=150,
        host_written_bytes=250,
    )
    assert estimate.staging_ratio == 1.5
    assert estimate.host_ratio == 2.5

    missing = write_amplification(unique_object_bytes=0, staged_write_bytes=10)
    assert missing.staging_ratio is None
    assert missing.host_ratio is None


def _smart_page(*, written_units: int, read_units: int = 0) -> bytes:
    page = bytearray(512)
    page[0x20:0x30] = read_units.to_bytes(16, "little")
    page[0x30:0x40] = written_units.to_bytes(16, "little")
    page[0x02:0x04] = (300).to_bytes(2, "little")
    page[0x04] = 99
    page[0x05] = 10
    page[0x06] = 7
    return bytes(page)


def test_data_units_written_parsing_delta_and_json_schema() -> None:
    first = parse_smart_log_page(
        _smart_page(written_units=5), at_ns=1_000_000_000, device="/dev/nvme0"
    )
    second = parse_smart_log_page(
        _smart_page(written_units=9), at_ns=3_000_000_000, device="/dev/nvme0"
    )
    delta = DuwMonitor.delta(first, second)
    assert delta.units == 4
    assert delta.bytes_est == 4 * DUW_UNIT_BYTES
    assert delta.seconds == 2
    assert delta.rate_bytes_per_second == 2 * DUW_UNIT_BYTES

    document = json.loads(sample_to_json(second))
    assert document["schema"] == "sparkcache-research-ssd-sample/v1"
    assert document["data_units_written_units"] == 9


def test_data_units_written_rejects_unreported_reset_and_device_mismatch() -> None:
    unreported = parse_smart_log_page(_smart_page(written_units=0), at_ns=0)
    reported = parse_smart_log_page(_smart_page(written_units=1), at_ns=1)
    with pytest.raises(ResearchFormatError, match="not reported"):
        DuwMonitor.delta(unreported, reported)

    high = parse_smart_log_page(
        _smart_page(written_units=9), at_ns=0, device="/dev/nvme0"
    )
    low = parse_smart_log_page(
        _smart_page(written_units=8), at_ns=1, device="/dev/nvme0"
    )
    with pytest.raises(ResearchFormatError, match="decreased"):
        DuwMonitor.delta(high, low)

    other = parse_smart_log_page(
        _smart_page(written_units=10), at_ns=1, device="/dev/nvme1"
    )
    with pytest.raises(ResearchFormatError, match="conflicting"):
        DuwMonitor.delta(high, other)


def test_write_ledger_rejects_wrong_type_and_out_of_order_events() -> None:
    ledger = WriteLedger()
    with pytest.raises(ResearchFormatError, match="WriteEvent"):
        ledger.add(object())  # type: ignore[arg-type]

    later = WriteEvent(
        at_ns=2,
        kind="commit",
        storage_key=_digest("a"),
        context_digest=_digest("b"),
        unique_object_bytes=1,
        staged_write_bytes=1,
    )
    earlier = WriteEvent(
        at_ns=1,
        kind="commit",
        storage_key=_digest("a"),
        context_digest=_digest("c"),
        unique_object_bytes=1,
        staged_write_bytes=1,
    )
    ledger.add(later)
    with pytest.raises(ResearchFormatError, match="non-decreasing"):
        ledger.add(earlier)
