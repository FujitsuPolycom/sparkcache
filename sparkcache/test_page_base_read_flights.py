from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from sparkcache.page_base_read_flights import (
    PageBaseReadCancelled,
    PageBaseReadError,
    PageBaseReadEvidence,
    PageBaseReadFlightKey,
    PageBaseReadFlights,
)


def _key(
    suffix: str = "a",
    *,
    encoded_bytes: int = 8,
    boundary_tokens: int = 131_072,
) -> PageBaseReadFlightKey:
    return PageBaseReadFlightKey(
        worker_generation="worker-generation",
        storage_mode="block_pages_v1",
        evidence=PageBaseReadEvidence(
            identity_storage_key="identity",
            base_context_digest=f"base-context-{suffix}",
            base_root_sha256=f"base-root-{suffix}",
            base_root_kind="page_snapshot",
            layout_sha256="layout",
            base_block_counts=(64, 64),
            base_boundary_tokens=boundary_tokens,
            base_encoded_bytes=encoded_bytes,
        ),
    )


def test_two_load_threads_share_one_base_across_sixteen_queued_results() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=9)
    request_ids = tuple(f"request-{index}" for index in range(16))
    registration = flights.register_cohort(key, request_ids)
    assert registration.member_ids == request_ids

    started = threading.Event()
    release = threading.Event()
    read_count = 0
    read_lock = threading.Lock()

    def read_base() -> bytes:
        nonlocal read_count
        with read_lock:
            read_count += 1
        started.set()
        assert release.wait(timeout=5)
        return b"base-data"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(flights.resolve, request_id, key, read_base)
            for request_id in request_ids
        ]
        assert started.wait(timeout=5)
        release.set()
        assert [future.result(timeout=5) for future in futures] == [
            b"base-data"
        ] * 16

    assert read_count == 1
    assert flights.snapshot().active_flights == 0
    assert flights.snapshot().retained_bytes == 0
    assert flights.snapshot().counters["base_reads_completed"] == 1
    assert flights.snapshot().counters["shared_results_acquired"] == 15
    assert flights.take_summaries() == (
        {
            "schema": "sparkcache-page-base-restore-flight/v1",
            "base_context_digest": "base-context-a",
            "base_root_sha256": "base-root-a",
            "participants": 16,
            "physical_base_reads": 1,
            "base_bytes": 9,
            "base_read_ms": pytest.approx(0, abs=5_000),
            "avoided_base_reads": 15,
            "cancelled_members": 0,
            "outcome": "verified",
            "worker_generation": "worker-generation",
            "storage_mode": "block_pages_v1",
        },
    )


def test_late_batches_and_singleton_join_pending_base_read() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=9)
    first = tuple(f"request-{index}" for index in range(8))
    second = tuple(f"request-{index}" for index in range(8, 15))
    singleton = "request-15"
    assert flights.register_cohort(key, first).member_ids == first

    started = threading.Event()
    release = threading.Event()
    read_count = 0

    def read_base() -> bytes:
        nonlocal read_count
        read_count += 1
        started.set()
        assert release.wait(timeout=5)
        return b"base-data"

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(flights.resolve, request_id, key, read_base)
            for request_id in first
        ]
        assert started.wait(timeout=5)
        late = flights.register_cohort(key, second)
        assert late.member_ids == second
        assert late.flight_state == "reading"
        one = flights.register_cohort(key, (singleton,))
        assert one.member_ids == (singleton,)
        assert one.flight_state == "reading"
        futures.extend(
            executor.submit(flights.resolve, request_id, key, read_base)
            for request_id in (*second, singleton)
        )
        release.set()
        assert all(future.result(timeout=5) == b"base-data" for future in futures)

    assert read_count == 1
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["participants"] == 16
    assert summary[0]["avoided_base_reads"] == 15


def test_reader_bytearray_is_copied_before_followers_observe_it() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=18)
    flights.register_cohort(key, ("leader", "follower"))
    source = bytearray(b"authenticated-base")

    leader_result = flights.resolve("leader", key, lambda: source)
    source[:] = b"x" * len(source)
    follower_result = flights.resolve("follower", key, lambda: b"wrong")

    assert isinstance(leader_result, bytes)
    assert leader_result == b"authenticated-base"
    assert follower_result == b"authenticated-base"


def test_reader_length_must_equal_authenticated_geometry() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=8)
    flights.register_cohort(key, ("leader", "follower"))

    with pytest.raises(PageBaseReadError, match="length differs"):
        flights.resolve("leader", key, lambda: b"too-long!")
    with pytest.raises(PageBaseReadError, match="length differs"):
        flights.resolve("follower", key, lambda: b"wrong")
    assert flights.take_summaries()[0]["outcome"] == "recompute"


def test_cancellation_is_request_local_and_reported_cumulatively() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=4)
    flights.register_cohort(key, ("leader", "cancelled", "remaining"))
    started = threading.Event()
    release = threading.Event()

    def read_base() -> bytes:
        started.set()
        assert release.wait(timeout=5)
        return b"base"

    with ThreadPoolExecutor(max_workers=3) as executor:
        leader = executor.submit(flights.resolve, "leader", key, read_base)
        cancelled = executor.submit(
            flights.resolve,
            "cancelled",
            key,
            lambda: b"wrong",
        )
        assert started.wait(timeout=5)
        assert flights.cancel("cancelled")
        with pytest.raises(PageBaseReadCancelled):
            cancelled.result(timeout=5)
        release.set()
        remaining = executor.submit(
            flights.resolve,
            "remaining",
            key,
            lambda: b"wrong",
        )
        assert leader.result(timeout=5) == b"base"
        assert remaining.result(timeout=5) == b"base"

    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["cancelled_members"] == 1
    assert summary[0]["participants"] == 3


def test_leader_failure_rejects_only_matching_members() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=4)
    flights.register_cohort(key, ("leader", "follower"))

    with pytest.raises(PageBaseReadError, match="corrupt base"):
        flights.resolve(
            "leader",
            key,
            lambda: (_ for _ in ()).throw(ValueError("corrupt base")),
        )
    with pytest.raises(PageBaseReadError, match="corrupt base"):
        flights.resolve("follower", key, lambda: b"wrong")

    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["outcome"] == "recompute"
    assert summary[0]["physical_base_reads"] == 1


def test_member_limit_bypasses_excess_requests_without_waiting() -> None:
    flights = PageBaseReadFlights(max_members=4)
    key = _key(encoded_bytes=4)
    registration = flights.register_cohort(
        key,
        tuple(f"request-{index}" for index in range(6)),
    )
    assert registration.member_ids == (
        "request-0",
        "request-1",
        "request-2",
        "request-3",
    )
    reads = 0

    def read_base() -> bytes:
        nonlocal reads
        reads += 1
        return b"base"

    for index in range(6):
        assert flights.resolve(f"request-{index}", key, read_base) == b"base"
    assert reads == 3
    assert flights.snapshot().counters["member_limit_bypasses"] == 2


def test_member_limit_bounds_sequential_late_joiners_and_flight_lifetime() -> None:
    flights = PageBaseReadFlights(max_members=4)
    key = _key(encoded_bytes=4)
    flights.register_cohort(key, ("leader", "anchor"))
    assert flights.resolve("leader", key, lambda: b"base") == b"base"
    assert flights.register_cohort(key, ("late-a",)).member_ids == ("late-a",)
    assert flights.resolve("late-a", key, lambda: b"wrong") == b"base"
    assert flights.register_cohort(key, ("late-b",)).member_ids == ("late-b",)
    assert flights.resolve("late-b", key, lambda: b"wrong") == b"base"

    assert not flights.register_cohort(key, ("late-overflow",)).member_ids
    assert flights.resolve("late-overflow", key, lambda: b"independent") == b"independent"
    assert flights.resolve("anchor", key, lambda: b"wrong") == b"base"
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["participants"] == 4
    assert summary[0]["avoided_base_reads"] == 3


def test_declared_byte_bounds_bypass_before_reading() -> None:
    gib = 1024**3
    flights = PageBaseReadFlights()
    fits_glm_base = _key("glm", encoded_bytes=813 * 1024**2)
    assert flights.register_cohort(fits_glm_base, ("glm-a", "glm-b")).member_ids

    second_large = _key("other", encoded_bytes=813 * 1024**2)
    assert not flights.register_cohort(second_large, ("other-a", "other-b")).member_ids
    over_per_flight = _key("over", encoded_bytes=gib + 1)
    assert not flights.register_cohort(over_per_flight, ("over-a", "over-b")).member_ids

    snapshot = flights.snapshot()
    assert snapshot.counters["total_byte_limit_bypasses"] == 2
    assert snapshot.counters["per_flight_byte_limit_bypasses"] == 2
    invalid = _key("invalid", encoded_bytes=0)
    assert not flights.register_cohort(invalid, ("invalid-a", "invalid-b")).member_ids
    assert flights.snapshot().counters["invalid_declared_byte_bypasses"] == 2
    flights.finish("glm-a")
    flights.finish("glm-b")


def test_close_reports_every_abandoned_member_and_releases_after_wakeup() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=4)
    flights.register_cohort(key, ("leader", "follower"))
    flights.close()
    assert flights.snapshot().active_flights == 0
    assert flights.snapshot().registered_members == 0
    assert flights.snapshot().retained_bytes == 0
    for request_id in ("leader", "follower"):
        with pytest.raises(PageBaseReadCancelled):
            flights.resolve(request_id, key, lambda: b"base")
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["cancelled_members"] == 2
    assert summary[0]["physical_base_reads"] == 0
    assert summary[0]["base_bytes"] == 0


def test_close_releases_ready_buffer_when_a_member_never_resolves() -> None:
    flights = PageBaseReadFlights()
    key = _key(encoded_bytes=4)
    flights.register_cohort(key, ("leader", "pending"))
    assert flights.resolve("leader", key, lambda: b"base") == b"base"
    assert flights.snapshot().retained_bytes == 4

    flights.close()

    assert flights.snapshot().active_flights == 0
    assert flights.snapshot().registered_members == 0
    assert flights.snapshot().retained_bytes == 0
    summary = flights.take_summaries()
    assert len(summary) == 1
    assert summary[0]["outcome"] == "cancelled"
    assert summary[0]["physical_base_reads"] == 1
    assert summary[0]["base_bytes"] == 4
    with pytest.raises(PageBaseReadCancelled):
        flights.resolve("pending", key, lambda: b"wrong")


def test_unrelated_base_can_complete_while_first_base_is_pending() -> None:
    flights = PageBaseReadFlights(max_bytes_per_flight=16, max_bytes_total=64)
    blocked_key = _key("blocked", encoded_bytes=7)
    unrelated_key = _key("unrelated", encoded_bytes=9)
    flights.register_cohort(blocked_key, ("blocked-a", "blocked-b"))
    flights.register_cohort(unrelated_key, ("unrelated-a", "unrelated-b"))
    started = threading.Event()
    release = threading.Event()

    def blocked_read() -> bytes:
        started.set()
        assert release.wait(timeout=5)
        return b"blocked"

    with ThreadPoolExecutor(max_workers=3) as executor:
        blocked = executor.submit(
            flights.resolve,
            "blocked-a",
            blocked_key,
            blocked_read,
        )
        blocked_follower = executor.submit(
            flights.resolve,
            "blocked-b",
            blocked_key,
            lambda: b"wrong",
        )
        assert started.wait(timeout=5)
        assert (
            flights.resolve(
                "unrelated-a",
                unrelated_key,
                lambda: b"unrelated",
            )
            == b"unrelated"
        )
        assert (
            flights.resolve(
                "unrelated-b",
                unrelated_key,
                lambda: b"wrong",
            )
            == b"unrelated"
        )
        release.set()
        assert blocked.result(timeout=5) == b"blocked"
        assert blocked_follower.result(timeout=5) == b"blocked"

    summaries = flights.take_summaries()
    assert len(summaries) == 2
    assert {summary["base_root_sha256"] for summary in summaries} == {
        "base-root-blocked",
        "base-root-unrelated",
    }
    assert sum(int(summary["physical_base_reads"]) for summary in summaries) == 2


def test_flight_count_limit_bypasses_an_unrelated_cohort() -> None:
    flights = PageBaseReadFlights(
        max_flights=1,
        max_bytes_per_flight=16,
        max_bytes_total=32,
    )
    first = _key("first", encoded_bytes=4)
    second = _key("second", encoded_bytes=4)
    assert flights.register_cohort(first, ("first-a", "first-b")).member_ids
    assert not flights.register_cohort(second, ("second-a", "second-b")).member_ids
    assert flights.resolve("second-a", second, lambda: b"base") == b"base"
    assert flights.snapshot().counters["flight_limit_bypasses"] == 2
    flights.finish("first-a")
    flights.finish("first-b")
    assert flights.snapshot().active_flights == 0
