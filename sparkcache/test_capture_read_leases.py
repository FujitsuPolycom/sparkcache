"""D21: exact source leases survive cleanup until distinct copy acknowledgements."""

from types import SimpleNamespace as NS
import threading

import pytest

from sparkcache.capture_read_leases import CaptureReadLeases
from sparkcache.streaming.test_manager_page_runtime import (
    FakeConnector,
    FakeRing,
    _plan,
)
from sparkcache.streaming.manager_page_runtime import ManagerPageCaptureRuntime


class Pool:
    def __init__(self):
        self.blocks = [
            NS(block_id=i, is_null=i == 0, ref_cnt=1, block_hash=f"hash-{i}")
            for i in range(128)
        ]

    def touch(self, blocks):
        for block in blocks:
            block.ref_cnt += 1

    def free_blocks(self, blocks):
        for block in blocks:
            block.ref_cnt -= 1
            assert block.ref_cnt >= 0


def test_d21_deduplicated_sources_survive_request_free_and_duplicate_rank_ack():
    pool = Pool()
    leases = CaptureReadLeases(pool, ranks=4, max_jobs=1)
    job = leases.reserve([2, 5, 2])
    assert job and [pool.blocks[i].ref_cnt for i in (2, 5)] == [2, 2]
    assert leases.reserve([7]) is None
    pool.free_blocks(pool.blocks[i] for i in (2, 5))
    assert all(pool.blocks[i].ref_cnt == 1 for i in (2, 5))
    assert not leases.complete(job, [0])
    assert not leases.complete(job, [0, 1, 2])
    assert not leases.complete(
        "another-aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1", [0, 1, 2, 3]
    )
    assert leases and all(pool.blocks[i].ref_cnt == 1 for i in (2, 5))
    assert leases.complete(job, [3])
    assert not leases and all(pool.blocks[i].ref_cnt == 0 for i in (2, 5))
    assert not leases.complete(job, [0, 1, 2, 3])


@pytest.mark.parametrize("ids", [[0], [-1], [128], [True]])
def test_d21_bad_source_ids_never_take_references(ids):
    pool = Pool()
    leases = CaptureReadLeases(pool, ranks=4, max_jobs=1)
    with pytest.raises(ValueError):
        leases.reserve(ids)
    assert not leases and all(block.ref_cnt == 1 for block in pool.blocks)


def test_d21_mutable_unhashed_sources_are_not_capture_candidates():
    pool = Pool()
    pool.blocks[5].block_hash = None
    leases = CaptureReadLeases(pool, ranks=4, max_jobs=1)
    with pytest.raises(ValueError, match="immutable"):
        leases.reserve([2, 5])
    assert not leases and pool.blocks[2].ref_cnt == 1


def test_d21_invalid_completion_rank_preserves_all_references():
    pool = Pool()
    leases = CaptureReadLeases(pool, ranks=4, max_jobs=1)
    job = leases.reserve([2])
    with pytest.raises(ValueError, match="physical rank"):
        leases.complete(job, [0, 4])
    assert leases and pool.blocks[2].ref_cnt == 2


class JobConnector(FakeConnector):
    def __init__(self):
        super().__init__()
        self.read_jobs = []
        self.read_done = threading.Event()
        self.failed_jobs = []
        self.read_failed = threading.Event()

    @staticmethod
    def _protect_capture_publication_base(plan):
        return plan

    def _capture_read_completed(self, job):
        self.read_jobs.append(job)
        self.read_done.set()

    def _capture_read_failed(self, job):
        self.failed_jobs.append(job)
        self.read_failed.set()


def test_d21_job_completes_at_read_fence_before_request_or_file_completion():
    connector = JobConnector()
    ring = FakeRing(b"aabbxyz")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    try:
        assert runtime.submit(plan, producer_stream=19, producer_ready=object())
        assert not connector.read_jobs
        assert runtime.take_finished({plan.request_id}) == set()
        ring.ready.set()
        assert connector.read_done.wait(2)
        assert connector.read_jobs == ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"]
        assert connector.completed_event.wait(2)
        assert runtime.take_finished({plan.request_id}) == set()
    finally:
        runtime.shutdown()


def test_d21_preemption_never_waits_for_job_copy_and_discards_publication():
    connector = JobConnector()
    ring = FakeRing(b"aabbxyz")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    try:
        assert runtime.submit(plan, producer_stream=19, producer_ready=object())
        runtime.preempt(plan.request_id)
        assert ring.drained == []
        ring.ready.set()
        assert connector.read_done.wait(2)
        assert runtime.wait_idle(2)
        assert not connector.completed
        assert connector.read_jobs == ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"]
    finally:
        runtime.shutdown()


def test_d21_uncertain_submit_drain_withholds_job_acknowledgement():
    class FailedRing(FakeRing):
        def submit(self, **kwargs):
            raise RuntimeError("submission status unavailable")

        def drain_context(self, sequence):
            raise RuntimeError("copy-event status unavailable")

    connector = JobConnector()
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=FailedRing(b""),
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    assert runtime.submit(plan, producer_stream=19, producer_ready=object())
    assert connector.read_failed.wait(2)
    runtime.finish_failed_job(plan)
    assert not connector.read_jobs and runtime.status()["ownership_uncertain"]
    assert not runtime.submit(plan, producer_stream=19, producer_ready=object())
    assert not connector.read_jobs, "repeated failed job acknowledged an uncertain read"


def test_d21_partial_enqueue_failure_drains_off_callback_thread():
    class DelayedDrain(FakeRing):
        def __init__(self):
            super().__init__(b"")
            self.draining = threading.Event()
            self.release_drain = threading.Event()

        def submit(self, **kwargs):
            raise RuntimeError("partial enqueue")

        def drain_context(self, sequence):
            self.draining.set()
            assert self.release_drain.wait(2)

    connector = JobConnector()
    ring = DelayedDrain()
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    try:
        assert runtime.submit(plan, producer_stream=19, producer_ready=object())
        assert ring.draining.wait(2)
        runtime.finish_failed_job(plan)
        assert not connector.read_jobs
        following = _plan("another-request")
        following.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2"
        assert runtime.submit(following, producer_stream=19, producer_ready=object())
        runtime.preempt(following.request_id)
        assert not connector.read_jobs
        assert runtime.status()["pending_requests"] == 2
        ring.release_drain.set()
        assert runtime.wait_idle(2)
        assert set(connector.read_jobs) == {
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2",
        }
    finally:
        ring.release_drain.set()
        runtime.shutdown()


def test_d21_uncertain_read_quarantines_refs_without_idle_spin():
    pool = Pool()
    leases = CaptureReadLeases(pool, ranks=4, max_jobs=2)
    job = leases.reserve([2])
    leases.quarantine(job, [0])
    assert leases.disabled and not leases
    assert leases.reserve([3]) is None
    assert pool.blocks[2].ref_cnt == 2
    assert not leases.complete(job, [1, 2, 3])
    assert pool.blocks[2].ref_cnt == 2
    assert leases.complete(job, [0])
    assert pool.blocks[2].ref_cnt == 1


@pytest.mark.parametrize("failure", ["backend", "invalid-ticket"])
def test_d21_native_wrapper_submission_and_recovery_cannot_block_callbacks(failure):
    """Exercise the real Python wrapper while its backend holds the ring lock."""
    from sparkcache.streaming.manager_page_native_ring import NativeManagerPageRing
    from sparkcache.streaming.native_ring import NativeStatus, RawTicket
    from sparkcache.streaming.test_manager_page_native_ring import (
        FakePageBackend,
        _config,
        _sources,
    )

    class BlockingBackend(FakePageBackend):
        def __init__(self):
            super().__init__()
            self.entered = threading.Event()
            self.unblock = threading.Event()
            self.thread_id = None

        def block_recovery(self):
            self.thread_id = threading.get_ident()
            self.entered.set()
            assert self.unblock.wait(3)

        def submit_pages(self, **kwargs):
            if failure == "backend":
                # The CUDA implementation can synchronize here before returning
                # its error status. No Python exception handler has run yet.
                self.block_recovery()
                return NativeStatus.CUDA_ERROR, None
            super().submit_pages(**kwargs)
            return NativeStatus.OK, RawTicket(0, 0)

        def drain_context(self, context_sequence):
            if failure == "invalid-ticket":
                # NativeManagerPageRing calls this before raising about an
                # invalid ticket, while retaining its own internal lock.
                self.block_recovery()
            return super().drain_context(context_sequence)

    backend = BlockingBackend()
    ring = NativeManagerPageRing(_config(), backend=backend)
    ring.configure_sources(_sources(), group_count=2)
    connector = JobConnector()
    ready = object()
    observed_events = []

    def stream_after_event(event):
        observed_events.append(event)
        return 19

    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_poll_seconds=0.001,
        progress_thread_initializer=lambda: None,
        job_stream_factory=stream_after_event,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    following = _plan("queued")
    following.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2"
    callbacks_done = threading.Event()
    try:
        assert runtime.submit(plan, producer_stream=19, producer_ready=ready)
        assert backend.entered.wait(2)
        assert backend.thread_id != threading.get_ident()
        assert observed_events == [ready]

        def callbacks():
            assert runtime.status()["pending_requests"] == 1
            runtime.preempt(plan.request_id)
            runtime.finish_failed_job(plan)
            assert runtime.submit(following, producer_stream=19, producer_ready=ready)
            runtime.preempt(following.request_id)
            runtime.finish_failed_job(following)
            callbacks_done.set()

        caller = threading.Thread(target=callbacks)
        caller.start()
        assert callbacks_done.wait(1), "callbacks waited on native submission/recovery"
        assert not connector.read_jobs
        backend.unblock.set()
        caller.join(2)
        assert runtime.wait_idle(2)
        assert set(connector.read_jobs) == {
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2",
        }
        assert observed_events == [ready], "cancelled queued job reached submission"
        assert not backend.entries and not connector.completed
    finally:
        backend.unblock.set()
        runtime.shutdown()


def test_d21_shutdown_waits_for_submitting_job_before_releasing_sources():
    initialized = threading.Event()
    proceed = threading.Event()
    connector = JobConnector()
    ring = FakeRing(b"")

    def initialize():
        initialized.set()
        assert proceed.wait(3)

    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_thread_initializer=initialize,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    assert runtime.submit(plan, producer_stream=19, producer_ready=object())
    assert initialized.wait(2)
    stopped = threading.Event()

    def shutdown():
        runtime.shutdown()
        stopped.set()

    thread = threading.Thread(target=shutdown)
    thread.start()
    assert not stopped.wait(0.05) and not connector.read_jobs
    proceed.set()
    assert stopped.wait(2)
    thread.join(2)
    assert not ring.submissions and connector.read_jobs == [
        "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    ]


def test_d21_duplicate_during_shutdown_drain_cannot_acknowledge_owned_read():
    class BlockingDrain(FakeRing):
        def __init__(self):
            super().__init__(b"")
            self.submitted = threading.Event()
            self.draining = threading.Event()
            self.retire = threading.Event()

        def submit(self, **kwargs):
            ticket = super().submit(**kwargs)
            self.submitted.set()
            return ticket

        def drain_context(self, sequence):
            self.draining.set()
            assert self.retire.wait(3)
            super().drain_context(sequence)

    connector = JobConnector()
    ring = BlockingDrain()
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1"
    assert runtime.submit(plan, producer_stream=19, producer_ready=object())
    assert ring.submitted.wait(2)
    stopped = threading.Event()

    def shutdown():
        runtime.shutdown()
        stopped.set()

    thread = threading.Thread(target=shutdown)
    thread.start()
    try:
        assert ring.draining.wait(2)
        assert not runtime.submit(plan, producer_stream=19, producer_ready=object())
        runtime.finish_failed_job(plan)
        assert not connector.read_jobs
        unrelated = _plan("unsubmitted")
        unrelated.capture_job_id = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2"
        assert not runtime.submit(
            unrelated, producer_stream=19, producer_ready=object()
        )
        assert connector.read_jobs == ["aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2"]
        ring.retire.set()
        assert stopped.wait(2)
        thread.join(2)
        assert connector.read_jobs == [
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:2",
            "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa:1",
        ]
    finally:
        ring.retire.set()
        thread.join(2)


def test_d21_retired_job_replay_and_epoch_reset_never_start_native_reads():
    connector = JobConnector()
    ring = FakeRing(b"aabbxyz")
    ring.ready.set()
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = "a" * 32 + ":3"
    try:
        assert runtime.submit(plan, producer_stream=19, producer_ready=object())
        assert connector.read_done.wait(2) and runtime.wait_idle(2)
        assert len(ring.submissions) == 1
        for job in (plan.capture_job_id, "a" * 32 + ":2", "b" * 32 + ":4"):
            replay = _plan("different-request")
            replay.capture_job_id = job
            assert not runtime.submit(
                replay, producer_stream=19, producer_ready=object()
            )
            assert connector.read_jobs[-1] == job
        assert len(ring.submissions) == 1
        assert runtime._job_epoch == "a" * 32 and runtime._job_high_watermark == 3
    finally:
        runtime.shutdown()


def test_d21_watermark_and_wrong_request_do_not_acknowledge_active_job():
    initialize = threading.Event()
    connector = JobConnector()
    ring = FakeRing(b"")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_thread_initializer=lambda: initialize.wait(3),
        job_stream_factory=lambda ready: 19,
    )
    lower = _plan("lower")
    lower.capture_job_id = "a" * 32 + ":1"
    higher = _plan("higher")
    higher.capture_job_id = "a" * 32 + ":2"
    try:
        assert runtime.submit(lower, producer_stream=19, producer_ready=object())
        assert runtime.submit(higher, producer_stream=19, producer_ready=object())
        assert runtime._job_high_watermark == 2
        assert runtime.submit(lower, producer_stream=19, producer_ready=object())
        wrong_request = _plan("wrong-request")
        wrong_request.capture_job_id = lower.capture_job_id
        assert not runtime.submit(
            wrong_request, producer_stream=19, producer_ready=object()
        )
        assert not connector.read_jobs and not ring.submissions
        runtime.preempt(lower.request_id)
        runtime.preempt(higher.request_id)
        initialize.set()
        assert runtime.wait_idle(2)
        assert set(connector.read_jobs) == {lower.capture_job_id, higher.capture_job_id}
        assert not ring.submissions
    finally:
        initialize.set()
        runtime.shutdown()


@pytest.mark.parametrize(
    "job",
    [
        "epoch:1",
        "a" * 32 + ":0",
        "a" * 32 + ":01",
        "A" * 32 + ":1",
        "a" * 32 + ":18446744073709551616",
    ],
)
def test_d21_invalid_epoch_or_sequence_cannot_submit(job):
    connector = JobConnector()
    ring = FakeRing(b"")
    runtime = ManagerPageCaptureRuntime(
        connector,
        ring=ring,
        progress_thread_initializer=lambda: None,
        job_stream_factory=lambda ready: 19,
    )
    plan = _plan()
    plan.capture_job_id = job
    with pytest.raises(ValueError, match="UUID epoch"):
        runtime.submit(plan, producer_stream=19, producer_ready=object())
    assert not ring.submissions and not connector.read_jobs
    runtime.shutdown()
