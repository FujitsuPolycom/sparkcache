"""Filesystem cohort preparation runs outside the model callback."""

from sparkcache.request_cache_scope import UNSALTED_SCOPE

import threading
from types import SimpleNamespace

from sparkcache.test_spark_context_cache_connector import _make_connector
import sparkcache.test_spark_context_cache_connector as connector_fixtures
from sparkcache.spark_context_cache_connector import SparkCacheConnectorMetadata, _ReqPlan


def test_slow_cohort_metadata_does_not_block_model_callback(tmp_path, monkeypatch):
    connector = _make_connector(tmp_path, 0)
    connector._storage_mode = "block_pages_v1"
    entered, release, returned = (threading.Event() for _ in range(3))
    plan = _ReqPlan("slow-metadata", "a" * 64, 256, (3,), False, request_scope=UNSALTED_SCOPE)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    timings = []

    def prepare(plans):
        entered.set()
        assert release.wait(5)
        return list(plans), [], {}

    def load(_plan, *, timing, native_lane):
        timings.append(timing)
        return True

    monkeypatch.setattr(connector, "_prepare_page_base_read_cohorts", prepare)
    monkeypatch.setattr(connector, "_load_one", load)

    def forward_callback():
        connector.start_load_kv(None)
        returned.set()

    foreground = threading.Thread(target=forward_callback)
    foreground.start()
    try:
        assert entered.wait(2)
        assert returned.wait(0.5), "model callback waited for filesystem metadata"
        assert not connector.wait_for_pending_loads(timeout=0)
    finally:
        release.set()
        foreground.join(5)
        assert connector.wait_for_pending_loads(timeout=5)
        connector.shutdown()
    assert len(timings) == 1
    assert timings[0].phase_ns["metadata_preparation"] > 0
    assert timings[0].end_to_end_ns >= timings[0].phase_ns["metadata_preparation"]


def test_cohort_preparation_failure_completes_as_recompute(tmp_path, monkeypatch):
    connector = _make_connector(tmp_path, 0)
    connector._storage_mode = "block_pages_v1"
    plan = _ReqPlan("bad-metadata", "b" * 64, 256, (3,), False, request_scope=UNSALTED_SCOPE)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))

    def failed(_plans):
        raise OSError("metadata unavailable")

    monkeypatch.setattr(connector, "_prepare_page_base_read_cohorts", failed)
    try:
        connector.start_load_kv(None)
        assert connector.wait_for_pending_loads(timeout=5)
        assert connector.get_finished(set())[1] == {plan.request_id}
        assert connector.get_block_ids_with_load_errors() == {3}
        assert connector.counters["load_failed"] == 1
    finally:
        connector.shutdown()


def test_finished_request_cannot_join_a_queued_cohort(tmp_path, monkeypatch):
    connector = _make_connector(tmp_path, 0)
    connector._storage_mode = "block_pages_v1"
    plan = _ReqPlan("cancelled-metadata", "c" * 64, 256, (3,), False, request_scope=UNSALTED_SCOPE)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    start_workers = connector._ensure_load_threads
    monkeypatch.setattr(connector, "_ensure_load_threads", lambda: None)
    prepared = []

    def prepare(plans):
        prepared.extend(plans)
        return list(plans), [], {}

    monkeypatch.setattr(connector, "_prepare_page_base_read_cohorts", prepare)
    monkeypatch.setattr(connector, "_load_one", lambda *_args, **_kwargs: True)
    try:
        connector.start_load_kv(None)
        connector.request_finished(SimpleNamespace(cache_salt=None, request_id=plan.request_id), [])
        start_workers()
        assert connector.wait_for_pending_loads(timeout=5)
        assert not prepared
        assert connector.counters["load_failed"] == 1
        assert connector.get_block_ids_with_load_errors() == {3}
    finally:
        connector.shutdown()


def test_closed_cohort_during_handoff_releases_deferred_requests(tmp_path, monkeypatch):
    connector, _evidence, plans = (
        connector_fixtures.IntegratedPublicationAndSharingTests()._page_base_queue_fixture(tmp_path)
    )
    registered, release = threading.Event(), threading.Event()
    prepare = connector._prepare_page_base_read_cohorts

    def paused_prepare(plans):
        result = prepare(plans)
        registered.set()
        assert release.wait(5)
        return result

    monkeypatch.setattr(connector, "_prepare_page_base_read_cohorts", paused_prepare)
    monkeypatch.setattr(connector, "_load_one", lambda *_args, **_kwargs: True)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=plans[:2]))
    try:
        connector.start_load_kv(None)
        assert registered.wait(2)
        # Reproduce shutdown's close/release edge before deferred installation.
        connector._page_base_reads.close()
        connector._release_all_page_base_deferred()
        release.set()
        assert connector.wait_for_pending_loads(timeout=2)
        assert connector._deferred_page_base_loads == {}
        assert connector._page_base_plan_keys == {}
    finally:
        release.set()
        connector._release_all_page_base_deferred()
        connector.shutdown()


def test_shutdown_timeout_does_not_stop_worker_before_requeued_load(tmp_path, monkeypatch):
    connector = _make_connector(tmp_path, 0)
    connector._storage_mode = "block_pages_v1"
    entered, release = threading.Event(), threading.Event()
    plan = _ReqPlan("shutdown-metadata", "d" * 64, 256, (3,), False, request_scope=UNSALTED_SCOPE)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))

    def prepare(plans):
        entered.set()
        assert release.wait(5)
        return list(plans), [], {}

    monkeypatch.setattr(connector, "_prepare_page_base_read_cohorts", prepare)
    monkeypatch.setattr(connector, "_load_one", lambda *_args, **_kwargs: True)
    try:
        connector.start_load_kv(None)
        assert entered.wait(2)
        with monkeypatch.context() as bounded_shutdown:
            bounded_shutdown.setattr(connector, "wait_for_pending_loads", lambda **_: False)
            for thread in connector._load_threads:
                bounded_shutdown.setattr(thread, "join", lambda **_: None)
            connector.shutdown()
        release.set()
        assert connector.wait_for_pending_loads(timeout=2)
        for thread in connector._load_threads:
            thread.join(timeout=2)
            assert not thread.is_alive()
    finally:
        release.set()
        connector.shutdown()


def test_load_after_shutdown_is_rejected_without_starting_workers(tmp_path):
    connector = _make_connector(tmp_path, 0)
    connector.shutdown()
    plan = _ReqPlan("after-shutdown", "e" * 64, 256, (3,), False, request_scope=UNSALTED_SCOPE)
    connector.bind_connector_metadata(SparkCacheConnectorMetadata(plans=[plan]))
    connector.start_load_kv(None)
    assert connector._load_threads == []
    assert connector.wait_for_pending_loads(timeout=0)
    assert connector.get_finished(set())[1] == {plan.request_id}
    assert connector.get_block_ids_with_load_errors() == {3}
