"""Whole-prefix placement must not write into already-computed local pages."""

from types import SimpleNamespace

from sparkcache import test_spark_context_cache_connector as fixtures


def make_case(tmp_path):
    fixture = fixtures.AsyncRestoreTests()
    connector = fixture._cohort_connector(tmp_path)
    request = SimpleNamespace(
        request_id="private-owner", prompt_token_ids=list(range(1100))
    )
    digest = fixture._offer(connector, request.prompt_token_ids)
    return connector, request, digest


def test_full_restore_declines_nonempty_local_prefix(tmp_path):
    connector, request, digest = make_case(tmp_path)
    try:
        # The worker passes the complete block table to update_state_after_alloc,
        # including local prefix blocks; whole-prefix placement has no source
        # offset or write mask to protect those potentially shared pages.
        assert connector.get_num_new_matched_tokens(request, 256) == (0, False)
        assert request.request_id not in connector._need_load
        assert digest not in connector._restore_flights
    finally:
        connector.shutdown()


def test_unallocated_offer_is_retired_if_local_prefix_appears(tmp_path):
    connector, request, digest = make_case(tmp_path)
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (1024, True)
        assert connector.get_num_new_matched_tokens(request, 256) == (0, False)
        assert request.request_id not in connector._need_load
        assert digest not in connector._restore_flights
    finally:
        connector.shutdown()


def test_dispatched_writer_still_prevents_recompute(tmp_path):
    connector, request, digest = make_case(tmp_path)
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (1024, True)
        connector._restore_flights[digest].dispatched = True
        assert connector.get_num_new_matched_tokens(request, 256) == (None, False)
        assert digest in connector._restore_flights
    finally:
        connector.shutdown()


def test_allocated_undispatched_restore_still_prevents_recompute(tmp_path):
    connector, request, digest = make_case(tmp_path)
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (1024, True)
        connector.update_state_after_alloc(
            request, SimpleNamespace(get_block_ids=lambda: ((1, 2, 3, 4),)), 1024
        )
        assert connector.get_num_new_matched_tokens(request, 256) == (None, False)
        assert request.request_id in connector._pending_async_loads
        assert digest in connector._restore_flights
    finally:
        connector.shutdown()
