"""Exact prefix analysis shared across scheduler callbacks."""

from types import SimpleNamespace
from unittest import mock

import pytest

from sparkcache.test_spark_context_cache_connector import (
    KVConnectorRole,
    _empty_scheduler_output,
    _make_connector,
)
from sparkcache import spark_context_cache_codec as codec


@pytest.fixture
def connector(tmp_path):
    value = _make_connector(
        tmp_path, 0, role=KVConnectorRole.SCHEDULER,
        extra_config={"spark_cache_publication_schema": "tail-cow-v1"},
    )
    yield value
    value.shutdown()


def publication_output(request, scheduled=1024):
    output = _empty_scheduler_output()
    output.scheduled_new_reqs = [SimpleNamespace(
        cache_salt=None, req_id=request.request_id,
        prompt_token_ids=request.prompt_token_ids,
        num_computed_tokens=0,
        block_ids=([10, 11, 12, 13],),
    )]
    output.num_scheduled_tokens = {request.request_id: scheduled}
    return output


def test_lease_lookup_and_publication_share_one_hash_pass(connector):
    request = SimpleNamespace(cache_salt=None, request_id="prefix", prompt_token_ids=list(range(1100)))
    base_digest = connector._digest(request.prompt_token_ids, 512)
    with mock.patch(
        "sparkcache.spark_context_cache_connector.chunk_prefix_digests",
        wraps=codec.chunk_prefix_digests,
    ) as hashes:
        assert connector.get_shared_prefix_lease_candidate(request) is None
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        # Availability is mutable even though the request's digest table is not.
        connector._quorum[base_digest] = {0, 1, 2, 3}
        metadata = connector.build_connector_meta(publication_output(request))
        assert metadata.plans[0].base_context_digest == base_digest
        assert metadata.plans[0].span_tokens == 1024
        assert metadata.plans[0].digest == connector._digest(request.prompt_token_ids, 1024)
        assert hashes.call_count == 1
    connector.request_finished(request, [])
    assert request.request_id not in connector._prefix_digest_candidates


def test_prefix_cache_rejects_same_length_mutation(connector):
    request = SimpleNamespace(cache_salt=None, request_id="mutable", prompt_token_ids=list(range(1100)))
    connector.get_shared_prefix_lease_candidate(request)
    request.prompt_token_ids[1] = 12345
    with mock.patch(
        "sparkcache.spark_context_cache_connector.chunk_prefix_digests",
        wraps=codec.chunk_prefix_digests,
    ) as hashes:
        connector.get_shared_prefix_lease_candidate(request)
        hashes.assert_called_once()
    metadata = connector.build_connector_meta(publication_output(request))
    assert metadata.plans[0].digest == connector._digest(request.prompt_token_ids, 1024)


def test_present_root_skips_publication_base_work(connector):
    request = SimpleNamespace(cache_salt=None, request_id="present", prompt_token_ids=list(range(1100)))
    connector._quorum[connector._digest(request.prompt_token_ids, 1024)] = {0, 1, 2, 3}
    with mock.patch.object(connector, "_publication_base", wraps=connector._publication_base) as base:
        assert connector.build_connector_meta(publication_output(request)).plans == []
        base.assert_not_called()


def test_local_prefix_filters_cached_candidates(connector):
    request = SimpleNamespace(cache_salt=None, request_id="local", prompt_token_ids=list(range(1100)))
    connector.get_shared_prefix_lease_candidate(request)
    connector._quorum[connector._digest(request.prompt_token_ids, 512)] = {0, 1, 2, 3}
    with mock.patch(
        "sparkcache.spark_context_cache_connector.chunk_prefix_digests",
        wraps=codec.chunk_prefix_digests,
    ) as hashes:
        assert connector.get_num_new_matched_tokens(request, 512) == (0, False)
        hashes.assert_not_called()
