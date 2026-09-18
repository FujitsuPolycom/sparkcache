"""Request-level persistent cache isolation without GPUs or vLLM imports."""

import tempfile
from pathlib import Path
from types import SimpleNamespace
import dataclasses
import pickle

import pytest

from sparkcache import test_spark_context_cache_connector as fixtures
from sparkcache.request_cache_scope import fingerprint, UNSALTED_SCOPE
from sparkcache.spark_context_cache_connector import _ReqPlan


def request(name, salt, tokens=None):
    return SimpleNamespace(
        request_id=name, cache_salt=salt, prompt_token_ids=tokens or list(range(1100))
    )


def test_salted_request_does_not_join_an_unsalted_restore_flight():
    fixture = fixtures.AsyncRestoreTests()
    with tempfile.TemporaryDirectory() as directory:
        connector = fixture._cohort_connector(Path(directory))
        tokens = list(range(1100))
        digest = fixture._offer(connector, tokens)
        unsalted = SimpleNamespace(
            request_id="none", prompt_token_ids=tokens, cache_salt=None
        )
        salted = SimpleNamespace(
            request_id="tenant", prompt_token_ids=tokens, cache_salt="tenant-A"
        )
        assert connector.get_num_new_matched_tokens(unsalted, 0) == (1024, True)
        assert connector.get_num_new_matched_tokens(salted, 0) == (0, False)
        assert salted.request_id not in connector._restore_flights[digest].followers


@pytest.mark.parametrize("salt", [None, "", "tenant-A", "租户🔐"])
def test_matching_scope_shares_flight_but_other_scopes_miss(salt):
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures.AsyncRestoreTests()._cohort_connector(Path(directory))
        tokens = list(range(1100))
        digest = c._digest(tokens, 1024, request_scope=fingerprint(salt))
        c._quorum[digest] = {0, 1, 2, 3}
        assert c.get_num_new_matched_tokens(request("a", salt), 0) == (1024, True)
        assert c.get_num_new_matched_tokens(request("b", salt), 0) == (None, False)
        for index, other in enumerate([None, "", "tenant-A", "租户🔐"]):
            if other == salt:
                continue
            assert c.get_num_new_matched_tokens(
                request(f"other-{index}", other), 0
            ) == (0, False)
        assert c._restore_flights[digest].followers == {"b"}


@pytest.mark.parametrize("salt", [1, False, [], {}, b"salt", "\ud800"])
def test_invalid_salt_bypasses_cache_without_raising(salt):
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures.AsyncRestoreTests()._cohort_connector(Path(directory))
        assert c.get_num_new_matched_tokens(request("bad", salt), 0) == (0, False)
        assert c.get_shared_prefix_lease_candidate(request("bad", salt)) is None


def test_missing_or_mutated_scope_is_poisoned_until_request_cleanup():
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures.AsyncRestoreTests()._cohort_connector(Path(directory))
        missing = SimpleNamespace(
            request_id="missing", prompt_token_ids=list(range(1100))
        )
        assert c.get_num_new_matched_tokens(missing, 0) == (0, False)
        original = request("changing", "A")
        assert c._remember_request_scope(original) == fingerprint("A")
        original.cache_salt = "B"
        assert c.get_num_new_matched_tokens(original, 0) == (0, False)
        original.cache_salt = "A"
        assert c._remember_request_scope(original) is None
        c.request_finished(original, [])
        assert original.request_id not in c._request_cache_scopes
        assert c._remember_request_scope(original) == fingerprint("A")


def new_output(name, done=1024):
    result = fixtures._empty_scheduler_output()
    result.scheduled_new_reqs = [
        SimpleNamespace(
            req_id=name,
            prompt_token_ids=list(range(1100)),
            num_computed_tokens=0,
            block_ids=([0, 1, 2, 3],),
        )
    ]
    result.num_scheduled_tokens = {name: done}
    return result


def test_zero_external_allocation_latches_original_scope_for_store_only_metadata():
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures._make_connector(Path(directory), 0, 64)
        c._restore_enabled = False
        c.update_state_after_alloc(request("A", "租户"), None, 0)
        plans = c.build_connector_meta(new_output("A")).plans
        assert len(plans) == 1
        assert plans[0].request_scope == fingerprint("租户")
        assert plans[0].digest == c._digest(
            list(range(1100)), 1024, request_scope=fingerprint("租户")
        )
        assert "租户" not in repr(plans[0])
        assert pickle.loads(pickle.dumps(plans[0])) == plans[0]
        assert c.build_connector_meta(new_output("unknown")).plans == []


@pytest.mark.parametrize("resumed", [False, True])
def test_chunked_prefill_retains_scope_after_cached_request_output(resumed):
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures._make_connector(Path(directory), 0, 64)
        c.update_state_after_alloc(request("A", "A"), None, 0)
        assert c.build_connector_meta(new_output("A", 512)).plans == []
        output = fixtures._empty_scheduler_output()
        output.num_scheduled_tokens = {"A": 512}
        output.scheduled_cached_reqs = SimpleNamespace(
            req_ids=["A"],
            resumed_req_ids={"A"} if resumed else set(),
            num_computed_tokens=[512],
            new_block_ids=[([0, 1, 2, 3] if resumed else [],)],
        )
        plans = c.build_connector_meta(output).plans
        assert len(plans) == 1
        assert plans[0].request_scope == fingerprint("A")
        assert plans[0].digest == c._digest(
            list(range(1100)), 1024, request_scope=fingerprint("A")
        )


def test_salted_row_publication_alias_and_fresh_worker_restore():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        c = fixtures._make_connector(root, 0, 64)
        pool = fixtures._make_pools(40, 64)
        c.register_kv_caches(pool)
        tokens = list(range(8192))
        scope = fingerprint("tenant-A")
        digest = c._digest(tokens, 8192, request_scope=scope)
        plan = _ReqPlan(
            "A",
            digest,
            8192,
            tuple(range(32)),
            True,
            token_ids=tuple(tokens),
            request_scope=scope,
        )
        c._store_one(plan)
        shorter = c._digest(tokens, 4096, request_scope=scope)
        assert c._lookup_reusable(c._identity(0), shorter)[0].is_hit
        for other in [UNSALTED_SCOPE, fingerprint("tenant-B")]:
            assert not c._lookup_reusable(
                c._identity(0), c._digest(tokens, 4096, request_scope=other)
            )[0].is_hit
        restored = fixtures._make_connector(root, 0, 64)
        restored.register_kv_caches(fixtures._make_pools(40, 64))
        assert restored._load_one(dataclasses.replace(plan, is_store=False))
        legacy = fixtures.codec.context_prefix_digest(
            tokens,
            "sparkcache-context-v2-multimodal:"
            + c._config.build_identity(0, 0).storage_key,
            token_count=8192,
        )
        assert legacy != digest
        restored._quorum = {legacy: {0, 1, 2, 3}}
        assert restored.get_num_new_matched_tokens(
            request("legacy", None, tokens + [8192]), 0
        ) == (0, False)


def test_scope_mutation_after_offer_completes_allocated_hit_as_failed_load():
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures.AsyncRestoreTests()._cohort_connector(Path(directory))
        c.register_kv_caches(fixtures._make_pools(8, 64))
        tokens = list(range(1100))
        digest = c._digest(tokens, 1024, request_scope=fingerprint("A"))
        c._quorum[digest] = {0, 1, 2, 3}
        req = request("changed", "A")
        follower = request("follower", "A")
        assert c.get_num_new_matched_tokens(req, 0) == (1024, True)
        assert c.get_num_new_matched_tokens(follower, 0) == (None, False)
        req.cache_salt = "B"
        assert c.get_num_new_matched_tokens(req, 0) == (0, False)
        assert digest not in c._restore_flights
        assert follower.request_id not in c._restore_flight_followers
        blocks = SimpleNamespace(get_block_ids=lambda: ([1, 2, 3, 4],))
        c.update_state_after_alloc(req, blocks, 1024)
        meta = c.build_connector_meta(fixtures._empty_scheduler_output())
        assert len(meta.plans) == 1 and meta.plans[0].request_scope is None
        c.bind_connector_metadata(meta)
        c.start_load_kv(None)
        assert fixtures._drain(c) == {req.request_id}
        assert c.get_block_ids_with_load_errors() == {1, 2, 3, 4}
        assert not c._pending_async_loads


@pytest.mark.parametrize(
    "scope", [None, "", "a" * 64, "sparkcache-request-scope-v0:" + "a" * 64]
)
def test_worker_rejects_missing_or_unknown_scope_before_placement(scope):
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures._make_connector(Path(directory), 0, 64)
        c.register_kv_caches(fixtures._make_pools(8, 64))
        plan = _ReqPlan(
            "invalid", "a" * 64, 1024, (1, 2, 3, 4), False, request_scope=scope
        )
        assert not c._load_one(plan)
        with pytest.raises(ValueError, match="request cache scope"):
            c._snapshot_store(dataclasses.replace(plan, is_store=True))


def test_valid_but_changed_scope_breaks_trusted_metadata_consistency_binding():
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures._make_connector(Path(directory), 0, 64)
        c.register_kv_caches(fixtures._make_pools(8, 64))
        plan = _ReqPlan(
            "A",
            c._digest(list(range(1024)), 1024, request_scope=fingerprint("A")),
            1024,
            (1, 2, 3, 4),
            False,
            request_scope=fingerprint("A"),
        )
        changed = dataclasses.replace(plan, request_scope=fingerprint("B"))
        assert not c._load_one(changed)
        with pytest.raises(ValueError, match="inconsistent"):
            c._snapshot_store(dataclasses.replace(changed, is_store=True))


@pytest.mark.parametrize("streaming", [False, True])
def test_mixed_scope_batch_preserves_distinct_store_metadata(streaming):
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures._make_connector(Path(directory), 0, 64)
        c._streaming_snapshots_enabled = streaming
        output = fixtures._empty_scheduler_output()
        for name in ["A", "B"]:
            c.update_state_after_alloc(request(name, name), None, 0)
            row = new_output(name)
            output.scheduled_new_reqs.extend(row.scheduled_new_reqs)
            output.num_scheduled_tokens.update(row.num_scheduled_tokens)
        meta = c.build_connector_meta(output)
        plans = meta.streaming_snapshot_offers if streaming else meta.plans
        assert len(plans) == 2
        assert {p.request_scope for p in plans} == {fingerprint("A"), fingerprint("B")}
        assert len({p.digest for p in plans}) == 2


def test_tail_publication_uses_only_matching_scope_base():
    with tempfile.TemporaryDirectory() as directory:
        c = fixtures._make_connector(
            Path(directory),
            0,
            64,
            extra_config={"spark_cache_publication_schema": "tail-cow-v1"},
        )
        c.register_kv_caches(fixtures._make_pools(12, 64))
        tokens = list(range(2048))
        scope = fingerprint("A")
        base = c._digest(tokens, 1024, request_scope=scope)
        result = c._digest(tokens, 2048, request_scope=scope)
        c._store_one(
            _ReqPlan(
                "base",
                base,
                1024,
                (0, 1, 2, 3),
                True,
                token_ids=tuple(tokens[:1024]),
                request_scope=scope,
            )
        )
        c._quorum[base] = {0, 1, 2, 3}
        assert c._publication_base(tokens, 2048, request_scope=scope) == (base, 1024)
        assert c._publication_base(tokens, 2048, request_scope=fingerprint("B")) == (
            "",
            0,
        )
        c._store_one(
            _ReqPlan(
                "tail",
                result,
                2048,
                tuple(range(8)),
                True,
                token_ids=tuple(tokens),
                base_context_digest=base,
                base_span_tokens=1024,
                request_scope=scope,
            )
        )
        assert c._store.lookup(c._identity(0), result).is_hit
