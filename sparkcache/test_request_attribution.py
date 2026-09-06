"""Prompt attribution follows accepted execution, not matching offers."""

import pytest

from sparkcache.request_attribution import RequestAttribution


def admit(state, local=0, external=0, generation=0):
    state.record("admitted", local_tokens=local, external_tokens=external,
                 preemptions=generation, source="prefix_lookup")


def completed(state, start, end, generation=0):
    state.record("prompt_step_completed", start_token=start, end_token=end,
                 preemptions=generation)


def test_local_partial_tail_uses_final_scheduler_count():
    state = RequestAttribution(1100)
    admit(state, local=1031)
    completed(state, 1031, 1100)
    assert state.summary("FINISHED_STOPPED") == {
        "prompt_tokens": 1100, "local_tokens_reused": 1031,
        "external_tokens_reused": 0, "prompt_tokens_computed": 69,
        "preemptions": 0, "attribution_complete": True,
        "status": "FINISHED_STOPPED",
        "accounting_scope": "accepted_target_prompt_work_across_attempts",
    }


def test_external_offer_without_finalization_is_not_reuse():
    state = RequestAttribution(1100)
    admit(state, local=256, external=768)
    completed(state, 256, 1100)
    result = state.summary("FINISHED_STOPPED")
    assert result["external_tokens_reused"] == 0
    assert not result["attribution_complete"]


def test_verified_restore_counts_only_effective_prompt_reuse_once():
    state = RequestAttribution(1024)
    admit(state, local=256, external=768)
    state.record("restore_finalized", valid_prefix_tokens=1023,
                 success=True, preemptions=0)
    # Re-admission of the parked request does not replace its source accounting.
    admit(state)
    completed(state, 1023, 1024)
    completed(state, 1023, 1024)
    result = state.summary("FINISHED_STOPPED")
    assert (result["local_tokens_reused"], result["external_tokens_reused"],
            result["prompt_tokens_computed"]) == (256, 767, 1)
    assert result["attribution_complete"]


def test_failed_restore_recomputes_and_does_not_credit_external_offer():
    state = RequestAttribution(1100)
    admit(state, local=256, external=768)
    state.record("restore_finalized", valid_prefix_tokens=256,
                 success=False, preemptions=0)
    completed(state, 256, 1100)
    result = state.summary("FINISHED_STOPPED")
    assert result["external_tokens_reused"] == 0
    assert result["prompt_tokens_computed"] == 844
    assert result["attribution_complete"]


def test_gpu_lease_is_local_reuse_without_external_credit():
    state = RequestAttribution(1024)
    state.record("admitted", local_tokens=1023, external_tokens=0,
                 preemptions=0, source="gpu_lease")
    completed(state, 1023, 1024)
    result = state.summary("FINISHED_STOPPED")
    assert result["local_tokens_reused"] == 1023
    assert result["external_tokens_reused"] == 0


def test_failed_private_restore_allows_fresh_local_lookup_in_same_generation():
    state = RequestAttribution(1100)
    admit(state, external=1024)
    state.record("restore_finalized", valid_prefix_tokens=0,
                 success=False, preemptions=0)
    admit(state, local=768)
    completed(state, 768, 1100)
    result = state.summary("FINISHED_STOPPED")
    assert result["external_tokens_reused"] == 0
    assert result["local_tokens_reused"] == 768
    assert result["prompt_tokens_computed"] == 332
    assert result["attribution_complete"]


def test_preemption_keeps_completed_work_and_counts_recomputation():
    state = RequestAttribution(1000)
    admit(state, local=256)
    completed(state, 256, 512)
    state.record("preempted", preemptions=1)
    admit(state, generation=1)
    completed(state, 0, 1000, generation=1)
    result = state.summary("FINISHED_STOPPED")
    assert result["local_tokens_reused"] == 256
    assert result["prompt_tokens_computed"] == 1256
    assert result["preemptions"] == 1
    assert result["attribution_complete"]


def test_abort_before_execution_does_not_credit_verified_cache_warming():
    state = RequestAttribution(1024)
    admit(state, external=1024)
    state.record("restore_finalized", valid_prefix_tokens=1023,
                 success=True, preemptions=0)
    result = state.summary("FINISHED_ABORTED")
    assert result["external_tokens_reused"] == 0
    assert result["prompt_tokens_computed"] == 0
    assert not result["attribution_complete"]


def test_decode_after_preemption_credits_resident_prompt_without_output_tokens():
    state = RequestAttribution(1000)
    admit(state)
    completed(state, 0, 1000)
    state.record("preempted", preemptions=1)
    admit(state, local=1000, generation=1)
    completed(state, 1000, 1000, generation=1)
    result = state.summary("FINISHED_STOPPED")
    assert result["local_tokens_reused"] == 1000
    assert result["prompt_tokens_computed"] == 1000


def test_missing_attempt_and_unobserved_prefix_do_not_claim_complete_attribution():
    state = RequestAttribution(1000)
    completed(state, 200, 1000)
    assert not state.summary("FINISHED_STOPPED")["attribution_complete"]


@pytest.mark.parametrize("field,value", [("local_tokens", -1),
                                          ("external_tokens", True),
                                          ("preemptions", -1)])
def test_invalid_scheduler_observation_is_rejected(field, value):
    fields = dict(local_tokens=0, external_tokens=0, preemptions=0)
    fields[field] = value
    with pytest.raises(ValueError):
        RequestAttribution(1000).record("admitted", **fields)
