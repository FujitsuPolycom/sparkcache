"""Qwen hybrid profiles isolate persistent bytes from other model layouts."""

from sparkcache.spark_context_cache_profiles import resolve_profile


def test_qwen38_hybrid_profile_isolates_layout_identity():
    qwen = resolve_profile("qwen38-flash-next-hybrid")
    glm = resolve_profile("glm53-flash-hybrid")
    assert qwen.quantization_layout != glm.quantization_layout
    assert qwen.rope_layout != glm.rope_layout
    assert qwen.storage_mode == "block_pages_v1"
    assert qwen.cuda_page_restore
    assert not qwen.kv_replicated_across_tp
    assert qwen.boundary_hidden_policy == "live_forward"
    assert qwen.persisted_families("separate") == frozenset({"target_ckv"})


def test_qwen38_profile_accepts_tp2_dcp1_page_geometry():
    profile = resolve_profile("qwen38-flash-next-hybrid")
    profile.validate_for_deployment(
        dcp_degree=1, block_size=16, min_span_tokens=4096, cuda_restore=True
    )
    profile.validate_for_deployment(
        dcp_degree=1, block_size=2848, min_span_tokens=4096, cuda_restore=True
    )
