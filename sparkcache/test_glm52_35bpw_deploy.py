"""Tests for the GLM-5.2 fixed-MTP4 deployment (SparkRing identifier R7)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest

from deploy.glm52_35bpw.profile import (
    DEFAULT_CACHE_ROOT,
    DEFAULT_CHECKPOINT_SHA256,
    LOW_WATERMARK_BYTES,
    MAX_BYTES,
    PROFILE,
    ProfileTransformError,
    transform_inspection,
)
import deploy.glm52_35bpw.profile as glm_profile
from deploy.glm52_35bpw import prepare_vllm_overlays
from deploy.glm52_35bpw.semantic_gate import run_hit_after_quorum
from deploy.glm52_35bpw import launch as glm_launch
from deploy.glm52_35bpw.launch import _inspection, main as launch_main
from sparkcache.spark_context_cache_profiles import resolve_profile


def _json(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _source_inspection() -> dict:
    speculative = {
        "model": "/models/glm52-exl3-r7-3.5bpw",
        "method": "mtp",
        "num_speculative_tokens": 4,
        "draft_tensor_parallel_size": 4,
        "quantization": "exl3",
        "moe_backend": "b12x",
        "attention_backend": "B12X_MLA_SPARSE",
        "use_local_argmax_reduction": False,
        "draft_sample_method": "greedy",
    }
    compilation = {
        "cudagraph_mode": "FULL_AND_PIECEWISE",
        "cudagraph_capture_sizes": list(range(1, 41)),
        "custom_ops": ["all"],
        "pass_config": {"fuse_allreduce_rms": True},
        "operator_q40_overlay": "preserve-this-field",
    }
    return {
        "Config": {
            "Cmd": [
                "serve",
                "/models/glm52-exl3-r7-3.5bpw",
                "--node-rank",
                "0",
                "--master-addr",
                "192.0.2.10",
                "--master-port",
                "29500",
                "--port",
                "8000",
                "--distributed-executor-backend",
                "mp",
                "--nnodes",
                "4",
                "--tensor-parallel-size",
                "4",
                "--decode-context-parallel-size",
                "4",
                "--dcp-comm-backend",
                "ag_rs",
                "--dcp-kv-cache-interleave-size",
                "1",
                "--quantization",
                "exl3",
                "--moe-backend",
                "b12x",
                "--attention-backend",
                "B12X_MLA_SPARSE",
                "--kv-cache-dtype",
                "nvfp4_ds_mla",
                "--max-model-len",
                "262144",
                "--kv-cache-memory-bytes",
                "9250000000",
                "--max-num-seqs",
                "8",
                "--max-num-batched-tokens",
                "4096",
                "--load-format",
                "instanttensor",
                "--gpu-memory-utilization",
                "0.85",
                "--served-model-name",
                "glm-5.2-exl3-r7-3.5bpw",
                "--max-cudagraph-capture-size",
                "40",
                "--speculative-config",
                _json(speculative),
                "--compilation-config",
                _json(compilation),
                "--kernel-config",
                '{"exact_q40":true}',
                "--enable-chunked-prefill",
                "--no-async-scheduling",
                "--host",
                "0.0.0.0",
                "--disable-hybrid-kv-cache-manager",
                "--kv-transfer-config",
                '{"kv_connector":"LMCacheConnector"}',
            ],
            "Env": [
                "RANK=0",
                "MASTER_PORT=29500",
                "SPARKRING_ATTEST_MODEL_REPOSITORY="
                "brandonmusic/GLM-5.2-EXL3-TR3v4-3.5bpw-MTP78",
                "SPARKRING_ATTEST_MODEL_REVISION="
                "9ab9579774cc432df91567a36f6e9e863e0d4c9f",
                "SPARKRING_ATTEST_MODEL_CONFIG_SHA256="
                "fabb73eb513ec64f3a365da396b38de8d55b3930edfb11baeecbf34ecafa6126",
                "SPARKRING_ATTEST_MODEL_INDEX_SHA256="
                "9fd852f69ed64442e31dce1cbc5fe7acd0a76bfb848e945d272fe98d00d0c9cd",
                "SPARKRING_MODEL_REVISION=46537e0e16fcd156627800139b41b9c497fc7ee2",
                "SPARKRING_MODEL_CONFIG_SHA256="
                "ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69",
                "KV_FP8_ROPE=1",
                "VLLM_NVFP4_MLA_DYNAMIC_SCALE=1",
                "VLLM_EXL3_PREFILL_CAPACITY=4096",
                "VLLM_SPARK_MAX_QUERY_ROWS=40",
                "VLLM_SPARK_MTP_MODE_ID=fixed-mtp4",
                "VLLM_SPARK_MTP_TOKENS=4",
                "SPARK_ADAPTIVE_MTP_CONTROL=0",
                "VLLM_SPARK_MTP_ADAPTIVE_WINDOW=0",
                "VLLM_SPARK_TRUE_ADAPTIVE_DRAFT=0",
                "VLLM_B12X_MLA_CKV_GATHER=1",
                "VLLM_B12X_MLA_CKV_GATHER_MAX_TOKENS=262144",
                "VLLM_SPARK_TP4_MODE=custom",
                "SPARK_TP4_LIBRARY=/opt/sparkring/spark_transport/"
                "libspark_transport_capi.so",
                "ONLINE_QUANT=exl3-b6",
                "VLLM_EXL3_ONLINE_CACHE_MODE=readwrite",
                "VLLM_EXL3_ONLINE_TRELLIS_BITS=6",
                "SPARK_Q40_EXACT_STATE_CHECKPOINT="
                "9ab9579774cc432df91567a36f6e9e863e0d4c9f",
                "SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256="
                "8fad5330c88f55dc57e4d8e298f2af23e16390b97153b569a2e572e0fb5065c2",
                "SPARK_Q40_EXACT_STATE_IMAGE_ID=sha256:"
                "02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513",
                "SPARKRING_IMAGE_DIGEST=sha256:"
                "02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513",
                "SPARK_TP4_CONTROL_PORT0=11100",
                "SPARK_TP4_CONTROL_PORT1=11101",
                "SPARK_TP4_ALLGATHER_BASE_PORT=10200",
                "SPARK_CONTEXT_CACHE_ENABLE=0",
                "LMCACHE_CONFIG_FILE=/cache/lmcache.yaml",
                "LEGACY_CONNECTOR_PATH=/opt/lmcache/connector.py",
                "PYTHONPATH=/opt/lmcache:/opt/sparkring-r7-tvm-ffi:/opt/spark-vllm",
                "SPARKRING_EXPLICITLY_UNSET=VLLM_ADAPTIVE_SPEC_DEPTHS",
            ],
            "Labels": {"org.sparkring.r7": "accepted-source"},
        },
        "Image": (
            "sha256:02881d5229d4f4d1cbba0cf40537492a2a505b9d4e43bbfe9a0b2a7bd0584513"
        ),
        "HostConfig": {
            "NetworkMode": "host",
            "IpcMode": "host",
            "ShmSize": 17179869184,
        },
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/srv/models/glm52-r7",
                "Destination": "/models/glm52-exl3-r7-3.5bpw",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "/srv/cache/glm52-r7-jit",
                "Destination": "/cache/jit",
                "RW": True,
            },
            {
                "Type": "bind",
                "Source": "/srv/cache/glm52-r7-online",
                "Destination": "/cache/exl3-online",
                "RW": True,
            },
            {
                "Type": "bind",
                "Source": "/srv/r7-overlays/spark_tp4_backend.py",
                "Destination": "/opt/spark-vllm/spark_tp4_backend.py",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "/srv/r7-overlays/model_runner.py",
                "Destination": (
                    "/opt/venv/lib/python3.12/site-packages/"
                    "vllm/v1/worker/gpu/model_runner.py"
                ),
                "RW": False,
            },
        ],
    }


def _option_value(arguments: list[str], option: str) -> str:
    assert arguments.count(option) == 1
    return arguments[arguments.index(option) + 1]


def _connector_extra(inspection: dict) -> dict:
    raw = _option_value(inspection["Config"]["Cmd"], "--kv-transfer-config")
    return json.loads(raw)["kv_connector_extra_config"]


def _environment(inspection: dict) -> dict[str, str]:
    return {
        item.partition("=")[0]: item.partition("=")[2]
        for item in inspection["Config"]["Env"]
    }


def test_deployment_alias_reuses_the_frozen_glm_cache_layout() -> None:
    assert resolve_profile("glm52-exl3-r7-3.5bpw") is resolve_profile("glm52-nvfp4")
    assert resolve_profile("glm52-exl3-r7-3.5bpw").name == "glm52-nvfp4"


def test_glm_hit_gate_primes_worker_quorum(monkeypatch, tmp_path: Path) -> None:
    budgets: list[int] = []
    hit_options: dict[str, int] = {}

    def request(endpoint, model, prompt, max_tokens):
        budgets.append(max_tokens)
        return {"choices": [{"message": {"content": "2"}}]}

    def hit(endpoint, model, reference, **options):
        hit_options.update(options)
        return {
            "content": "SPARKCACHE_OK:9540",
            "post_restore_canary": "42",
        }

    monkeypatch.setattr(
        "deploy.glm52_35bpw.semantic_gate._request",
        request,
    )
    monkeypatch.setattr(
        "deploy.glm52_35bpw.semantic_gate.run_hit",
        hit,
    )

    result = run_hit_after_quorum(
        "http://stack",
        "glm-5.2-exl3-r7-3.5bpw",
        tmp_path / "reference.json",
    )

    assert result["quorum_prime"] == "2"
    assert budgets == [128]
    assert hit_options == {"long_max_tokens": 512, "short_max_tokens": 128}


def test_launch_reader_accepts_bom_prefixed_docker_inspection(tmp_path: Path) -> None:
    path = tmp_path / "inspect.json"
    path.write_bytes(b"\xef\xbb\xbf" + json.dumps([_source_inspection()]).encode())
    assert _inspection(path) == _source_inspection()


def test_glm_cli_passes_required_rank_local_cache_bind(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps([_source_inspection()]), encoding="utf-8")
    observed: list[tuple[tuple, dict]] = []
    monkeypatch.setattr(
        "deploy.glm52_35bpw.launch.launch",
        lambda *args, **kwargs: observed.append((args, kwargs)),
    )
    monkeypatch.setattr(
        glm_launch,
        "_validate_sparkcache_source",
        lambda path, digest: None,
    )
    monkeypatch.setattr(
        glm_launch,
        "_validate_overlay_inputs",
        lambda receipt, scheduler, config: "a" * 64,
    )

    assert (
        launch_main(
            [
                "--inspect",
                str(path),
                "--image",
                _source_inspection()["Image"],
                "--name",
                "glm52-r0",
                "--checkpoint-sha256",
                DEFAULT_CHECKPOINT_SHA256,
                "--cache-host-path",
                "/host-cache/glm52-r0",
                "--sparkcache-source-host-path",
                "/host-code/sparkcache",
                "--scheduler-overlay-host-path",
                "/host-overlays/scheduler.py",
                "--vllm-config-overlay-host-path",
                "/host-overlays/vllm.py",
                "--vllm-overlay-receipt-host-path",
                "/host-overlays/receipt.json",
                "--create-only",
            ]
        )
        == 0
    )

    assert observed[0][1]["extra_binds"] == (
        ("/host-cache/glm52-r0", "/cache/sparkcache-glm52-r7", False),
        ("/host-code/sparkcache", "/opt/sparkcache-src/sparkcache", True),
        (
            "/host-overlays/scheduler.py",
            "/opt/venv/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py",
            True,
        ),
        (
            "/host-overlays/vllm.py",
            "/opt/venv/lib/python3.12/site-packages/vllm/config/vllm.py",
            True,
        ),
    )
    assert observed[0][1]["preserve_all_binds"] is True


def test_launch_rejects_image_other_than_inspected_identity(
    monkeypatch, tmp_path: Path
) -> None:
    path = tmp_path / "inspect.json"
    path.write_text(json.dumps([_source_inspection()]), encoding="utf-8")
    with pytest.raises(SystemExit):
        launch_main(
            [
                "--inspect",
                str(path),
                "--image",
                "sha256:" + "0" * 64,
                "--name",
                "glm52-r0",
                "--checkpoint-sha256",
                DEFAULT_CHECKPOINT_SHA256,
                "--cache-host-path",
                "/host-cache/glm52-r0",
                "--sparkcache-source-host-path",
                "/host-code/sparkcache",
                "--scheduler-overlay-host-path",
                "/host-overlays/scheduler.py",
                "--vllm-config-overlay-host-path",
                "/host-overlays/vllm.py",
                "--vllm-overlay-receipt-host-path",
                "/host-overlays/receipt.json",
            ]
        )


def test_overlay_receipt_and_files_must_match(monkeypatch, tmp_path: Path) -> None:
    scheduler = tmp_path / "scheduler.py"
    config = tmp_path / "vllm.py"
    scheduler.write_bytes(b"scheduler")
    config.write_bytes(b"config")
    scheduler_hash = hashlib.sha256(b"scheduler").hexdigest()
    config_hash = hashlib.sha256(b"config").hexdigest()
    monkeypatch.setattr(glm_launch, "EXPECTED_SCHEDULER_SHA256", scheduler_hash)
    monkeypatch.setattr(glm_launch, "EXPECTED_VLLM_CONFIG_SHA256", config_hash)
    monkeypatch.setattr(glm_launch, "SPARKCACHE_SOURCE_SHA256", "c" * 64)
    receipt = tmp_path / "receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "sparkcache-glm52-r7-vllm-overlays/v1",
                "sparkcache_source_sha256": "c" * 64,
                "files": [
                    {"output": "scheduler.py", "sha256": scheduler_hash},
                    {"output": "vllm.py", "sha256": config_hash},
                ],
            }
        ),
        encoding="utf-8",
    )

    assert glm_launch._validate_overlay_inputs(receipt, scheduler, config) == ("c" * 64)
    config.write_bytes(b"changed")
    with pytest.raises(ProfileTransformError, match="hash differs"):
        glm_launch._validate_overlay_inputs(receipt, scheduler, config)


def test_sparkcache_source_requires_runtime_package_files(tmp_path: Path) -> None:
    source = tmp_path / "sparkcache"
    for relative in (
        "spark_context_cache_connector.py",
        "spark_context_cache_profiles.py",
        "spark_context_cache_store.py",
        "persistent_context_cache/cache_manifest.py",
        "streaming/factory.py",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture\n", encoding="utf-8")
    digest = prepare_vllm_overlays.source_tree_sha256(source)
    glm_launch._validate_sparkcache_source(source, digest)
    (source / "streaming/factory.py").unlink()
    with pytest.raises(ProfileTransformError, match="lacks the required"):
        glm_launch._validate_sparkcache_source(source, digest)


def test_sparkcache_source_digest_is_line_ending_independent(tmp_path: Path) -> None:
    source = tmp_path / "sparkcache"
    source.mkdir()
    path = source / "connector.py"
    path.write_bytes(b"first\nsecond\n")
    lf_digest = prepare_vllm_overlays.source_tree_sha256(source)
    path.write_bytes(b"first\r\nsecond\r\n")
    assert prepare_vllm_overlays.source_tree_sha256(source) == lf_digest


def test_sparkcache_source_digest_uses_posix_relative_path_order(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sparkcache"
    source.mkdir()
    files = {"Z.py": b"upper\n", "a.py": b"lower\n"}
    for relative, content in files.items():
        (source / relative).write_bytes(content)

    expected = hashlib.sha256(b"sparkcache-source-tree/v1\x00")
    for relative in sorted(files):
        encoded = relative.encode("utf-8")
        expected.update(len(encoded).to_bytes(4, "little"))
        expected.update(encoded)
        expected.update(hashlib.sha256(files[relative]).digest())

    assert prepare_vllm_overlays.source_tree_sha256(source) == expected.hexdigest()


def test_profile_matches_the_public_r7_contract() -> None:
    repository = Path(__file__).resolve().parents[1]
    assert PROFILE["sparkcache"]["source_sha256"] == (
        prepare_vllm_overlays.source_tree_sha256(repository / "sparkcache")
    )
    assert PROFILE["model"]["revision"] == ("9ab9579774cc432df91567a36f6e9e863e0d4c9f")
    assert PROFILE["serving"] == {
        "served_model_name": "glm-5.2-exl3-r7-3.5bpw",
        "tensor_parallel_size": 4,
        "decode_context_parallel_size": 4,
        "dcp_backend": "ag_rs",
        "dcp_kv_cache_interleave_size": 1,
        "mtp_tokens": 4,
        "max_query_rows": 40,
        "kv_cache_dtype": "nvfp4_ds_mla",
        "kv_dynamic_per_token_scale": True,
        "kv_fp8_rope": True,
        "block_size": 64,
        "max_model_len": 262144,
        "kv_cache_bytes_per_rank": 9250000000,
        "max_num_seqs": 8,
        "max_num_batched_tokens": 4096,
        "execution_mode": "FULL_AND_PIECEWISE",
        "native_prefix_caching": True,
    }
    assert (MAX_BYTES, LOW_WATERMARK_BYTES) == (
        200 * 1024**3,
        180 * 1024**3,
    )


def test_transform_replaces_only_cache_wiring_and_removes_lmcache() -> None:
    source = _source_inspection()
    original = copy.deepcopy(source)
    transformed = transform_inspection(source)

    assert source == original
    arguments = transformed["Config"]["Cmd"]
    assert arguments[:2] == ["serve", "/models/glm52-exl3-r7-3.5bpw"]
    assert _option_value(arguments, "--speculative-config") == _option_value(
        original["Config"]["Cmd"], "--speculative-config"
    )
    assert _option_value(arguments, "--compilation-config") == _option_value(
        original["Config"]["Cmd"], "--compilation-config"
    )
    assert _option_value(arguments, "--kernel-config") == '{"exact_q40":true}'
    assert "--disable-hybrid-kv-cache-manager" in arguments
    assert "--enable-prefix-caching" not in arguments
    assert _option_value(arguments, "--block-size") == "64"
    assert not any("lmcache" in value.lower() for value in arguments)

    connector = json.loads(_option_value(arguments, "--kv-transfer-config"))
    assert connector["kv_connector_module_path"] == (
        "sparkcache.spark_context_cache_connector"
    )
    assert connector["kv_load_failure_policy"] == "recompute"
    extra = connector["kv_connector_extra_config"]
    assert extra == {
        "spark_cache_root": DEFAULT_CACHE_ROOT,
        "spark_cache_model_profile": "glm52-exl3-r7-3.5bpw",
        "spark_cache_target_checkpoint_sha256": DEFAULT_CHECKPOINT_SHA256,
        "spark_cache_draft_policy": "colocated_target",
        "spark_cache_store": True,
        "spark_cache_restore": True,
        "spark_cache_scheduler_probe": "none",
        "spark_cache_streaming_snapshots": False,
        "spark_cache_cuda_restore": False,
        "spark_cache_max_bytes": 200 * 1024**3,
        "spark_cache_low_watermark_bytes": 180 * 1024**3,
        "spark_cache_ttl_seconds": 0,
        "spark_cache_min_span_tokens": 256,
        "spark_cache_max_span_tokens": 262144,
    }
    assert "spark_cache_draft_checkpoint_sha256" not in extra

    environment = _environment(transformed)
    assert "LMCACHE_CONFIG_FILE" not in environment
    assert "LEGACY_CONNECTOR_PATH" not in environment
    assert "lmcache" not in environment["PYTHONPATH"].lower()
    assert environment["PYTHONPATH"].startswith("/opt/sparkcache-src:")
    assert "/opt/sparkcache-src/sparkcache" not in environment["PYTHONPATH"]
    assert "SPARK_CONTEXT_CACHE_ENABLE" not in environment
    assert environment["SPARKRING_MODEL_REVISION"] == (
        "46537e0e16fcd156627800139b41b9c497fc7ee2"
    )
    assert environment["SPARKRING_MODEL_CONFIG_SHA256"] == (
        "ffd30e72ab8bb7e8ad560f2aaab03cc595f3106f0acf793ef96eedaf90f66d69"
    )
    assert "LMCACHE_CONFIG_FILE" in environment["SPARKRING_EXPLICITLY_UNSET"]
    assert "LEGACY_CONNECTOR_PATH" in environment["SPARKRING_EXPLICITLY_UNSET"]
    assert "SPARK_CONTEXT_CACHE_ENABLE" in environment["SPARKRING_EXPLICITLY_UNSET"]
    assert transformed["Config"]["Labels"] == {
        "org.sparkring.r7": "accepted-source",
        "org.sparkcache.deployment-profile": "glm52-exl3-r7-3.5bpw",
    }


def test_transform_rejects_the_unsupported_q35_q40_state_variant() -> None:
    inspection = _source_inspection()
    environment = inspection["Config"]["Env"]
    index = next(
        index
        for index, value in enumerate(environment)
        if value.startswith("SPARK_Q40_EXACT_STATE_EXPECTED_EXL3_SHA256=")
    )
    environment[index] = (
        "SPARK_Q35_EXACT_STATE_EXPECTED_EXL3_SHA256="
        "91187fcd3c1bdd23367c4ec9ad4085e080ef22d42fb00ee71f7e299724ce2050"
    )

    with pytest.raises(ProfileTransformError, match="target-only exact-Q40"):
        transform_inspection(inspection)


@pytest.mark.parametrize(
    ("streaming", "cuda_restore"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_streaming_and_cuda_restore_are_independent(
    streaming: bool, cuda_restore: bool
) -> None:
    kwargs: dict[str, object] = {
        "streaming_snapshots": streaming,
        "cuda_restore": cuda_restore,
    }
    if streaming:
        kwargs.update(
            streaming_native_library="/opt/sparkcache/lib/libsnapshot.so",
            streaming_native_library_sha256="a" * 64,
        )
    if cuda_restore:
        kwargs.update(
            cuda_placement_library="/opt/sparkcache/lib/libplacement.so",
            cuda_placement_library_sha256="b" * 64,
        )
    transformed = transform_inspection(_source_inspection(), **kwargs)
    extra = _connector_extra(transformed)

    assert extra["spark_cache_streaming_snapshots"] is streaming
    assert extra["spark_cache_cuda_restore"] is cuda_restore
    assert extra["spark_cache_max_bytes"] == 200 * 1024**3
    assert extra["spark_cache_low_watermark_bytes"] == 180 * 1024**3
    assert ("spark_cache_streaming_native_library" in extra) is streaming
    assert ("spark_cache_cuda_placement_library" in extra) is cuda_restore
    if streaming:
        assert extra["spark_cache_streaming_timing"] == 0


def test_streaming_timing_uses_the_runtime_zero_or_one_contract() -> None:
    transformed = transform_inspection(
        _source_inspection(),
        streaming_snapshots=True,
        streaming_native_library="/opt/sparkcache/lib/libsnapshot.so",
        streaming_native_library_sha256="a" * 64,
        streaming_timing=True,
    )
    assert _connector_extra(transformed)["spark_cache_streaming_timing"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"streaming_snapshots": True},
        {"cuda_restore": True},
        {
            "streaming_native_library": "/opt/libsnapshot.so",
            "streaming_native_library_sha256": "a" * 64,
        },
    ],
)
def test_cuda_feature_configuration_is_rejected(kwargs: dict) -> None:
    with pytest.raises(ProfileTransformError):
        transform_inspection(_source_inspection(), **kwargs)


def test_legacy_profile_names_warn_and_emit_canonical_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(glm_profile, "_LEGACY_CUDA_RESTORE_WARNING_EMITTED", False)
    with pytest.warns(FutureWarning, match="CUDA restore"):
        transformed = transform_inspection(
            _source_inspection(),
            native_restore=True,
            native_restore_library="/opt/sparkcache/lib/libplacement.so",
            native_restore_library_sha256="b" * 64,
        )
    extra = _connector_extra(transformed)
    assert extra["spark_cache_cuda_restore"] is True
    assert "spark_cache_native_restore" not in extra


def test_conflicting_profile_aliases_are_rejected() -> None:
    with pytest.raises(ProfileTransformError, match="conflicting"):
        transform_inspection(
            _source_inspection(),
            cuda_restore=True,
            native_restore=False,
        )


def test_instance_ports_are_overridden_together_and_cannot_collide() -> None:
    transformed = transform_inspection(
        _source_inspection(), api_port=8100, master_port=29600
    )
    arguments = transformed["Config"]["Cmd"]
    assert _option_value(arguments, "--port") == "8100"
    assert _option_value(arguments, "--master-port") == "29600"
    assert _environment(transformed)["PORT"] == "8100"
    assert _environment(transformed)["MASTER_PORT"] == "29600"
    assert _option_value(arguments, "--master-addr") == "192.0.2.10"

    with pytest.raises(ProfileTransformError, match="must differ"):
        transform_inspection(_source_inspection(), api_port=8100, master_port=8100)

    for kwargs in (
        {"api_port": 11100},
        {"master_port": 11100},
        {"api_port": 10211},
    ):
        with pytest.raises(ProfileTransformError, match="collective port"):
            transform_inspection(_source_inspection(), **kwargs)


def test_direct_and_underscore_wrapped_r7_commands_transform_identically() -> None:
    direct = _source_inspection()
    wrapped = copy.deepcopy(direct)
    wrapped["Config"]["Cmd"] = [
        "-c",
        "underscore shell wrapper",
        "_",
        *direct["Config"]["Cmd"],
    ]
    assert (
        transform_inspection(wrapped)["Config"]["Cmd"]
        == (transform_inspection(direct)["Config"]["Cmd"])
    )


@pytest.mark.parametrize(
    ("target", "replacement", "match"),
    [
        ("--decode-context-parallel-size", "2", "decode-context"),
        ("--max-model-len", "65536", "max-model-len"),
    ],
)
def test_transform_rejects_source_profile_drift(
    target: str, replacement: str, match: str
) -> None:
    inspection = _source_inspection()
    arguments = inspection["Config"]["Cmd"]
    arguments[arguments.index(target) + 1] = replacement
    with pytest.raises(ProfileTransformError, match=match):
        transform_inspection(inspection)


def test_transform_validates_attested_checkpoint_pins() -> None:
    inspection = _source_inspection()
    environment = inspection["Config"]["Env"]
    index = next(
        index
        for index, value in enumerate(environment)
        if value.startswith("SPARKRING_ATTEST_MODEL_REVISION=")
    )
    environment[index] = "SPARKRING_ATTEST_MODEL_REVISION=" + "0" * 40
    with pytest.raises(ProfileTransformError, match="ATTEST_MODEL_REVISION"):
        transform_inspection(inspection)


@pytest.mark.parametrize(
    ("key", "replacement"),
    [
        ("model", "/models/other"),
        ("draft_tensor_parallel_size", 2),
        ("quantization", "none"),
        ("moe_backend", "triton"),
        ("attention_backend", "OTHER"),
        ("use_local_argmax_reduction", True),
    ],
)
def test_transform_rejects_draft_contract_drift(key: str, replacement: object) -> None:
    inspection = _source_inspection()
    arguments = inspection["Config"]["Cmd"]
    index = arguments.index("--speculative-config") + 1
    speculative = json.loads(arguments[index])
    speculative[key] = replacement
    arguments[index] = _json(speculative)
    with pytest.raises(ProfileTransformError, match=key):
        transform_inspection(inspection)


def test_prefix_cache_flags_preserve_native_default_or_explicit_enable() -> None:
    implicit = transform_inspection(_source_inspection())["Config"]["Cmd"]
    assert "--enable-prefix-caching" not in implicit
    assert "--disable-prefix-caching" not in implicit

    enabled_source = _source_inspection()
    enabled_source["Config"]["Cmd"].append("--enable-prefix-caching")
    enabled = transform_inspection(enabled_source)["Config"]["Cmd"]
    assert enabled.count("--enable-prefix-caching") == 1

    disabled_source = _source_inspection()
    disabled_source["Config"]["Cmd"].append("--disable-prefix-caching")
    with pytest.raises(ProfileTransformError, match="native prefix-cache default"):
        transform_inspection(disabled_source)


def test_explicit_block_size_is_preserved_and_other_values_are_rejected() -> None:
    inspection = _source_inspection()
    inspection["Config"]["Cmd"].extend(("--block-size", "64"))
    transformed = transform_inspection(inspection)
    assert transformed["Config"]["Cmd"].count("--block-size") == 1
    assert _option_value(transformed["Config"]["Cmd"], "--block-size") == "64"

    inspection["Config"]["Cmd"][
        inspection["Config"]["Cmd"].index("--block-size") + 1
    ] = "16"
    with pytest.raises(ProfileTransformError, match="block size 64"):
        transform_inspection(inspection)


def test_containerfile_inherits_r7_entrypoint_and_applies_only_glm_patches() -> None:
    containerfile = (
        Path(__file__).resolve().parents[1] / "deploy/glm52_35bpw/Containerfile"
    ).read_text(encoding="utf-8")
    instructions = [
        line.strip().split(maxsplit=1)[0].upper()
        for line in containerfile.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert "ENTRYPOINT" not in instructions
    assert "011-sparkcache-glm52-async-rollback.patch" in containerfile
    assert "010-sparkcache-async-rollback.patch" not in containerfile
    assert "020-sparkcache-vmm-exemption.patch" in containerfile
    assert "030-sparkcache-hma-load-failure.patch" not in containerfile
    assert containerfile.splitlines()[0] == "ARG BASE_IMAGE"


def test_prepare_vllm_overlays_patches_exact_preimages(
    monkeypatch, tmp_path: Path
) -> None:
    repository = tmp_path / "repository"
    package = repository / "sparkcache"
    package.mkdir(parents=True)
    (package / "connector.py").write_text("# source\n", encoding="utf-8")
    vllm_root = tmp_path / "source"
    source = vllm_root / "vllm/example.py"
    source.parent.mkdir(parents=True)
    source.write_text("old\n", encoding="utf-8", newline="\n")
    patch = repository / "patches/example.patch"
    patch.parent.mkdir(parents=True)
    patch.write_bytes(
        b"diff --git a/vllm/example.py b/vllm/example.py\r\n"
        b"--- a/vllm/example.py\r\n"
        b"+++ b/vllm/example.py\r\n"
        b"@@ -1 +1 @@\r\n"
        b"-old\r\n"
        b"+new\r\n"
    )
    monkeypatch.setattr(
        prepare_vllm_overlays,
        "OVERLAYS",
        (
            prepare_vllm_overlays.OverlaySpec(
                source="vllm/example.py",
                patch="patches/example.patch",
                output="example.py",
                preimage_sha256=hashlib.sha256(b"old\n").hexdigest(),
                postimage_sha256=hashlib.sha256(b"new\n").hexdigest(),
            ),
        ),
    )
    output = tmp_path / "overlays"

    receipt = prepare_vllm_overlays.prepare(
        vllm_root,
        repository,
        output,
    )

    assert receipt["schema"] == "sparkcache-glm52-r7-vllm-overlays/v1"
    assert [record["disposition"] for record in receipt["files"]] == ["patched"]
    assert {path.name for path in output.iterdir()} == {"example.py", "receipt.json"}
    assert (output / "example.py").read_bytes() == b"new\n"
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        prepare_vllm_overlays.prepare(
            vllm_root,
            repository,
            output,
        )
