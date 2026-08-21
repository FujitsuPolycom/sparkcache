from __future__ import annotations

import json
import hashlib
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from deploy.deepseek_v4.capacity_gate import (
    _chunk as capacity_chunk,
    _identity as capacity_identity,
    run as run_capacity_gate,
)
from deploy.deepseek_v4.corruption_gate import run as run_corruption_gate
from deploy.deepseek_v4.install_overlay import _sha256, _tree_sha256, install
from deploy.deepseek_v4.launch_from_inspect import _vllm_args, launch
from deploy.deepseek_v4.prepare_context import _write_rootfs_tar
from deploy.deepseek_v4.semantic_gate import build_long_prompt, run_hit, run_miss


def test_vllm_args_drop_the_underscore_shell_wrapper() -> None:
    command = [
        "-c",
        'exec bash /leg3pair-inner.sh "$@"',
        "_",
        "serve",
        "/models/checkpoint",
        "--port",
        "8000",
    ]
    assert _vllm_args(command) == [
        "serve",
        "/models/checkpoint",
        "--port",
        "8000",
    ]


@pytest.mark.parametrize(
    "command",
    [
        ["serve", "/models/checkpoint", "--port", "8000"],
        ["/opt/venv/bin/vllm", "serve", "/models/checkpoint", "--port", "8000"],
    ],
)
def test_vllm_args_accept_direct_r7_commands(command: list[str]) -> None:
    assert _vllm_args(command) == [
        "serve",
        "/models/checkpoint",
        "--port",
        "8000",
    ]


def test_create_only_launcher_keeps_only_data_mounts(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        "deploy.deepseek_v4.launch_from_inspect.subprocess.run",
        lambda command, check: observed.append((command, check)),
    )
    inspection = {
        "Config": {
            "Env": ["RANK=0"],
            "Cmd": ["-c", "wrapper", "_", "serve", "/models/checkpoint"],
        },
        "HostConfig": {"NetworkMode": "host", "IpcMode": "host", "ShmSize": 16},
        "Mounts": [
            {"Source": "/data/model", "Destination": "/models/checkpoint", "RW": False},
            {"Source": "/private/code.py", "Destination": "/opt/code.py", "RW": False},
        ],
    }

    launch(
        inspection,
        "image:test",
        "created",
        "a" * 64,
        create_only=True,
        sparkcache_root="/cache/sparkcache-bounded",
        max_bytes=200,
        low_watermark_bytes=180,
        ttl_seconds=3600,
    )

    command, check = observed[0]
    assert command[:2] == ["docker", "create"]
    assert "/data/model:/models/checkpoint:ro" in command
    assert "/private/code.py:/opt/code.py:ro" not in command
    assert "SPARKCACHE_ROOT=/cache/sparkcache-bounded" in command
    assert "SPARKCACHE_MAX_BYTES=200" in command
    assert "SPARKCACHE_LOW_WATERMARK_BYTES=180" in command
    assert "SPARKCACHE_TTL_SECONDS=3600" in command
    assert check is True


def test_launcher_adds_validated_explicit_read_write_bind(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        "deploy.deepseek_v4.launch_from_inspect.subprocess.run",
        lambda command, check: observed.append((command, check)),
    )
    inspection = {
        "Config": {
            "Env": [],
            "Cmd": ["serve", "/models/checkpoint"],
        },
        "HostConfig": {},
        "Mounts": [],
    }

    launch(
        inspection,
        "image:test",
        "created",
        "a" * 64,
        create_only=True,
        extra_binds=(
            ("/host-cache/glm52-r0", "/cache/sparkcache-glm52-r7"),
        ),
    )

    command, check = observed[0]
    assert "/host-cache/glm52-r0:/cache/sparkcache-glm52-r7:rw" in command
    assert check is True


def test_launcher_adds_explicit_read_only_bind(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        "deploy.deepseek_v4.launch_from_inspect.subprocess.run",
        lambda command, check: observed.append((command, check)),
    )
    inspection = {
        "Config": {"Env": [], "Cmd": ["serve", "/models/checkpoint"]},
        "HostConfig": {},
        "Mounts": [],
    }

    launch(
        inspection,
        "image:test",
        "created",
        "a" * 64,
        create_only=True,
        extra_binds=(("/host/code", "/opt/sparkcache-src/sparkcache", True),),
    )

    assert "/host/code:/opt/sparkcache-src/sparkcache:ro" in observed[0][0]


def test_launcher_adds_explicit_entrypoint_and_sorted_labels(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        "deploy.deepseek_v4.launch_from_inspect.subprocess.run",
        lambda command, check: observed.append((command, check)),
    )
    inspection = {
        "Config": {"Env": [], "Cmd": ["serve", "/models/checkpoint"]},
        "HostConfig": {},
        "Mounts": [],
    }

    launch(
        inspection,
        "image:test",
        "created",
        "a" * 64,
        create_only=True,
        entrypoint="/opt/venv/bin/vllm",
        labels={"z.example": "last", "a.example": "first"},
    )

    command = observed[0][0]
    assert command[command.index("--entrypoint") + 1] == "/opt/venv/bin/vllm"
    labels = [command[index + 1] for index, value in enumerate(command) if value == "--label"]
    assert labels == ["a.example=first", "z.example=last"]


def test_launcher_preserves_all_inspected_bind_mounts_for_r7(monkeypatch) -> None:
    observed = []
    monkeypatch.setattr(
        "deploy.deepseek_v4.launch_from_inspect.subprocess.run",
        lambda command, check: observed.append((command, check)),
    )
    inspection = {
        "Config": {"Env": [], "Cmd": ["serve", "/models/checkpoint"]},
        "HostConfig": {},
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/host/sircl/spark_tp4_backend.py",
                "Destination": "/opt/spark-vllm/spark_tp4_backend.py",
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "/host/vllm/model_runner.py",
                "Destination": (
                    "/opt/venv/lib/python3.12/site-packages/"
                    "vllm/v1/worker/gpu/model_runner.py"
                ),
                "RW": False,
            },
            {
                "Type": "bind",
                "Source": "/host/jit",
                "Destination": "/cache/jit",
                "RW": True,
            },
        ],
    }

    launch(
        inspection,
        "image:test",
        "created",
        "a" * 64,
        create_only=True,
        extra_binds=(("/host/sparkcache", "/cache/sparkcache-glm52-r7"),),
        preserve_all_binds=True,
    )

    command, check = observed[0]
    assert "/host/sircl/spark_tp4_backend.py:/opt/spark-vllm/spark_tp4_backend.py:ro" in command
    assert (
        "/host/vllm/model_runner.py:/opt/venv/lib/python3.12/site-packages/"
        "vllm/v1/worker/gpu/model_runner.py:ro"
    ) in command
    assert "/host/jit:/cache/jit:rw" in command
    assert "/host/sparkcache:/cache/sparkcache-glm52-r7:rw" in command
    assert check is True


def test_preserve_all_binds_rejects_duplicate_extra_destination() -> None:
    inspection = {
        "Config": {"Env": [], "Cmd": ["serve", "model"]},
        "HostConfig": {},
        "Mounts": [
            {
                "Type": "bind",
                "Source": "/host/cache-a",
                "Destination": "/cache/sparkcache-glm52-r7",
                "RW": True,
            }
        ],
    }
    with pytest.raises(ValueError, match="duplicate extra bind destination"):
        launch(
            inspection,
            "image",
            "name",
            "a" * 64,
            create_only=True,
            extra_binds=(("/host/cache-b", "/cache/sparkcache-glm52-r7"),),
            preserve_all_binds=True,
        )


@pytest.mark.parametrize(
    "mount",
    [
        {
            "Type": "volume",
            "Source": "/host/cache",
            "Destination": "/cache/jit",
            "RW": True,
        },
        {
            "Type": "bind",
            "Source": "relative/cache",
            "Destination": "/cache/jit",
            "RW": True,
        },
        {
            "Type": "bind",
            "Source": "/host/cache",
            "Destination": "cache/jit",
            "RW": True,
        },
        {
            "Type": "bind",
            "Source": "/host/cache",
            "Destination": "/cache/jit",
            "RW": "true",
        },
    ],
)
def test_preserve_all_binds_rejects_non_bind_or_malformed_mount(mount: dict) -> None:
    inspection = {
        "Config": {"Env": [], "Cmd": ["serve", "model"]},
        "HostConfig": {},
        "Mounts": [mount],
    }
    with pytest.raises(ValueError):
        launch(
            inspection,
            "image",
            "name",
            "a" * 64,
            create_only=True,
            preserve_all_binds=True,
        )


@pytest.mark.parametrize(
    "bind",
    [
        ("relative/cache", "/cache/sparkcache-glm52-r7"),
        ("/host/cache", "cache/sparkcache-glm52-r7"),
        ("/host/../cache", "/cache/sparkcache-glm52-r7"),
        ("/host/cache", "/cache/../sparkcache-glm52-r7"),
        ("/", "/cache/sparkcache-glm52-r7"),
        ("/host/cache", "/"),
    ],
)
def test_launcher_rejects_unsafe_explicit_bind(bind: tuple[str, str]) -> None:
    inspection = {
        "Config": {"Env": [], "Cmd": ["serve", "model"]},
        "HostConfig": {},
        "Mounts": [],
    }
    with pytest.raises(ValueError, match="normalized absolute POSIX path"):
        launch(
            inspection,
            "image",
            "name",
            "a" * 64,
            create_only=True,
            extra_binds=(bind,),
        )


def test_launcher_rejects_unsafe_capacity_geometry() -> None:
    inspection = {
        "Config": {"Env": [], "Cmd": ["_", "serve", "model"]},
        "HostConfig": {},
        "Mounts": [],
    }
    for kwargs in (
        {"sparkcache_root": "/cache/../outside"},
        {"low_watermark_bytes": 1},
        {"max_bytes": 10, "low_watermark_bytes": 11},
    ):
        with pytest.raises(ValueError):
            launch(inspection, "image", "name", "a" * 64, **kwargs)


def test_overlay_installer_verifies_and_copies_files(tmp_path: Path) -> None:
    context = tmp_path / "context"
    artifacts = context / "artifacts"
    file_artifact = artifacts / "000"
    directory_artifact = artifacts / "001"
    directory_artifact.mkdir(parents=True)
    file_artifact.write_bytes(b"scheduler")
    (directory_artifact / "library.so").write_bytes(b"library")
    file_destination = tmp_path / "installed" / "scheduler.py"
    directory_destination = tmp_path / "installed" / "runtime"
    manifest = {
        "schema": "sparkcache-private-overlay/v1",
        "artifacts": [
            {
                "artifact": "000",
                "destination": str(file_destination),
                "kind": "file",
                "sha256": _sha256(file_artifact),
            },
            {
                "artifact": "001",
                "destination": str(directory_destination),
                "kind": "directory",
                "sha256": _tree_sha256(directory_artifact),
            },
        ],
    }
    (context / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    install(context)

    assert file_destination.read_bytes() == b"scheduler"
    assert (directory_destination / "library.so").read_bytes() == b"library"


def test_rootfs_tar_normalizes_identity_metadata(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    deploy = repo / "deploy/deepseek_v4"
    deploy.mkdir(parents=True)
    (deploy / "install_overlay.py").write_text("installer\n", encoding="utf-8")
    (deploy / "entrypoint.sh").write_text("#!/bin/sh\n", encoding="utf-8")
    output = tmp_path / "context"
    (output / "sparkcache").mkdir(parents=True)
    (output / "sparkcache/connector.py").write_text("connector\n", encoding="utf-8")
    (output / "overlay").mkdir()
    (output / "overlay/manifest.json").write_text("{}\n", encoding="utf-8")

    _write_rootfs_tar(output, repo)

    with tarfile.open(output / "rootfs.tar") as archive:
        members = archive.getmembers()
    assert any(member.name == "opt/sparkcache-src/sparkcache/connector.py" for member in members)
    assert all(member.uid == 0 and member.gid == 0 and member.mtime == 0 for member in members)


def test_semantic_gate_requires_stable_long_answer_and_post_hit_canary(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "reference.json"
    responses = iter(
        [
            {"choices": [{"message": {"content": "SPARKCACHE_OK:9540"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_CANARY_OK"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_OK:9540"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_CANARY_OK"}}]},
        ]
    )
    monkeypatch.setattr(
        "deploy.deepseek_v4.semantic_gate._request",
        lambda endpoint, model, prompt, max_tokens: next(responses),
    )

    miss = run_miss("http://stack", "dsv4-flash", reference)
    hit = run_hit("http://stack", "dsv4-flash", reference)

    assert miss["content"] == "SPARKCACHE_OK:9540"
    assert hit == {
        "content": "SPARKCACHE_OK:9540",
        "post_restore_canary": "SPARKCACHE_CANARY_OK",
    }
    assert len(build_long_prompt().split()) > 3_000


def test_semantic_gate_scales_prompt_and_pins_size_in_reference(
    monkeypatch, tmp_path: Path
) -> None:
    reference = tmp_path / "large-reference.json"
    prompts: list[str] = []
    responses = iter(
        [
            {"choices": [{"message": {"content": "SPARKCACHE_OK:9540"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_CANARY_OK"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_OK:9540"}}]},
            {"choices": [{"message": {"content": "SPARKCACHE_CANARY_OK"}}]},
        ]
    )

    def request(endpoint, model, prompt, max_tokens):
        prompts.append(prompt)
        return next(responses)

    monkeypatch.setattr("deploy.deepseek_v4.semantic_gate._request", request)
    miss = run_miss("http://stack", "model", reference, records=4096)
    hit = run_hit("http://stack", "model", reference)
    assert miss["records"] == 4096
    assert hit["content"] == "SPARKCACHE_OK:9540"
    assert prompts[0] == prompts[2]
    assert len(prompts[0]) > len(build_long_prompt()) * 10

    with pytest.raises(RuntimeError, match="record count"):
        run_hit("http://stack", "model", reference, records=2048)


def test_capacity_gate_evicts_to_physical_byte_limit(tmp_path: Path) -> None:
    result = run_capacity_gate(tmp_path / "capacity-gate")
    assert result["capacity_satisfied"] is True
    assert result["manifests_evicted"] == 2
    assert result["orphan_chunks_deleted"] == 1
    assert result["survivors"] == 1


def test_corruption_gate_damages_only_a_disposable_copy(tmp_path: Path) -> None:
    source = tmp_path / "source"
    identity = capacity_identity()
    digest = hashlib.sha256(b"corruption-gate").hexdigest()
    from sparkcache.spark_context_cache_store import ManifestStore

    source_store = ManifestStore(source)
    source_store.commit(
        identity=identity,
        context_digest=digest,
        chunks=[capacity_chunk(7)],
        span_tokens=256,
    )

    result = run_corruption_gate(source, tmp_path / "damaged-copy")

    assert result["corruption_reason"] == "corrupt"
    assert result["invalidated"] is True
    assert result["absent_after_invalidation"] is True
    assert source_store.lookup(identity, digest, verify_chunks=True).is_hit


def test_capacity_gate_documented_script_invocation(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    completed = subprocess.run(
        [
            sys.executable,
            "deploy/deepseek_v4/capacity_gate.py",
            "--root",
            str(tmp_path / "script-capacity-gate"),
        ],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )

    assert json.loads(completed.stdout)["schema"] == "sparkcache-capacity-gate/v1"
