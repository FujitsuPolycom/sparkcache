from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from deploy.glm53_flash import build_public_image as public


BASE = "ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:" + "1" * 64


def test_public_builder_rejects_moving_parent_tag() -> None:
    with pytest.raises(public.PublicBuildError, match="immutable"):
        public.build_public_image(
            repository=Path("/repo"),
            base_image="ghcr.io/example/runtime:moving-tag",
            output_image="sparkcache:implemented",
        )


def test_clean_source_scope_returns_revision() -> None:
    with mock.patch(
        "deploy.glm53_flash.build_public_image.run",
        side_effect=["2" * 40, ""],
    ) as called:
        assert public.require_clean_sources(Path("/repo")) == "2" * 40
    assert called.call_count == 2
    status_command = called.call_args_list[1].args[0]
    assert "deploy/deployment_contract/source.py" in status_command


def test_parent_license_boundary_is_explicit() -> None:
    assert public.EXPECTED_LICENSES == (
        "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0 AND BSD-3-Clause"
    )


def _valid_parent() -> dict:
    labels = dict(public.EXPECTED_PARENT_LABELS)
    labels.update(
        {
            "org.opencontainers.image.revision": "3" * 40,
            "org.sparkring.source-receipt-sha256": "4" * 64,
        }
    )
    return {
        "Architecture": "arm64",
        "Os": "linux",
        "Config": {"Labels": labels},
    }


def test_parent_contract_binds_source_platform_and_transport() -> None:
    labels = public.validate_parent(_valid_parent())
    assert labels["org.jovian.vllm.commit"] == (
        "da4d7be6c97434f6942292ed8abbf4b32dc44355"
    )
    assert labels["org.jovian.transport"] == (
        "sparkring-nccl-2.30.7-source-built"
    )


def test_parent_contract_rejects_component_or_source_identity_drift() -> None:
    parent = _valid_parent()
    parent["Config"]["Labels"]["org.jovian.transport"] = "stock-nccl"
    with pytest.raises(public.PublicBuildError, match="org.jovian.transport drift"):
        public.validate_parent(parent)

    parent = _valid_parent()
    parent["Config"]["Labels"]["org.opencontainers.image.revision"] = "moving"
    with pytest.raises(public.PublicBuildError, match="immutable identity"):
        public.validate_parent(parent)


def test_script_help_runs_from_outside_the_repository(tmp_path: Path) -> None:
    script = Path(public.__file__).resolve()
    completed = subprocess.run(
        (sys.executable, str(script), "--help"),
        cwd=tmp_path,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert completed.returncode == 0, completed.stderr
