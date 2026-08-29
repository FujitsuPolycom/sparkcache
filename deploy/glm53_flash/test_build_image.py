from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from deploy.glm53_flash.build_image import build_command


IMAGE_ID = "sha256:" + "1" * 64
SOURCE_ID = "2" * 64


def test_build_command_binds_inspected_parent_and_source_digest() -> None:
    with mock.patch(
        "deploy.glm53_flash.build_image.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, IMAGE_ID + "\n", ""),
    ):
        command = build_command(
            repository=Path("/repo"),
            base_image="local-glm53:source-locked",
            base_image_id=IMAGE_ID,
            source_sha256=SOURCE_ID,
            output_image="glm53-sparkcache:qualified",
        )

    assert f"BASE_IMAGE_ID={IMAGE_ID}" in command
    assert f"SPARKCACHE_SOURCE_SHA256={SOURCE_ID}" in command
    assert Path(command[-1]) == Path("/repo")


def test_build_command_rejects_parent_identity_drift() -> None:
    with mock.patch(
        "deploy.glm53_flash.build_image.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, "sha256:" + "3" * 64, ""),
    ):
        with pytest.raises(RuntimeError, match="base image identity mismatch"):
            build_command(
                repository=Path("/repo"),
                base_image="local-glm53:source-locked",
                base_image_id=IMAGE_ID,
                source_sha256=SOURCE_ID,
                output_image="glm53-sparkcache:qualified",
            )
