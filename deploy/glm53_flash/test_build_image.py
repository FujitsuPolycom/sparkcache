from __future__ import annotations

import subprocess
from pathlib import Path
from unittest import mock

import pytest

from deploy.glm53_flash.build_image import build_command


IMAGE_ID = "sha256:" + "1" * 64
SOURCE_ID = "2" * 64
SPARKCACHE_REVISION = "4" * 40
LICENSES = "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0"


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
            sparkcache_revision=SPARKCACHE_REVISION,
            base_image_licenses=LICENSES,
            output_image="glm53-sparkcache:qualified",
        )

    assert f"BASE_IMAGE_ID={IMAGE_ID}" in command
    assert f"SPARKCACHE_SOURCE_SHA256={SOURCE_ID}" in command
    assert f"SPARKCACHE_REVISION={SPARKCACHE_REVISION}" in command
    assert f"BASE_IMAGE_LICENSES={LICENSES}" in command
    assert Path(command[-1]) == Path("/repo")


def test_build_command_selects_the_e105_source_built_recipe(tmp_path: Path) -> None:
    recipe = tmp_path / "deploy/glm53_flash/Containerfile.e10536a"
    recipe.parent.mkdir(parents=True)
    recipe.write_text("FROM scratch\n", encoding="utf-8")
    with mock.patch(
        "deploy.glm53_flash.build_image.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, IMAGE_ID + "\n", ""),
    ):
        command = build_command(
            repository=tmp_path,
            base_image="local-glm53:e10536a",
            base_image_id=IMAGE_ID,
            source_sha256=SOURCE_ID,
            sparkcache_revision=SPARKCACHE_REVISION,
            base_image_licenses=LICENSES,
            output_image="glm53-sparkcache:e10536a",
            containerfile="deploy/glm53_flash/Containerfile.e10536a",
        )

    assert command[command.index("-f") + 1] == str(recipe.resolve())


def test_build_command_selects_the_b12x_kda_adaptive_mtp_recipe(
    tmp_path: Path,
) -> None:
    relative = "deploy/glm53_flash/Containerfile.b12x-kda-adaptive-mtp"
    recipe = tmp_path / relative
    recipe.parent.mkdir(parents=True)
    recipe.write_text("FROM scratch\n", encoding="utf-8")
    with mock.patch(
        "deploy.glm53_flash.build_image.subprocess.run",
        return_value=subprocess.CompletedProcess([], 0, IMAGE_ID + "\n", ""),
    ):
        command = build_command(
            repository=tmp_path,
            base_image="local-glm53:b12x-kda-adaptive-mtp",
            base_image_id=IMAGE_ID,
            source_sha256=SOURCE_ID,
            sparkcache_revision=SPARKCACHE_REVISION,
            base_image_licenses=LICENSES,
            output_image="glm53-sparkcache:b12x-kda-adaptive-mtp",
            containerfile=relative,
        )

    assert command[command.index("-f") + 1] == str(recipe.resolve())


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
                sparkcache_revision=SPARKCACHE_REVISION,
                base_image_licenses=LICENSES,
                output_image="glm53-sparkcache:qualified",
            )


def test_build_command_rejects_missing_revision_or_license() -> None:
    arguments = {
        "repository": Path("/repo"),
        "base_image": "local-glm53:source-locked",
        "base_image_id": IMAGE_ID,
        "source_sha256": SOURCE_ID,
        "sparkcache_revision": SPARKCACHE_REVISION,
        "base_image_licenses": LICENSES,
        "output_image": "glm53-sparkcache:implemented",
    }
    with pytest.raises(ValueError, match="40-character Git commit"):
        build_command(**{**arguments, "sparkcache_revision": "short"})
    with pytest.raises(ValueError, match="parent image terms"):
        build_command(**{**arguments, "base_image_licenses": ""})


def test_containerfile_records_revision_license_and_notice() -> None:
    recipe = (
        Path(__file__).with_name("Containerfile").read_text(encoding="utf-8")
    )
    assert 'org.opencontainers.image.revision="${SPARKCACHE_REVISION}"' in recipe
    assert 'org.opencontainers.image.licenses="${BASE_IMAGE_LICENSES}"' in recipe
    assert "COPY LICENSE /usr/share/licenses/SparkCache/LICENSE" in recipe
