from __future__ import annotations

import json
import re
from pathlib import Path
from unittest import mock

import pytest

from deploy.glm53_flash.publish_image import PublishError, load_json, validate_destination


def test_destination_is_semantic_ghcr_tag() -> None:
    validate_destination(
        "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:da4d7be-dflash2-bf16-arm64"
    )
    with pytest.raises(PublishError, match="latest"):
        validate_destination("ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:latest")
    with pytest.raises(PublishError, match="semantic"):
        validate_destination("docker.io/example/image:release")
    with pytest.raises(PublishError, match="SparkCache GLM-5.3"):
        validate_destination("ghcr.io/fujitsupolycom/other-image:release")


def test_load_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(PublishError, match="one JSON object"):
        load_json(path, "receipt")


def _write_build_receipt(path: Path, image_id: str) -> None:
    path.write_text(
        json.dumps(
            {
                "schema": "sparkcache-glm53-public-image/v1",
                "image": {"image_id": image_id},
            }
        ),
        encoding="utf-8",
    )


def test_publish_generates_sbom_from_immutable_image_id_before_push(
    tmp_path: Path,
) -> None:
    from deploy.glm53_flash import publish_image as publication

    image_id = "sha256:" + "1" * 64
    destination = (
        "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:"
        "da4d7be-dflash2-bf16-arm64"
    )
    build_receipt = tmp_path / "build.json"
    sbom = tmp_path / "image.spdx.json"
    _write_build_receipt(build_receipt, image_id)
    calls: list[tuple[str, ...]] = []

    def fake_run(argv) -> str:
        arguments = tuple(argv)
        calls.append(arguments)
        if arguments[:4] == (
            "docker",
            "image",
            "inspect",
            "--format",
        ):
            if arguments[-1] == "local-image":
                return image_id
            return json.dumps(
                [
                    "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache@sha256:"
                    + "2" * 64
                ]
            )
        if arguments[0] == "syft":
            assert arguments == (
                "syft",
                "scan",
                f"docker:{image_id}",
                "--output",
                f"spdx-json={sbom}",
            )
            sbom.write_text(
                json.dumps({"spdxVersion": "SPDX-2.3"}), encoding="utf-8"
            )
        return ""

    with mock.patch.object(publication, "run", side_effect=fake_run):
        receipt = publication.publish(
            image="local-image",
            destination=destination,
            build_receipt_path=build_receipt,
            sbom_path=sbom,
        )

    syft_index = next(index for index, call in enumerate(calls) if call[0] == "syft")
    push_index = next(
        index for index, call in enumerate(calls) if call[:2] == ("docker", "push")
    )
    assert syft_index < push_index
    assert receipt["sbom"]["source_image_id"] == image_id
    assert receipt["sbom"]["generator"] == "syft"


def test_publish_refuses_to_overwrite_an_existing_sbom(tmp_path: Path) -> None:
    from deploy.glm53_flash import publish_image as publication

    image_id = "sha256:" + "1" * 64
    build_receipt = tmp_path / "build.json"
    sbom = tmp_path / "existing.spdx.json"
    _write_build_receipt(build_receipt, image_id)
    sbom.write_text("operator-reviewed", encoding="utf-8")

    with mock.patch.object(publication, "run") as run:
        with pytest.raises(publication.PublishError, match="already exists"):
            publication.publish(
                image="local-image",
                destination=(
                    "ghcr.io/fujitsupolycom/sparkring-glm53-sparkcache:release"
                ),
                build_receipt_path=build_receipt,
                sbom_path=sbom,
            )
    run.assert_not_called()


def test_community_announcement_has_the_required_support_contract() -> None:
    announcement = Path(__file__).with_name("IMAGE_ANNOUNCEMENT.md").read_text(
        encoding="utf-8"
    )
    for heading in (
        "## Status",
        "## Image and digest",
        "## Based on",
        "## Build recipe",
        "## Source commits, pull requests, patches, and overlays",
        "## Changes from the base image",
        "## Tested configuration",
        "## Validation commands",
        "## Validation results",
        "## Performance claims",
        "## Known limitations",
        "## Support contact or issue tracker",
        "## Upstream useful work",
    ):
        assert heading in announcement
    assert "Experimental community derivative" in announcement
    assert "not the recommended community image" in announcement
    assert "UNKNOWN — needs verification" in announcement
    assert "Not tested" in announcement
    assert "@sha256:" in announcement
    assert "Support owner: `FujitsuPolycom`" in announcement
    assert "Dedicated Discord support thread:" in announcement
    assert re.search(
        r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)",
        announcement,
    ) is None
    assert re.search(r"(?i)\b[A-Z]:\\(?:Users|home)\\", announcement) is None
    assert not any(
        secret in announcement
        for secret in ("HF_TOKEN=", "GH_TOKEN=", "PASSWORD=", "PRIVATE_KEY=")
    )
