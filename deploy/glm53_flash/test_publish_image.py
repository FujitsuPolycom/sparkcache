from __future__ import annotations

import json
import re
from pathlib import Path

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


def test_load_json_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "receipt.json"
    path.write_text(json.dumps([]), encoding="utf-8")
    with pytest.raises(PublishError, match="one JSON object"):
        load_json(path, "receipt")


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
    assert re.search(r"\b(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)", announcement) is None
    assert re.search(r"(?i)\b[A-Z]:\\(?:Users|home)\\", announcement) is None
    assert not any(
        secret in announcement
        for secret in ("HF_TOKEN=", "GH_TOKEN=", "PASSWORD=", "PRIVATE_KEY=")
    )
