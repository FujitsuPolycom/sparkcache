from __future__ import annotations

import json
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
