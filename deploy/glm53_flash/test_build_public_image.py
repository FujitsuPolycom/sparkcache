from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from deploy.glm53_flash import build_public_image as public


BASE = "ghcr.io/fujitsupolycom/sparkring-glm53-runtime@sha256:" + "1" * 64


def test_public_builder_rejects_moving_parent_tag() -> None:
    with pytest.raises(public.PublicBuildError, match="registry reference"):
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


def test_parent_license_boundary_is_explicit() -> None:
    assert public.EXPECTED_LICENSES == (
        "LicenseRef-NVIDIA-Deep-Learning-Container AND Apache-2.0 AND BSD-3-Clause"
    )
