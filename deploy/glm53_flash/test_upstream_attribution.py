from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]
JJ_URL = "https://github.com/local-inference-lab/vllm"
B12X_URL = "https://github.com/local-inference-lab/b12x"
TARGET_URL = "https://huggingface.co/local-inference-lab/GLM-5.3-Flash-NVFP4"
MXFP8_DRAFT_URL = (
    "https://huggingface.co/local-inference-lab/GLM-5.3-Flash-DFlash2-MXFP8"
)
BF16_DRAFT_URL = "https://huggingface.co/incoai/GLM-5.3-Flash-DFlash2"


def test_glm_documentation_credits_exact_upstream_sources_and_artifacts() -> None:
    documents = (
        ROOT / "README.md",
        ROOT / "deploy/glm53_flash/README.md",
        ROOT / "deploy/glm53_flash/IMAGE_ANNOUNCEMENT.md",
        ROOT / "deploy/glm53_flash/JJ_R7_ARM64_IMAGE.md",
    )
    for path in documents:
        text = path.read_text(encoding="utf-8")
        assert JJ_URL in text
        assert B12X_URL in text
        assert TARGET_URL in text
        assert BF16_DRAFT_URL in text
        assert "BF16" in text

    historical = documents[2].read_text(encoding="utf-8")
    assert MXFP8_DRAFT_URL in historical

    deployment = documents[1].read_text(encoding="utf-8")
    assert "The external draft is not Local Inference Lab's separate" in deployment

    public_image = documents[3].read_text(encoding="utf-8")
    assert "FujitsuPolycom/vllm/commit/331573d20bd47e78327ed8d8b4d2e6d350bbb1ab" in public_image
    assert "6255090a03b12c3f7d552102a02fac0b542fb8c9" in public_image
