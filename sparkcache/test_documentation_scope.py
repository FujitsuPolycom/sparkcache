from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

GENERIC_DOCUMENTS = (
    ROOT / "sparkcache/README.md",
    ROOT / "ROADMAP.md",
    ROOT / "CONTEXT.md",
    ROOT / "sparkcache/native/README.md",
    ROOT / "sparkcache/runtime_patches/README.md",
    ROOT / "sparkcache/streaming/FEATURE_GATE.md",
    ROOT / "sparkcache/streaming/RUNTIME.md",
    ROOT / "sparkcache/replication/README.md",
    ROOT / "docs/sparkcache-prefix-explainer.html",
    ROOT / "deploy/deployment_contract/README.md",
)


def test_generic_documents_do_not_embed_model_profile_details() -> None:
    forbidden = (
        "GLM-",
        "DeepSeek",
        "Qwen",
        "DFlash",
        "Jovian Judgement",
        "PR535",
        "TP4",
        "DCP4",
    )

    for path in GENERIC_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for fragment in forbidden:
            assert fragment.casefold() not in text.casefold(), (path, fragment)


def test_root_readme_routes_model_details_to_deployment_profiles() -> None:
    readme = (ROOT / "README.md").read_text(encoding="utf-8")

    assert "## Deployment profiles" in readme
    assert "deploy/glm53_flash/README.md" in readme
    assert "deploy/glm52_35bpw/README.md" in readme
    assert "deploy/deepseek_v4/README.md" in readme
    assert "Model-specific settings, launch commands" in readme
