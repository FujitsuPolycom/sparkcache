from __future__ import annotations

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]

GENERIC_DOCUMENTS = (
    ROOT / "CONTRIBUTING.md",
    ROOT / "SECURITY.md",
    ROOT / "RELEASING.md",
    ROOT / "sparkcache/README.md",
    ROOT / "ROADMAP.md",
    ROOT / "CONTEXT.md",
    ROOT / "sparkcache/native/README.md",
    ROOT / "sparkcache/native/SNAPSHOT_RING_STATE_MODEL.md",
    ROOT / "sparkcache/runtime_patches/README.md",
    ROOT / "sparkcache/streaming/OPT_IN.md",
    ROOT / "sparkcache/streaming/RUNTIME.md",
    ROOT / "sparkcache/replication/README.md",
    ROOT / "docs/sparkcache-prefix-explainer.html",
    ROOT / "docs/agents/domain.md",
    ROOT / "docs/agents/issue-tracker.md",
    ROOT / "docs/agents/triage-labels.md",
    ROOT / "deploy/deployment_contract/README.md",
)

GENERIC_MARKDOWN_DOCUMENTS = tuple(
    path for path in GENERIC_DOCUMENTS if path.suffix == ".md"
)

PLAIN_LANGUAGE_DOCUMENTS = (ROOT / "README.md",) + GENERIC_MARKDOWN_DOCUMENTS

HIDDEN_CONTEXT_PATTERNS = (
    re.compile(r"\bphase\s+[0-9ivx]+\b", re.IGNORECASE),
    re.compile(r"\bpilot\b", re.IGNORECASE),
    re.compile(r"\bthe (?:current|new|old|latest)\b", re.IGNORECASE),
    re.compile(r"\bthe experiment\b", re.IGNORECASE),
    re.compile(r"\bthis approach\b", re.IGNORECASE),
    re.compile(r"\bPR\s*#?\d+\b", re.IGNORECASE),
)


def _prose_paragraphs(text: str) -> list[str]:
    paragraphs: list[str] = []
    pending: list[str] = []
    in_fence = False

    def flush() -> None:
        if pending:
            paragraphs.append(" ".join(pending))
            pending.clear()

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if line.startswith("```"):
            flush()
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        if not line:
            flush()
            continue
        if line.startswith(("#", "|", "- ", "* ")) or re.match(
            r"\d+\.\s", line
        ):
            flush()
            continue
        pending.append(line)
    flush()
    return paragraphs


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
    assert "Profiles keep model-specific settings" in readme
    assert "| Capability | In plain language | Status |" in readme
    assert len(readme.split()) < 900


def test_generic_documents_avoid_hidden_context_shorthand() -> None:
    for path in PLAIN_LANGUAGE_DOCUMENTS:
        text = path.read_text(encoding="utf-8")
        for pattern in HIDDEN_CONTEXT_PATTERNS:
            assert pattern.search(text) is None, (path, pattern.pattern)


def test_generic_prose_paragraphs_are_short() -> None:
    for path in PLAIN_LANGUAGE_DOCUMENTS:
        for paragraph in _prose_paragraphs(path.read_text(encoding="utf-8")):
            assert len(paragraph) <= 240, (path, len(paragraph), paragraph)


def test_agent_docs_name_the_public_issue_tracker() -> None:
    issue_tracker = (ROOT / "docs/agents/issue-tracker.md").read_text(
        encoding="utf-8"
    )
    assert "https://github.com/FujitsuPolycom/sparkcache/issues" in issue_tracker
    assert "has no configured Git remote" not in issue_tracker
