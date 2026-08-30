"""GPU-free contracts for the interactive SparkCache prefix-reuse explorer."""

from html.parser import HTMLParser
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
EXPLAINER = ROOT / "docs" / "sparkcache-prefix-explainer.html"
README = ROOT / "README.md"


class _ExplorerParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []
        self.aria_targets: list[str] = []
        self.links: list[str] = []
        self.scenarios: set[str] = set()
        self.concurrency_values: set[str] = set()
        self.statuses: list[list[str]] = []
        self._open_status: int | None = None
        self._hidden_depth = 0
        self.visible_text: list[str] = []

    def handle_starttag(
        self, tag: str, attrs: list[tuple[str, str | None]]
    ) -> None:
        attributes = dict(attrs)
        if identifier := attributes.get("id"):
            self.ids.append(identifier)
        if labelled_by := attributes.get("aria-labelledby"):
            self.aria_targets.extend(labelled_by.split())
        if href := attributes.get("href"):
            self.links.append(href)
        if scenario := attributes.get("data-scenario"):
            self.scenarios.add(scenario)
        if concurrency := attributes.get("data-concurrency"):
            self.concurrency_values.add(concurrency)

        classes = (attributes.get("class") or "").split()
        if "status-tag" in classes:
            self.statuses.append([])
            self._open_status = len(self.statuses) - 1
        if tag in {"script", "style"}:
            self._hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag == "span" and self._open_status is not None:
            self._open_status = None
        if tag in {"script", "style"}:
            self._hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._open_status is not None:
            self.statuses[self._open_status].append(data)
        if self._hidden_depth == 0 and data.strip():
            self.visible_text.append(data)


def _parse_explorer() -> _ExplorerParser:
    parser = _ExplorerParser()
    parser.feed(EXPLAINER.read_text(encoding="utf-8"))
    parser.close()
    return parser


def test_prefix_explorer_html_contract_is_complete() -> None:
    parser = _parse_explorer()

    assert len(parser.ids) == len(set(parser.ids))
    assert set(parser.aria_targets) <= set(parser.ids)
    assert parser.scenarios == {"exact", "alias", "missing", "corrupt", "cold"}
    assert parser.concurrency_values == {"1", "8", "16"}

    for href in parser.links:
        if "://" in href or href.startswith("#"):
            continue
        assert (EXPLAINER.parent / href).resolve().is_file(), href


def test_prefix_explorer_describes_present_storage_and_restore_behavior() -> None:
    parser = _parse_explorer()
    visible = " ".join(" ".join(parser.visible_text).split())

    required = (
        "256-token logical boundaries",
        "longest stored-prefix restore",
        "descriptor segment contains at most 16 chunk descriptors",
        "4,096-token boundaries",
        "at most 64 aliases",
        "sparkcache-tail-manifest/v1",
        "sparkcache-hybrid-page-delta/v1",
        "at most two deltas",
        "Different row roots may coalesce one authenticated trunk restore",
        "request-private GPU tail",
        "persistent copy-on-write tail objects",
        "When enabled, SparkCache CUDA restore owns verified reconstruction and device transfer",
        "SparkCache CUDA placement component",
        "GLM53_NATIVE_RESTORE_PERFORMANCE_VALIDATION.md",
    )
    for fragment in required:
        assert fragment.casefold() in visible.casefold()

    forbidden = (
        "Fail closed",
        "Fail open",
        "tail-only publication is unsupported",
        "Row tail publication is unsupported",
        "Publishing the grown request writes another complete snapshot",
        "native placement",
        "native restore record",
        "unqualified",
    )
    for fragment in forbidden:
        assert fragment.casefold() not in visible.casefold()


def test_prefix_explorer_uses_canonical_status_vocabulary() -> None:
    parser = _parse_explorer()
    allowed = ("Implemented", "Qualified", "Research-only", "Unsupported")
    statuses = [" ".join(parts).strip() for parts in parser.statuses]

    assert statuses
    for status in statuses:
        assert any(
            status == word or status.startswith(f"{word} ") for word in allowed
        ), status


def test_readme_links_to_prefix_explorer() -> None:
    readme = README.read_text(encoding="utf-8")

    assert "[interactive prefix-reuse explorer](docs/sparkcache-prefix-explainer.html)" in readme
