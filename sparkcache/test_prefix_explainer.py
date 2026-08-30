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

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
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
        "sparkcache-page-delta-manifest/v2",
        "sparkcache-page-snapshot-manifest/v2",
        "objects of at most 64 MiB",
        "813,068,464 bytes",
        "13 authenticated delta objects",
        "at most two deltas",
        "Different row roots may coalesce one authenticated trunk restore",
        "16 distinct request tails sharing one restored 128K block_pages_v1 prefix",
        "no live model artifact qualifies it",
        "request-private GPU tail",
        "persistent copy-on-write tail objects",
        "When enabled, SparkCache CUDA restore owns verified reconstruction and device transfer",
        "SparkCache CUDA placement component",
        "GLM53_SPARKCACHE_CUDA_RESTORE_PERFORMANCE_VALIDATION.md",
        "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8",
    )
    for fragment in required:
        assert fragment.casefold() in visible.casefold()


def test_prefix_explorer_uses_canonical_status_vocabulary() -> None:
    parser = _parse_explorer()
    allowed = ("Implemented", "Qualified", "Research-only", "Unsupported")
    statuses = [" ".join(parts).strip() for parts in parser.statuses]

    assert statuses
    for status in statuses:
        assert any(
            status == word or status.startswith(f"{word} ") for word in allowed
        ), status


def test_readme_links_explorer_and_states_capabilities() -> None:
    readme = README.read_text(encoding="utf-8")

    required = (
        "[interactive prefix-reuse explorer](docs/sparkcache-prefix-explainer.html)",
        "Tail-only opaque-page deltas | **qualified**",
        "64 MiB flat page macro objects | **implemented**",
        "Different-root shared row segments | **implemented**",
        "16 distinct request tails shared one restored 128K `block_pages_v1` prefix",
        "sha256:ed60be066d6d9eadea267bc4597a0687869f3ddb95a3e5c6f86649893a838eb8",
        "Flat `sparkcache-page-snapshot-manifest/v2` objects at SparkCache `90946fd6`",
    )
    for fragment in required:
        assert fragment in readme
