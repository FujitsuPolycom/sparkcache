"""GPU-free contract tests for release artifact promotion."""

import ast
from pathlib import Path
import tomllib

import pytest

from tools.verify_distribution import (
    LIFECYCLE_SNAPSHOT_LABEL,
    verify_archive_contract,
)


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"
README = ROOT / "README.md"
PACKAGE_INIT = ROOT / "sparkcache" / "__init__.py"
LIFECYCLE_LABEL_EXAMPLE = "proto" + "type"


def test_publish_workflow_verifies_the_build_artifact_before_pypi() -> None:
    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    build = workflow.index("  build:")
    identity_record = workflow.index("release-artifact-sha256.txt")
    distribution_upload = workflow.index("name: python-distributions")
    identity_upload = workflow.index("name: release-artifact-identity")
    publish = workflow.index("  publish:")
    qualification_environment = workflow.index("name: pypi", publish)
    distribution_download = workflow.index("name: python-distributions", publish)
    identity_download = workflow.index("name: release-artifact-identity", publish)
    identity_verification = workflow.index("sha256sum --check", publish)
    pypi_upload = workflow.index("pypa/gh-action-pypi-publish", publish)

    assert (
        build
        < identity_record
        < distribution_upload
        < identity_upload
        < publish
        < qualification_environment
        < distribution_download
        < identity_download
        < identity_verification
        < pypi_upload
    )


def test_clean_ci_declares_numeric_and_lint_dependencies() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    optional = project["optional-dependencies"]
    ci = CI_WORKFLOW.read_text(encoding="utf-8")
    publish = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert "numpy>=1.26" in optional["test"]
    assert optional["lint"] == ["ruff==0.15.17"]
    assert "ruff==0.15.17" in optional["release"]
    assert 'python -m pip install -e ".[lint]"' in ci
    assert 'python -m pip install -e ".[test,release]"' in publish


def test_release_verifier_checks_wheel_and_source_distribution() -> None:
    workflow = PUBLISH_WORKFLOW.read_text(encoding="utf-8")

    assert (
        "python tools/verify_distribution.py dist/*.whl dist/*.tar.gz "
        '--version "$version"'
    ) in workflow


def test_package_metadata_is_publication_neutral_for_its_own_version() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    version = project["version"]
    readme = README.read_text(encoding="utf-8")
    metadata = f"Metadata-Version: 2.4\nVersion: {version}\n\n{readme}"

    assert version == "0.1.0a3"
    verify_archive_contract(
        {"sparkcache/native/python/snapshot_ring_state_model.py"},
        metadata,
        version,
        Path(f"sparkcache-{version}-py3-none-any.whl"),
    )


def test_source_tree_version_fallback_matches_project_metadata() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    module = ast.parse(PACKAGE_INIT.read_text(encoding="utf-8"))
    literal_versions = [
        node.value.value
        for node in ast.walk(module)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__version__"
            for target in node.targets
        )
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    ]

    assert literal_versions == [project["version"]]


def test_snapshot_artifact_paths_use_semantic_names() -> None:
    native_root = ROOT / "sparkcache" / "native"
    lifecycle_labeled = sorted(
        path.relative_to(ROOT).as_posix()
        for path in native_root.rglob("*")
        if path.is_file()
        and "snapshot" in path.as_posix().casefold()
        and LIFECYCLE_SNAPSHOT_LABEL.search(path.as_posix())
    )

    assert lifecycle_labeled == []


@pytest.mark.parametrize(
    ("artifact", "member"),
    [
        (
            Path("sparkcache-0.1.0a3-py3-none-any.whl"),
            "sparkcache/native/python/"
            f"snapshot_ring_state_{LIFECYCLE_LABEL_EXAMPLE}.py",
        ),
        (
            Path("sparkcache-0.1.0a3.tar.gz"),
            "sparkcache-0.1.0a3/sparkcache/native/"
            f"SNAPSHOT_RING_{LIFECYCLE_LABEL_EXAMPLE.upper()}.md",
        ),
    ],
)
def test_release_verifier_rejects_lifecycle_labeled_snapshot_paths(
    artifact: Path,
    member: str,
) -> None:
    metadata = "Metadata-Version: 2.4\nVersion: 0.1.0a3\n"

    with pytest.raises(RuntimeError, match="lifecycle-labeled snapshot paths"):
        verify_archive_contract({member}, metadata, "0.1.0a3", artifact)


@pytest.mark.parametrize(
    "claim",
    [
        "Publication has not been performed.",
        "After version `0.1.0a3` is published, install it from PyPI.",
        "Version `0.1.0a3` is published on PyPI.",
    ],
)
def test_release_verifier_rejects_publication_timing_claims(claim: str) -> None:
    metadata = f"Metadata-Version: 2.4\nVersion: 0.1.0a3\n\n{claim}\n"

    with pytest.raises(RuntimeError, match="publication|published"):
        verify_archive_contract(
            set(),
            metadata,
            "0.1.0a3",
            Path("sparkcache-0.1.0a3.tar.gz"),
        )
