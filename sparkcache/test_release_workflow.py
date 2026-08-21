"""GPU-free contract tests for release artifact promotion."""

from pathlib import Path
import tomllib


ROOT = Path(__file__).resolve().parents[1]
PUBLISH_WORKFLOW = ROOT / ".github" / "workflows" / "publish.yml"
CI_WORKFLOW = ROOT / ".github" / "workflows" / "ci.yml"
PYPROJECT = ROOT / "pyproject.toml"


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
