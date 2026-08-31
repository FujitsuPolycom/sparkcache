# Releasing SparkCache

Version tags start `.github/workflows/publish.yml`. The workflow builds the
Python distributions and publishes them through PyPI trusted publishing. The
repository stores no PyPI token.

## Repository setup

1. Create or reserve the PyPI project named `sparkcache`.
2. Add this repository as a trusted publisher. Use workflow `publish.yml` and
   environment `pypi`.
3. Create the GitHub Actions environment `pypi` with a required reviewer.

The owner and repository names must match the public GitHub location. A
single-maintainer repository must allow that maintainer to approve the waiting
deployment.

## Publish a version

1. Set `project.version` in `pyproject.toml` to a PEP 440 version such as
   `0.1.0` or `0.1.0rc1`.
2. Update canonical documentation to describe the resulting package behavior.
   Keep live hardware results in their profile or evidence records.
3. Run the local checks:

   ```bash
   python -m ruff check .
   python -m pytest sparkcache -q
   python -m pytest deploy -q
   python -m build
   python -m twine check dist/*
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   python tools/verify_distribution.py dist/*.whl dist/*.tar.gz --version "$version"
   ```

4. Commit the version and documentation changes. Describe the resulting
   behavior, reason, compatibility impact, and validation.
5. Create and push an annotated tag:

   ```bash
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   git tag -a "v${version}" -m "Release SparkCache ${version}"
   git push origin "v${version}"
   ```

6. Download both artifacts from the tag workflow's `build` job and verify that
   their checksums match:

   ```bash
   gh run download <workflow-run-id> --name python-distributions --dir dist
   gh run download <workflow-run-id> --name release-artifact-identity \
     --dir release-artifact-identity
   (cd dist && sha256sum --check \
     ../release-artifact-identity/release-artifact-sha256.txt)
   ```

7. Match the test depth to the changed behavior:

   - Documentation, metadata, packaging, and test-tool changes need the
     GPU-free suite, archive verification, and isolated installation.
   - Connector control-plane changes need the GPU-free TP/DCP matrix and one
     representative four-rank live check.
   - Model-profile changes need live checks for the affected profiles.
   - Storage formats, identity, CUDA ownership, placement, or vLLM patches need
     live checks for every affected profile.

A prerelease may use **implemented** status after offline and package checks.
Its README and release notes must say which live checks were not run.

8. Approve the waiting `pypi` deployment after reviewing the artifact hashes,
   resulting status, completed checks, and omitted checks.

The publish job verifies the same artifacts immediately before uploading them.
It rejects a tag that does not match `project.version`.

Live evidence belongs to the exact published artifact tested. Do not copy a
result from a different artifact, even when the source trees appear equivalent.

## Verify the published package

Install from PyPI in a clean virtual environment. The isolated Python probe
excludes the checkout and user site, so local files cannot satisfy the import.

```bash
python -m venv sparkcache-release-check
sparkcache-release-check/bin/python -m pip install --upgrade pip
version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
sparkcache-release-check/bin/python -m pip install "sparkcache==${version}"
sparkcache-release-check/bin/python -I -c "import sparkcache; print(sparkcache.__version__, sparkcache.__file__)"
```

On Windows, use `sparkcache-release-check\Scripts\python.exe`.
