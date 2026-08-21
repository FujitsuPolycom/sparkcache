# Releasing SparkCache

SparkCache releases are built from version tags and published to PyPI by
`.github/workflows/publish.yml`. The workflow uses PyPI trusted publishing;
the repository does not store a PyPI token.

## One-time repository setup

1. Create a PyPI project named `sparkcache`, or reserve that name during the
   first trusted publication.
2. In the PyPI project publishing settings, add a trusted publisher for this
   GitHub repository. Set the workflow filename to `publish.yml` and the
   environment name to `pypi`.
3. Create a GitHub Actions environment named `pypi` with a required reviewer.
   The environment gate must hold the publish job until that reviewer has
   qualified the exact workflow artifact. A single-maintainer repository must
   allow the configured reviewer to approve their own deployment.

The trusted-publisher owner and repository values must match the public GitHub
location exactly. No password, API token, or signing key is required by the
workflow.

## Release procedure

1. Set `project.version` in `pyproject.toml` to the version being released.
   Use a PEP 440 version such as `0.1.0` or `0.1.0rc1`.
2. Replace development-version wording in canonical documentation with the
   resulting supported behavior. Record qualified hardware evidence separately
   from GPU-free package validation.
3. Run the local release checks:

   ```bash
   python -m ruff check .
   python -m pytest sparkcache -q
   python -m pytest deploy -q
   python -m build
   python -m twine check dist/*
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   python tools/verify_distribution.py "dist/sparkcache-${version}-py3-none-any.whl" --version "$version"
   ```

4. Commit the version and documentation changes. The commit message must state
   resulting behavior, technical reason, compatibility impact, and validation.
5. Create and push an annotated tag whose name is `v` followed by the exact
   `project.version` value:

   ```bash
   version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
   git tag -a "v${version}" -m "Release SparkCache ${version}"
   git push origin "v${version}"
   ```

6. Wait for the tag workflow's `build` job to finish. Its `python-distributions`
   artifact contains the wheel and source distribution. Its
   `release-artifact-identity` artifact contains their SHA-256 values. Download
   both artifacts from the same workflow run and verify them before testing:

   ```bash
   gh run download <workflow-run-id> --name python-distributions --dir dist
   gh run download <workflow-run-id> --name release-artifact-identity \
     --dir release-artifact-identity
   (cd dist && sha256sum --check \
     ../release-artifact-identity/release-artifact-sha256.txt)
   ```

7. Install the downloaded wheel on the qualification systems and run the
   store, coordinated-restart, restore, semantic-canary, capacity, and
   corruption gates required by the release's support table. Record the tag,
   workflow run, wheel SHA-256, deployment conditions, measurements, and
   result. A failing or interrupted gate rejects the deployment; it does not
   authorize publication.

8. Approve the waiting `pypi` environment deployment only after qualification
   passes. The publish job downloads the same two workflow artifacts and
   verifies their SHA-256 values immediately before uploading the distributions
   to PyPI.

The publish workflow rejects a tag that does not match `project.version`. Never
approve a different workflow run from the one whose artifacts were qualified.

## Verification

After publication, install from PyPI in a clean virtual environment and verify
the installed package rather than the checkout:

```bash
python -m venv sparkcache-release-check
sparkcache-release-check/bin/python -m pip install --upgrade pip
version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
sparkcache-release-check/bin/python -m pip install "sparkcache==${version}"
sparkcache-release-check/bin/python -c "import sparkcache; print(sparkcache.__version__, sparkcache.__file__)"
```

On Windows, replace `sparkcache-release-check/bin/python` with
`sparkcache-release-check\Scripts\python.exe`.
