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
3. Create a GitHub Actions environment named `pypi`. An optional required
   reviewer provides a human confirmation before publication.

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

The publish workflow rejects a tag that does not match `project.version`. It
builds a source distribution and wheel, runs `twine check`, uploads the files as
a workflow artifact, and publishes that exact artifact through the `pypi`
environment.

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
