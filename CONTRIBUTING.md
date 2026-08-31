# Contributing to SparkCache

SparkCache accepts focused bug fixes, compatibility improvements,
documentation corrections, and tests.

The repository is a personal project. Changes should improve cache
correctness, portability, or usability rather than add unrelated operational
machinery.

## Development setup

SparkCache requires Python 3.11 or newer. The continuous-integration matrix
tests Python 3.11, 3.12, and 3.13.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -e ".[test]"
python -m pip install ruff build twine
```

Activate the virtual environment using the command for your shell before
running the checks below.

## Required checks

Run the GPU-free checks for every code change:

```bash
python -m ruff check .
python -m pytest sparkcache -q
python -m pytest deploy -q
python -m build
python -m twine check dist/*
version="$(python -c 'import tomllib; print(tomllib.load(open("pyproject.toml", "rb"))["project"]["version"])')"
python tools/verify_distribution.py "dist/sparkcache-${version}-py3-none-any.whl" --version "$version"
```

Live tests must state the model, checkpoint revision, runtime image or source
revision, topology, command, measured result, and conclusion.

A successful test on one topology says nothing about a different topology
unless that topology is tested separately.

## Correctness requirements

- A cache entry whose identity or integrity cannot be proven must produce a
  cache miss and recomputation. It must never supply unverified state.
- Inference must not wait for asynchronous cache storage work.
- Changes to cache identity, digest salts, or chunk geometry must miss cleanly
  against incompatible on-disk entries. Describe the namespace impact in the
  pull request.
- Behavioral fixes require a GPU-free regression test. If the defect has an
  entry in `DEFECTS.md`, remove the entry in the fixing commit and include its
  stable identifier in the regression-test name.

## Documentation and change descriptions

Follow the **Write without hidden context** rule in `AGENTS.md`. Describe the
system as it exists and use `implemented`, `qualified`, `research-only`, or
`unsupported` when a status label is useful.

Evidence must identify its conditions, measurement, result, and conclusion.

A pull request description and its commits must state:

1. resulting behavior;
2. technical reason;
3. compatibility and cache-namespace impact; and
4. validation performed.

Do not include generated-by or co-author trailers in commits.
