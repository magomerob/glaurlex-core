# Contributing

## Setup

Python 3.10 or later and [uv](https://docs.astral.sh/uv/):

```bash
uv venv && uv pip install -e . --group dev
```

## Checks

CI runs these on Python 3.10 (lowest supported dependency versions) and 3.11-3.13 (latest):

```bash
.venv/bin/python -m ruff check . && .venv/bin/python -m ruff format --check .
.venv/bin/python -m mypy
.venv/bin/python -m pytest --cov=glaurlex --cov-fail-under=85
```

To test against the lowest supported dependencies:

```bash
uv venv -p 3.10 /tmp/glaurlex-lowest
VIRTUAL_ENV=/tmp/glaurlex-lowest uv pip install --resolution lowest-direct .
VIRTUAL_ENV=/tmp/glaurlex-lowest uv pip install pytest
/tmp/glaurlex-lowest/bin/python -m pytest
```

## Rules

- The library never imports a UI, reads environment variables or writes files; the only file it
  reads is the input of `io.read_salamanca`. `tests/test_purity.py` enforces this.
- Docstrings follow the NumPy convention.
- `tests/data/reference.json` holds outputs of the original GLAURLex application on
  `tests/data/{topic,informants}.csv`. It is frozen: never edit a value to make a test pass. A
  deliberate behaviour change updates the value and states why in the commit message.
- The availability values in `tests/test_stats.py` come from LexPro, errata included. Do not
  correct them.
- Bug fixes start with a test that reproduces the bug.

## Release

1. Set `version` in `pyproject.toml` and move the `CHANGELOG.md` entry from "unreleased" to the
   release date.
2. Merge to `master` and wait for CI to pass.
3. Tag and push: `git tag v0.1.0 && git push origin v0.1.0`.

The release workflow checks that the tag matches `project.version`, runs the checks, builds the
wheel and sdist, and attaches them to a GitHub Release.
