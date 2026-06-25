# Development setup

## Install dev dependencies

```bash
uv sync --all-extras
source .venv/bin/activate
uv pip install --link-mode=copy -e ".[all]"
uv pip install --link-mode=copy -e "./library[all]"
```

## Tests

The test suite is registry-driven:

```bash
pytest                         # default run, no coverage
pytest tests/core -v           # framework-feature coverage only
pytest tests/_registry -v      # registry-driven conformance only
pytest -m "not slow"           # skip slow training smoke tests
```

To see coverage locally, run the CI-style coverage command:

```bash
pytest --cov=nexuml --cov=nexuml_library \
  --cov-report=term-missing \
  --cov-report=xml:reports/coverage.xml
```

The terminal report shows total coverage and missing line ranges per file. The
XML report is written to `reports/coverage.xml` for tools that consume Cobertura
coverage. The failure threshold is configured in `pyproject.toml` under
`[tool.coverage.report]`.

For a browsable report:

```bash
pytest --cov=nexuml --cov=nexuml_library --cov-report=html:reports/htmlcov
python -m http.server 8000 --directory reports/htmlcov
```

Then open <http://localhost:8000>.

Gated tests are skipped automatically when their prerequisites are missing:

- `requires_data` — set `NEXUML_RUN_DATA_TESTS=1` and a valid `NEXUML_DATA_ROOT`.
- `requires_gpu` — skipped when no CUDA device is available.
- `requires_optional(name)` — skipped when the optional dependency is missing.

## Type checking

```bash
ty check
```

## Linting and formatting

```bash
ruff check .
ruff format --check .
```

## OpenSpec validation

```bash
openspec validate <change> --strict
```

## CI gates

The `main` branch runs `.github/workflows/ci.yml` on every push and PR:

1. Static job (`ika-runner`): `ruff check`, `ruff format --check`, `ty check`.
2. Test job (`ika-runner-gpu`): full `pytest --cov=nexuml --cov=nexuml_library --cov-report=term-missing` suite with the configured coverage gate.

Run the same gates locally before opening or updating a PR:

```bash
ruff check .
ruff format --check .
ty check
pytest --cov=nexuml --cov=nexuml_library \
  --cov-report=term-missing \
  --cov-report=xml:reports/coverage.xml
```

To trigger the GitHub Actions workflow manually from GitHub:

1. Open the repository on GitHub.
2. Go to **Actions** → **CI**.
3. Click **Run workflow**.
4. Select the branch and click **Run workflow**.

The same manual trigger is available from the GitHub CLI:

```bash
gh workflow run ci.yml --ref <branch>
gh run list --workflow ci.yml --limit 5
gh run watch <run-id>
gh run view <run-id> --log-failed
```

Local workflow emulation with `act` is useful for YAML syntax and basic job
shape, but it does not reproduce the self-hosted `ika-runner` /
`ika-runner-gpu` environment, CUDA devices, NexuCluster scheduling, or private
runner credentials. Treat `act` as a smoke check only; the authoritative result
is the GitHub run on the real runners.

## Dependency management

```bash
uv add <package>        # add runtime dep
uv add --dev <package>  # add dev dep
uv lock --upgrade       # update the lock file to latest packages
uv sync                 # install all deps from uv.lock
uv lock                 # regenerate uv.lock
```

## Pull requests

Label every PR before merging — labels drive the automated release changelog generated on each `v*` tag.

| Label | Release section |
|---|---|
| `feature`, `enhancement` | 🚀 Features |
| `bug`, `fix` | 🐛 Bug Fixes |
| `chore`, `refactor` | 🧰 Maintenance |
| `documentation`, `docs` | 📝 Documentation |
| `ignore-for-release` | excluded entirely |
| *(anything else)* | Other Changes |

PRs without a matching label land in **Other Changes**. Add `ignore-for-release` to omit a PR from the changelog completely (e.g. CI tweaks, lock-file-only bumps).

## Docs

```bash
uv sync --extra docs
DISABLE_MKDOCS_2_WARNING=true mkdocs serve
```
