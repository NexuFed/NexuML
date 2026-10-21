# Development workflow

Use [Development installation](../development/install.md), then activate `.venv`.
The `dev` extra includes the docs tools.

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

CI checks Python static/GPU tests, packaging and Studio's Linux project/tool workflows
under one required gate. It runs on trusted branch pushes or manual dispatch, not fork
PR events; see the [runner trust boundary](../development/release.md#institute-runner-trust-boundary).
macOS/Windows jobs remain disabled and unverified.

Run all gates locally before opening a PR (see the sections above for individual commands). To trigger or monitor CI manually:

```bash
gh workflow run ci.yml --ref <branch>
gh run list --workflow ci.yml --limit 5
gh run watch <run-id>
gh run view <run-id> --log-failed
```

Local checks do not reproduce the institute runner or certify the disabled platforms.

## Studio and skill checks

```bash
python skills/tests/check_skills.py
python skills/tests/check_runtime.py
npm --prefix studio run typecheck
npm --prefix studio run lint
npm --prefix studio test
```

For installed-package/browser verification, build Python distributions from the same
checkout, then from `studio/`:

```bash
npm pack  # prepack builds and prepares the distribution
STUDIO_TEST_PYTHON=/path/to/isolated/python npm run test:package -- /path/to/built-studio.tgz
```

Use matching backend/API and library wheels plus Playwright Chromium. CI prepares both
environments with `scripts/studio-test-runtime.py` (see `--help`). Installed startup and
browser assets are checked offline; traces stay off to protect credentials. Never
publish during local checks; see [Release](../development/release.md).

## Dependency management

```bash
uv add <package>        # add runtime dep
uv add --optional dev <package>  # add tooling to the project's dev extra
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

When merging, use **Squash and merge** and delete the branch afterwards. Set the commit title to match the PR title exactly, and replace the default commit description with a bullet-point list of what changed:

```
feat(NEX-123): add export pipeline

- Add ONNX export step to pipeline registry
- Expose --output-format CLI flag
- Update docs with export examples
```

## Docs

```bash
DISABLE_MKDOCS_2_WARNING=true mkdocs serve
DISABLE_MKDOCS_2_WARNING=true mkdocs build --strict
```
