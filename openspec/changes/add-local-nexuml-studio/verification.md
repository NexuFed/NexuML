# Verification — 2026-10-02

**Status: Linux acceptance passed; cross-platform release acceptance pending.**
`tasks.md` records 37/39 verified tasks. Tasks 7.2 and 8.2 remain open until the
packaged macOS/Windows jobs pass and the final portability review is complete.

## Executed checks

| Check | Result |
| --- | --- |
| Focused Ruff check over changed Python modules/tests | Passed |
| Python API/core/CLI/training/execution/distribution regressions | 198 passed, 1 skipped |
| `npm run typecheck`, `npm run lint`, `npm test` | Passed; 8 frontend tests |
| Production build, `npm pack`, clean offline npm installation | Passed within `npm run test:package` |
| Packaged workflow against genuinely installed project runtime | Passed |
| Packaged workflow against genuinely installed uv-tool runtime | Passed |
| `openspec validate add-local-nexuml-studio --strict` | Passed |
| `git diff --check` | Passed |
| Whitespace checks for 59 new files | Passed |
| Built core wheel compared with all 82 current Python source files | Exact match |

The Python skip is the existing CIFAR-dependent export test: its dataset is not
available locally. No dataset was downloaded to hide that skip.

## Evidence by capability

- **Runtime/security/configuration:** `tests/api/test_transport.py`,
  `tests/api/test_config.py`, `tests/api/test_packaged_api.py`, and
  `studio/tests/bootstrap.test.ts` cover identity, lazy imports/missing extras,
  protected REST/first-frame WebSockets, origin/interface rejection, discovery,
  field addresses, ordinary YAML, ordered numeric-like stage names, authorized
  paths, atomic saves and external revision conflicts.
- **Native operations:** `tests/api/test_operations.py` covers real CPU
  build/train, single-active rejection, frozen configuration, observed metrics
  compared with terminal results, replay/gaps/frame limits, logs-root selection,
  checkpoint resume, timeout/failure, cancellation/descendant escalation, restart
  ownership uncertainty and Ray dispatch/resume boundaries. Package reload compares
  every exported state tensor with the actual trained checkpoint. Existing
  `tests/training`, `tests/execution` and `tests/core/test_export.py` bracket reuse
  of NexuML's lifecycle, Ray restrictions and export implementations.
- **Editing/rendering:** `studio/tests/model.test.ts` covers alias/label/metadata
  routing, preceding overwritten producers, evaluation placement, ordered edits,
  unrendered/factory fields, undo/redo, per-editor state and layout-only edits.
  The real browser workflow covers installed recipes/custom library roots,
  insertion/removal, keyboard reordering, non-drag routing, inspector errors,
  retained invalid/unknown YAML, save/reopen, stale layouts, visible nodes/edges
  after edits, source-bound build shapes, frozen training, reconnect and export.
- **Distribution:** `studio/scripts/package-smoke.mjs` installs the tarball into
  a clean directory, blocks external browser assets, checks bundled fonts and
  package exclusions, uses paths with spaces, exercises interpreter/executable/
  PATH/attach modes, and checks owned versus attached shutdown. It checks npm
  lockfile and installed Python RECORD/venv metadata for unexpected changes, then
  builds the browser-saved YAML through the installed CLI. Python artifact tests
  check extras, API inclusion and absence of frontend assets/dependency trees.
- **Appearance/accessibility:** desktop 1440px, tablet 1024px and mobile 390px
  screenshots were inspected. Browser assertions check the exact background/action
  colors, responsive panels and no document-level horizontal overflow. Keyboard
  controls, focus styling, drag alternatives and reduced-motion CSS are present;
  this is not a claim of a comprehensive accessibility certification.

## Reproduce

From the repository root:

```sh
NEXUML_API_WHEEL=/tmp/opencode/studio-verified-dist/nexuml-0.2.1-py3-none-any.whl \
NEXUML_DIST_ROOT=/tmp/opencode/studio-verified-audit \
PYTHONPATH=src:library/src /env/bin/python -m pytest \
  tests/api tests/core tests/cli tests/training tests/execution \
  tests/packaging/test_distribution_artifacts.py -q -ra
```

From `studio/`, select either installed test runtime with `STUDIO_TEST_PYTHON`
and `STUDIO_TEST_NEXUML`, then run `TMPDIR=/tmp/opencode STUDIO_KEEP_SMOKE=1 npm run test:package`.
Installation setup is explicit test preparation via `scripts/studio-test-runtime.py`,
never normal launcher provisioning.

## Retained artifacts and limits

- Python wheels/sdists: `/tmp/opencode/studio-verified-dist/`.
- Project-runtime npm tarball/results: `/tmp/opencode/nexuml packaged smoke fIvuEV/`.
- Latest uv-tool npm tarball/results: `/tmp/opencode/nexuml packaged smoke RIPXLE/`.
- Latest screenshots: `studio/test-results/` (generated, ignored by Git).
- `.github/workflows/studio.yml` defines Linux/macOS/Windows × project/tool jobs;
  macOS/Windows have not run here. Real remote Ray execution was not started.
- No hosted features, browser training, parallel ML engine, automatic package
  provisioning, publication or OpenSpec archival were performed.
- Owned diagnostic launchers were stopped; explicitly attached APIs are never
  stopped by Studio. Acceptance artifacts/logs remain available for inspection.
