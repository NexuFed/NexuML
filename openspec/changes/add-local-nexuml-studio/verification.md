# Verification — 2026-10-02

**Status: Linux acceptance passed; cross-platform release acceptance pending.**
Initial acceptance recorded 37/39 verified tasks; section 9 adds the approved UX follow-up.
Tasks 7.2 and 8.2 remain open until the
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

## Graph-first UX follow-up — 2026-10-03

The same configuration/editor and execution seam now provide recursive schema forms,
field-focused diagnostics, categorized drag/drop, sidecar-only names, removable/reconnectable
routes and native progress/terminal overwrite rendering. No new runtime provisioning,
configuration format, ML execution backend, account system or frontend dependency was added.
Section 9 is verified complete; the current checklist is **44/46**, with only the original
cross-platform release gates 7.2 and 8.2 open.

| Check | Result |
| --- | --- |
| `npm test` | 15 passed: schema refs/defaults/unions, routing/order, typed collection connections, names/undo, terminal overwrite/fragmentation/bounds |
| `npm run typecheck`, `npm run lint` | Passed |
| Ruff over changed Python definitions/API/tests | Passed |
| `PYTHONPATH=src:library/src NEXUML_API_WHEEL=… python -m pytest tests/api tests/core tests/training tests/execution tests/cli -q -ra` | 200 passed, 1 existing missing-CIFAR-data skip; built-wheel API/core-only startup included |
| Built Python wheels compared with current source | Exact match: 82 core and 81 library Python files |
| `npm run test:package` with final installed project runtime | Passed: production build/pack, offline installation/assets, 3 browser workflows, explicit/PATH/attached launch and ownership cleanup, unchanged installed package metadata, saved YAML readable by CLI |
| `npm run test:package` with final installed uv-tool runtime | Passed: the same 3 browser workflows and packaged lifecycle/installation invariants |
| `openspec validate add-local-nexuml-studio --strict`, `git diff --check` | Passed |

Project smoke artifacts: `/tmp/opencode/nexuml packaged smoke pg4l76/`.
Uv-tool smoke artifacts: `/tmp/opencode/nexuml packaged smoke 03zDqp/`.
Final Python wheels: `/tmp/opencode/studio-ux-final-dist/`; genuine release-test project
and uv-tool installations: `/tmp/opencode/studio-ux-final-project/` and
`/tmp/opencode/studio-ux-final-tool/` (test setup only, never normal launcher provisioning).

Browser acceptance covers:

- Ray/local union settings without launching Ray; custom installed schema controls and
  invalid Expert buffers retained without changing the draft.
- Basic-hidden data fields plus nested data/layer/evaluation diagnostics navigate and focus
  actual typed inputs; real build failure messages remain associated with frozen launch settings.
  Opening those settings as a draft is explicit and does not mutate the failed operation.
- Category search, native HTML drag/drop onto a stage header, input connection/removal via
  accessible selects, keyboard and pointer edge selection/deletion, undo, sidecar naming and save/reopen.
- Native CPU callbacks, metrics, reconnect and checkpoint-derived export through the installed
  package. No simulated counters or terminal-to-training telemetry parser is used.
- Desktop 1440×1000, tablet 1024×900 and mobile 390×844 screenshots; approved background/action
  colours and no document-level horizontal overflow. Latest screenshots are under
  `studio/test-results/authoring-typed-settings-e-9cf4e-s-and-removable-key-routing/` and the
  existing workflow screenshot directory.

Counter unit proof covers throttling, phase transitions, multiple validation loaders,
final batches and nonfinite totals. Terminal tests cover fragmented ANSI, CRLF, CR/backspace,
multiline updates and rendering limits; API observation preserves the exact control bytes and
raw logs. This does not claim a new live MNIST/CIFAR download was exercised. Downloads without
native totals remain honestly indeterminate, with their actual progress text in the terminal.

Inherited evaluator defaults cannot be disconnected by hiding an edge: removal explains
that the user must reroute or disable that evaluator. Empty required layer inputs remain
visible and block UI launch; dynamic dataset/grouping/undeclared keys remain runtime-only.

Intermediate failures were resolved before acceptance: incorrect SVG/overlapping drag targets
in browser checks, an incorrect nullable-field diagnostic path expectation, an incomplete
typed test catalog fixture and a typecheck/build race over generated `.next/types`.

Cross-platform packaged acceptance is still unrun; tasks 7.2 and 8.2 remain unchecked.
OpenChamber browser forwarding still produced `chrome-error://chromewebdata/`; local Playwright
and server readiness establish local browser functionality, not remote panel access.
No commit, push, CI dispatch, package publication or OpenSpec archival was performed for this follow-up.

The persistent preview on UI/API ports 41240/41241 was restarted only after checking that
no observed work was running and verifying its recorded PID/create-time identity. It uses
the final installed uv-tool Python and npm package. Playwright opened its actual `tiny.yaml`,
verified Vision categories/typed training settings with no page errors, and captured
`studio/test-results/ux-preview-desktop.png` and `ux-preview-training.png`. The auxiliary
41340/41341 test preview was stopped after the same idle/identity checks; logs were retained.

## Live scalar chart correction — 2026-10-03

The previous callback sampled only at epoch end, and validation/test callback hooks ran
before the module finalized classification/evaluator metrics. A one-point SVG polyline
was also invisible. Regression tests reproduced both missing batch losses and missing
finalized accuracy/F1 before the correction.

Training now publishes actual finite `train/*` scalars with throttled batch progress,
including first/final batches. Validation/test snapshots run after module finalization;
sanity-check and stale cross-phase values are not reused as new measurements. Charts
use actual optimizer steps, draw a visible single/latest point, and retain metric history
separately from verbose logs. NexuML training, loss weighting and metric computation are
unchanged; CIFAR accuracy/F1 remain validation/test-only pipeline metrics.

- `PYTHONPATH=src:library/src /env/bin/python -m pytest tests/api/test_ux_observation.py tests/api/test_operations.py -q --tb=short`: **14 passed**, including real CPU classification and transport/replay/checkpoint export.
- `npm test`: **16 passed**; frontend typecheck and lint passed.
- Ruff over changed Python files, strict OpenSpec validation and `git diff --check`: passed.
- `npm run test:package`: passed with all **4 browser workflows**, including first-epoch
  loss updates while running, finalized accuracy/F1, reconnect, and mobile layout. Offline
  assets/install, saved YAML/CLI build/export, unchanged installation records and
  owned/attached API shutdown passed. Optional executable-selection checks were not enabled.
  Artifacts: `/tmp/opencode/nexuml packaged smoke jW2sff/`; current-source API overlay on the
  existing isolated uv-tool test runtime, without installing/upgrading Python.
- Initial smoke failures were corrected: unrelated legacy `/env` discovery plugins/slow
  requests, a synthetic recipe's dataset-list source, and an oversized CPU test timing out
  and leaving the test API busy. The final test uses small real synthetic tensors/layers
  with test-only single-thread CPU settings, not simulated values or artificial delays.

The original preview's CIFAR operation `05796c20d5fe48a8b28177b8c3a8cc5f` was confirmed
running. Its launcher identity matched the recorded PID/create time; it was not stopped
or modified. Batch samples missing from that old operation cannot be reconstructed.
The separate current-source preview is on UI/API ports 41340/41341, recorded in
`/tmp/opencode/nexuml-studio-live-metrics-preview.json`. Its existing runtime and prebuilt
frontend start without provisioning; desktop/mobile readiness checks passed. Browser panel
forwarding still fails, independently of local HTTP/Playwright readiness.
Cross-platform release gates 7.2/8.2 remain open; this correction is not committed,
pushed, published or archived.

## Stage-container follow-up — 2026-10-03

Tasks 10.1–10.7 are complete; the checklist is **51/53**, with cross-platform release
gates 7.2/8.2 still open. The changes extend the existing graph/draft rather than adding an execution model:
stage headers carry their children, resize/position controls edit layout only, and
Structure/order-strip insertion markers edit native stage/layer lists explicitly.
Transfers preserve stable identity, selection, configuration and routing. Skipped-stage
previews warn without changing skip settings; existing orphan/cancelled drops restore
their source, while new unowned layers require an explicit destination. Positions,
display names and sizes remain revision-bound sidecar data, not executable YAML.

Focused proof:

- `npm test`: **21 passed**; typecheck/lint passed. Model regressions cover measured
  containment, integer-like stage names, identical-layer transfers, declared dependency
  reprojection, one-entry transactions, older sidecars and unrelated settings.
- `PYTHONPATH=src:library/src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /env/bin/python -m pytest tests/api/test_config.py tests/api/test_operations.py tests/api/test_ux_observation.py -q --tb=short`:
  **17 passed, 8 warnings**. Includes size-sidecar conflicts and native pipeline/order
  edits leaving an already accepted CPU run frozen. Explicit source paths avoid an
  unrelated installed checkout; no Python packages were installed or upgraded.
- Two source UI/API browser workflows passed: header movement carries children,
  free placement preserves order, resize respects containment, semantic transfers/order
  stale build evidence while layout edits retain it, and undo/redo preserves identity.
  Tests cover valid/orphan/Escape/skipped transfers, sortable tree/strip ordering,
  unique stage creation, explicit before/after/append insertion, empty-stage focus,
  keyboard alternatives, sidecar save/reopen and retained invalid Expert text.
- An early packaged failure exposed the canvas pane overlapping diagnostics; containing
  it within the editor fixed that regression. Pointer checks await Fit view before
  measuring and refit after expanding a container. The next packaged attempt passed
  **5/6** workflows but compared an entire library-generated style string: the restored
  transform matched exactly while React Flow's selected-child z-index differed.
  Placement checks now assert transforms and retained redo history instead of stacking.
  Failed artifacts: `/tmp/opencode/nexuml packaged smoke lHo09a/`.

Final Linux proof:

- The packed/offline-installed frontend in `/tmp/opencode/nexuml packaged smoke SPUNdb/`
  passed **6/6 browser workflows**, including real CPU training/reconnect/export, live
  scalar charts, typed authoring and both container workflows. Production build,
  typecheck, lint, **21 frontend tests** and focused Ruff passed.
- The package runner's tool call was interrupted, so its final exit status is unknown,
  not reported as a full command pass. Retained attachment-setup artifacts prove it
  reached the checks after unchanged npm lockfile/Python installation records,
  saved-YAML CLI build and explicit owned-API shutdown assertions had passed.
- The missing final attached-API ownership check was verified separately using that
  same installed package and existing Python runtime with a current-source overlay:
  launcher shutdown leaves the authenticated attached API alive; the separately owned
  proof API then stops on explicit cleanup. No rebuild or suite rerun was needed.
  Evidence: `/tmp/opencode/studio-attached-proof-WInGpW/result.json`.
- Inspected packaged desktop 1440px, tablet 1024px and mobile 390px screenshots under
  `studio/test-results/containers-stage-container-84542-e-undo-and-sidecar-recovery/`.
  Corporate colors/fonts, contained canvas, scrollable order strip and responsive panels
  remain intact. Browser checks cover keyboard/single-pointer alternatives, focus recovery,
  reduced-motion handling and absence of document-level horizontal overflow; this is not
  a comprehensive accessibility certification.

The idle owned debug services on 41440/41441 were stopped after checking the operation
list and launcher identity. Existing previews on 41240/41241 and 41340/41341 were neither
stopped nor replaced. No Python packages were installed/upgraded and no commit, push,
publication, CI dispatch or archival was performed. Cross-platform gates 7.2/8.2 remain open.
