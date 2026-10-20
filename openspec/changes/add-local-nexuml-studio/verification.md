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

## PR #16 CI repair (2026-10-20)

Run `39339644310` failed in all six platform/install combinations at the same npm
installation step: `ENOTCACHED` for `@base-ui/react` registry metadata. Every job had
already passed frontend static checks. A warm developer cache had hidden the harness
assumption: `npm ci` fetches a locked dependency graph but does not guarantee metadata
for resolving a new consumer installation with `npm install --offline`.

Studio checks now live in `.github/workflows/ci.yml`, not a separate Studio workflow.
The existing CI runs on `main` and `dev`, checks frontend static gates once, reuses the
validated Python distribution artifact across all six Studio package jobs, and includes
both Studio jobs in its always-running required `CI` gate and result summary. Package
installation may download dependencies; installed operation still requires no external
assets, package installation, or Python environment mutation. CI keeps screenshot/log
artifacts and HTML browser reports for seven days, with credential-bearing traces off.

Focused checks passed: Actionlint 1.7.12 (excluding existing custom runner labels from
its built-in allowlist); frontend typecheck, lint, and 32 tests; both wheel/sdist builds;
and three packaging/CI regression tests. The required-gate test executes its actual
shell body with each dependency failed, skipped, and cancelled to prove it fails closed.

The first local cold-cache package run passed build/pack/install, then found a second
harness defect: the preliminary browser check used a Config path field before opening
the current file dialog. The smoke check now uses welcome Open YAML, then dialog Open
file and waits for the visible pipeline, matching the committed browser helpers. The
first full run passed 10/11 browser tests, exposing another stale expectation in the
custom-library path: optional Scale is under Advanced, which the test had not opened.
The test now opens Advanced and asserts visibility before editing; no forced interaction
or application change was made. The earlier four backend-test failures remain separate
from this CI harness repair; no unrelated backend behavior was changed.

Final local proof: `npm run test:package` passed with an empty npm cache and an isolated,
wheel-installed Python 3.12 uv-tool runtime. All **11 browser tests passed in 3.3 minutes**,
including the custom-library path and real CPU training/reconnect/export. Production
build/pack, offline assets, explicit/PATH runtime selection, unchanged Python/npm package
records, saved-YAML CLI loading, and owned/attached shutdown checks also passed.
Retained package artifacts: `/tmp/opencode/nexuml packaged smoke wI0Is3/`; the final HTML
report and screenshots are generated under ignored `studio/playwright-report/` and
`studio/test-results/`. Project and tool wheel-install/handshake setup both passed; the
project's full suite was not repeated after correcting its custom-library test. Full
OS/install coverage remains the replacement CI matrix's responsibility, not a local
macOS/Windows acceptance claim.

### Hosted CI follow-up

Run `39342129833` passed all three Python/CUDA suites (3.12/3.13/3.14), Python package
validation, Studio static checks, and both Linux package modes. This supersedes the
earlier local-environment backend failures as hosted Python test evidence. The remaining
failures were separate portability/checking defects:

- Colored Rich help splits option strings with ANSI styling. The skills runtime check
  now removes SGR styling before checking the same required/forbidden CLI options.
  `FORCE_COLOR=1 CI=true python skills/tests/check_runtime.py` reproduced the assertion
  before the fix and passed afterward; ordinary help also passed.
- macOS temporary paths traverse aliases. The CLI guard compared raw `process.argv[1]`
  with the canonical module URL and exited successfully without starting. A new
  directory-alias entrypoint test reproduced the empty stdout locally; resolving argv
  with the launcher's existing `realpath` fixes it.
- Windows uses both `RUNNER~1` and `runneradmin` for the same temporary directory. The
  package smoke compares canonical working paths, matching the actual launcher check,
  rather than requiring identical path spelling.
- Studio now uses `actions/setup-node@v7`, removing the Node 20 action deprecation while
  keeping the selected application runtime on Node 22.

Final local checks passed: all **33 frontend tests**, typecheck/lint, Actionlint,
repository-wide Ruff lint/format, skill structural/runtime checks, and two CI gate tests.
The complete packaged workflow also passed with `TMPDIR` pointing through a directory
symlink: **11 browser tests in 3.2 minutes**, plus runtime selection/immutability and
owned/attached shutdown. Retained artifacts:
`/tmp/opencode/studio-ci-LkdO4F-alias/nexuml packaged smoke 9I1hRH/`.
This reproduces the alias mechanism on Linux, not a claim of local macOS/Windows runs.

Run `39344195578` confirmed the skill check, Python lint/format gates, all three
Python/GPU suites, Studio static checks, and Linux tool packaging passed. Remaining
failures were resolved at their responsible layers:

- Canonicalize both sides of the launcher entrypoint comparison: Windows can retain a
  short path in the module URL as well as argv.
- Keep the browser test subprocess's original environment. Its initial Chromium launch
  already finds the correct cache; inferring a cache root from three executable parents
  breaks on macOS's `.app` bundle. Installed API/UI services still isolate their HOME.
- Link the out-of-docs Studio README through the repository URL and annotate the existing
  `Observation.progress` boolean return, fixing both strict-documentation warnings.
- Two Linux project assertions expired after five seconds while validation was still
  running. Failure-context snapshots show pending requests; screenshots captured just
  afterward show the expected dialog/error. Those two assertions now use the existing
  30-second API budget. No retries, sleeps, or acceptance conditions were removed.

Local final proof passed: strict MkDocs build to a disposable output directory; four
observation tests; 33 frontend tests; frontend lint; focused Ruff lint/format; and the
entire packaged workflow through an aliased TMPDIR, including **11 browser tests in
3.3 minutes** and owned/attached shutdown. Artifacts:
`/tmp/opencode/studio-ci-LkdO4F-alias/nexuml packaged smoke tSMWjW/`.
Documentation tools were installed only in a disposable test environment. Hosted
macOS/Windows completion remains required before claiming portability acceptance.

Correction: the Python type-check step uses `continue-on-error`; its successful step
conclusion was initially misread. Its logs report **61 diagnostics**, and the final
static gate correctly rejects them. Those diagnostics span feature code and skill/test
fixtures and are a separate remaining blocker, not a passing gate or suppressed check.

## Production packaging and release hardening — 2026-10-20

Scope: PR #16 / NEX-115, independent Python/npm installation, packaging, compatibility,
hosted transport and release CI. No UI redesign, Python provisioning, package publication,
release tag or merge. Maintainer override disables macOS/Windows CI; their original
portability acceptance remains unchecked. Linux CPU jobs use ika-runner, Python GPU
suites use the explicitly approved ika-runner-gpu pool, and the future npm publish-only
GitHub-hosted exception is required by npm OIDC.

### Diagnosis and corrections

- Reviewed run `39346484315` logs and all six diagnostic artifacts. Browser assertions
  were observing pending validation/catalog/save/review operations. Existing helpers now
  await the exact response body and settled UI state within the existing API budget;
  no retries, blanket timeout increases or acceptance checks were removed. Raw settings
  use the actual semantic alert. Native reference and pending status are separate events;
  the Python cancellation fixture now waits for both before asserting pending.
- Compared original HEAD and `origin/dev` with ty 0.0.89 and all 248 CI dependency
  versions matched: 76 versus 21 diagnostics (the historical run logged 61). Targeted
  type/validation-boundary fixes now give zero diagnostics. Exactly five original base
  suppressions were restored for PyTorch's documented `nn.Module.forward` override versus
  its mutable-Callable stub; no blanket ignores, exclusions or disabled gates were added.
  A typing import initially broke Torch package export; removing that import preserved
  the existing export dependency boundary, verified against newly built wheels.
- Retained standard `next start`: the standalone trace contains Linux Sharp/libvips
  binaries. One portable npm tarball instead uses npm-resolved Next/React dependencies;
  bundled frontend-only dependencies are build-time dependencies. The temporary runtime
  copy protects installed assets/cache and only three runtime packages need links.
- Shared launcher/browser compatibility policy requires stable NexuML >=0.2.1,<0.3.0
  and interface 2 before exposing the workbench/catalog. Runtime identity reports Python
  version/executable separately. Hosted mode stays loopback by default and requires a
  separate private proxy credential; spoofed Host/Origin alone cannot obtain API access.
  Existing package smoke exercises an actual authenticated proxy and Next HTTP/WS path,
  including negative access paths; Next upgrade rejection is not assumed to be HTTP401.

### Verification and release boundary

- Frontend typecheck/lint, 36 unit tests and production build passed. The rebuilt
  project-installed artifact passed all 11 browser workflows, including real CPU
  training/reconnect/export, save conflict recovery and custom library removal.
- Full Python suite passed: **463 passed, 8 skipped**, 62.46% coverage with the existing
  50% gate unchanged. The complete uv-tool package smoke passed all **11 browser tests**,
  offline assets, unchanged Python/npm installation records, explicit/PATH selection,
  saved YAML CLI build, owned/attached shutdown and actual hosted HTTP/WebSocket security.
  Both modes used the same tarball. Core-only wheel installation without the API/library
  extras also passed. Evidence: `/tmp/opencode/studio-hardening-final/`,
  `/tmp/opencode/nexuml packaged smoke vMaDWv/` and
  `/tmp/opencode/studio-hardening-coverage-final.xml`. Remote GPU/platform acceptance is
  not inferred from these local CPU results.
- Repository Ruff lint/format and ty (explicit isolated interpreter), strict MkDocs,
  Actionlint, Python wheels/sdists/Twine and distribution contents checks passed. Every
  Python file in both final wheels matches the checkout. Runtime npm audit reports zero
  vulnerabilities; five existing development-only ESLint/braces advisories have no safe
  compatible fix advertised and were not hidden with a forced dependency downgrade.
- The actual npm publication metadata validator accepted the built artifact and rejected
  wrong commit identity and tampered bytes, without publishing. CI builds one npm
  tarball with integrity/commit metadata, uploads immutable artifacts and tests those
  exact files in clean project/tool installations. Normal startup never changes installed
  dependencies. The required gate fails on failed/skipped/cancelled mandatory jobs.
- OpenSpec 1.14.1 strict validation reports the same three pre-existing long-requirement
  warnings in the untouched workbench spec on both original HEAD and this checkout.
  They were not suppressed or rewritten outside the approved packaging/security scope.
- Final full Python/package commands and remote Actions outcomes are recorded on PR #16;
  no local Linux result establishes macOS/Windows portability. CI must pass for the exact
  release commit before the protected publish workflow can consume its artifact.
- Initial pushed run `39423072836` passed Python/Studio static checks and the single
  Studio build. All three GPU assertions correctly failed because ika-runner has no
  CUDA, so dependent package jobs did not run and the required gate failed closed.
  The maintainer explicitly approved restoring ika-runner-gpu only for those mandatory
  suites; no CUDA checks were removed. Both local project and tool package commands
  subsequently completed successfully, including all HTTP/WS and ownership checks.
- Follow-up `39443435790` passed Python GPU suites on 3.12/3.13/3.14, Python/Studio
  static gates and the single Studio build. Python package validation then exposed the
  institute runner's unwritable default uv tools directory under HOME. The package job
  now gives uvx explicit per-run tool/bin directories under writable runner temp, rather
  than changing HOME permissions, installed packages or weakening validation.
  Twine validation of both wheels/sdists passed offline with HOME deliberately set to
  unwritable `/proc` and those explicit tool paths; focused CI tests and Actionlint passed.
- GitHub repository approval policy was changed and confirmed from first-time-only to
  **all external fork contributors**. CI does not run fork PR checkout code. Organization
  runner-group inspection returned 403: its owner must confirm approved-workflow
  restrictions and ephemeral/network-isolated runners. npm's public registry returns
  E404 for Studio and the GitHub `npm` environment is not yet configured; package/scope,
  MFA bootstrap if required, protected environment/tags and Trusted Publisher setup remain
  owner actions documented in `docs/development/release.md`.

### Installed project workflow aggregate deadline

- Exact-head run `39446250947` (`bf0ec2a`) passed both static gates, all three
  GPU/Python suites, the single Studio build, Python packaging and the complete
  installed uv-tool mode. Installed project mode passed 10/11 browser tests; its
  required CI gate correctly failed rather than hiding the remaining failure.
- Extracted Playwright's embedded step report for both modes. Tool mode completed
  the combined edit/save/check/build/train/reconnect/export/responsive workflow in
  72.656s. Project mode completed training at 101.71s, reconnected at 112.31s and
  accepted export at 117.66s. The 120s **test** deadline interrupted its unchanged
  30s export assertion after only 2.45s, with export still running/connected.
  Ordinary selected-interpreter calls took about 5s versus 2.8s in tool mode;
  no individual API deadline expired.
- This single combined workflow now has a 180s aggregate budget, enough to honor
  the existing final export/open checks after the measured setup. The default 120s
  deadline, 30s API/export waits, 45s training wait, all 11 workflows, assertions and
  artifact/security/ownership checks remain unchanged. No retries or product edits.
- Focused frontend typecheck/lint, 36 unit tests, strict MkDocs and the 11-test listing
  passed. A source comparison confirms every prior workflow line/assertion remains
  verbatim apart from the two added deadline/comment lines; default config, helpers
  and package smoke are byte-identical to the previous head. Full project-installed
  same-tarball proof passed all 11 workflows in 3.5m, including CPU workflow in 48.8s,
  offline/dependency invariants, hosted HTTP/WS security and owned/attached cleanup.
  Evidence: `/tmp/opencode/nexuml packaged smoke dHz9dG/`.

### Linux developer preview: first-launch polish

- The welcome screen exposes actual backend-reported NexuML/Python identity and a
  collapsed executable/directory/interface disclosure. The existing footer retains
  runtime details after loading, rather than adding another permanent panel.
- `Try an example` uses installed scenario discovery and the original resolve route.
  Inspection of `synthetic-linear-ae-reconstruction` confirmed generated local data,
  a small linear autoencoder and no service/download dependency. Only its editable
  draft gets one CPU epoch / one device via normal config validation; library defaults,
  graph projection, YAML save and explicit Run review/confirmation remain unchanged.
- Existing launcher errors now identify the selected executable, invalid directory,
  detected incompatible version or startup exit and user-run recovery commands. The
  browser reports disconnection while retaining the draft; runtime identity refresh
  uses the existing authenticated endpoint, with query retries still disabled.
- Extended existing unit/browser/package checks cover the welcome actions, actual
  identity, missing example, incompatible interface, retained disconnected draft,
  real CPU example training/results/logs/artifacts and default browser-opener invocation.
  All previous browser assertions and timeouts remain. No Python, transport, security,
  dependency or platform-matrix changes. The two Studio guides lead with Linux setup
  and first training, keeping unpublished-package and portability limitations explicit.
- Local frontend typecheck/lint, 37 unit tests, production pack and strict MkDocs passed.
  Full local Python suite: 463 passed / 8 skipped; ty and Ruff lint/format passed.
  API transport/optional-extra checks: 5 passed. Read-only integrated review found no
  material defects. Complete project-installed final-tarball proof passed 13 browsers
  without retries in 4.1m; the new CPU example workflow took 23.3s. Automatic opener
  invocation, selected interpreter, offline assets, dependency invariants, hosted
  HTTP/WebSocket protection and owned/attached shutdown all passed. Desktop/mobile
  welcome screenshots were inspected. Evidence: `/tmp/opencode/studio-preview-polish/`
  and `/tmp/opencode/nexuml packaged smoke 1UFO8n/`. Full uv-tool proof and exact pushed
  head CI are required before the final remote-green claim.
