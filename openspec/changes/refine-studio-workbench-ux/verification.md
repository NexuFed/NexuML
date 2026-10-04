# Verification

## Baseline (2026-10-04)

- Clean implementation checkout at `e49754c` before edits.
- Read Studio's installed Next.js `use-client` guide and prerequisite workbench/run-target specifications.
- `npm run typecheck`: passed. `npm test`: 6 files, 22 tests passed.
- Fresh 1440×1000 Pipeline, Training and empty Runs captures: `/tmp/opencode/studio-ux-before-{pipeline,training,runs}.png`.
- Test services: isolated working directory `/tmp/opencode/studio-workbench-ux-o62WMY`, CPU-only Torch 2.14.1 installation with current checkout's `src` on `PYTHONPATH`, loopback UI/API ports 41640/41641. No package installation or cluster submission.
- OpenChamber browser returned an error page; used the repository's installed Playwright instead.

## Focused implementation checks

- Source/YAML/conflict/sidecar checks: `workbench-ux.spec.ts` and adapted `workflow.spec.ts` passed against the CPU runtime.
- Graph gestures/order/skipped stages/insertion/undo/sidecars: both `containers.spec.ts` scenarios passed. `authoring.spec.ts` passed for typed fields, routing, removal, names and mobile controls. `diagnostics.spec.ts` passed for hidden/advanced fields and frozen-source failures.
- `training-ux.spec.ts` passed: exact fixed/automatic fields, scientific entry and incomplete buffers, string/integer alternatives, null loader inheritance, custom factory args/kwargs, retained settings and save/reopen.
- `run-targets.spec.ts` passed: contributed backend metadata, incomplete/stale capacity, denied submission/unsupported API, staged cancel, review invalidation, duplicate confirmation and native controls. These are routed fixtures, **not** cluster submissions.
- `run-observation.spec.ts` passed: fixture loss points (train steps 2/7, validation step 12) on shared axes, separate accuracy/custom scalars, one subscription across tabs, retained logs, gap/disconnect notices, semantic draft comparison, cancelled/confirmed draft replacement and unavailable frozen source. No workload submitted in this fixture test.
- `live-metrics.spec.ts` passed: real CPU training, live first-epoch points, completed classification metrics and replay after reload. Real CPU build/train/export workflow passed separately.
- At the latest focused gate: lint passed; 6 model test files / 23 tests passed; 3 form/run browser scenarios passed. Earlier full-suite attempt timed out after outdated diagnostics selectors; corrected them and reran affected checks. Missing/interrupted output was not counted as a pass.
- Visual inspection caught canvas toolbar overlap with a stage resize handle and mobile Check/Run overlap. Reserved canvas-control space and bounded mobile action columns fix the respective causes. Final integrated rerun pending below.

## Final acceptance (2026-10-04)

Source: planning commit `e49754c` plus this change's implementation diff, uncommitted at verification time. These results are frontend implementation evidence, not a release, packaged-runtime or cluster acceptance claim.

| Command | Final result |
| --- | --- |
| `npm run typecheck` (in `studio/`) | Passed |
| `npm run lint` (in `studio/`) | Passed, no warnings |
| `npm test` (in `studio/`) | Passed: 6 files, 23 tests |
| `npm run build` (in `studio/`) | Passed: Next.js 16.3.8 production build |
| `STUDIO_TEST_URL=http://127.0.0.1:41640 npm run test:browser` (in `studio/`) | Passed: all 11 scenarios, 3.3 minutes |
| `openspec validate refine-studio-workbench-ux --strict` | Passed |
| `git diff --check` | Passed |

The complete real CPU open/edit/save/check/build/train/reconnect/export workflow passed, with no page errors or external asset requests. Remote capacity, contributed backend and observation-edge cases use explicitly routed fixtures; no real cluster launch occurred. Retained source records, ordinary YAML and sidecar round trips remain covered.

Final-review corrections were verified before the last integrated run:

- Numeric input previously kept scientific-notation text after Undo had restored a different draft value. The buffer now tracks its own committed source, resets on external changes, and still retains incomplete text across navigation. `training-ux.spec.ts` verifies both the displayed value and serialized draft through Undo/Redo.
- Input borders previously measured only 2.04:1 against the canvas. Control boundaries now use a separate `#777b85` outline; decorative dividers stay restrained. Browser checks enforce at least 3:1 on input/control boundaries and 4.5:1 for normal text. Approved blue/black primary actions measure 5.93:1, input text 14.74:1, panel body 13.23:1 and secondary panel text 10.02:1.
- At the minimum 220px navigation width, long names and the collapse control were clipped. Tabs/names now wrap without overlapping order controls. Browser bounds checks cover this at both desktop acceptance sizes.
- The old live-metrics assertion assumed a loss card had exactly one point marker. It now targets the actual training series; `run-observation.spec.ts` separately proves that a later single validation point remains visible on the shared step/value axes.
- Failed config writes and failed sidecar writes do not report Saved. The source-menu check verifies successful retry after a partial config/sidecar failure using the updated config revision.

### Visual evidence

Screenshots are generated locally under `studio/test-results/` (ignored, regenerated by the browser command above). Reviewed normal Pipeline/Training/Runs at 1440×1000, 1024×900 and 390×844; compact editor at 1280×800; error, empty and unsupported states; mobile loss charts; and desktop/mobile Run review. Desktop panels scroll independently; tablet/mobile panel and section views are non-overlapping. The canvas is intentionally pannable, not forced into an unreadable overview.

The main capture directory is `studio/test-results/workbench-ux-documented-pa-4a40d-esponsive-visual-acceptance/`:

| Evidence | Files |
| --- | --- |
| Pipeline normal, all three sizes | `pipeline-normal-1440.png`, `pipeline-normal-1024.png`, `pipeline-normal-390.png` |
| Minimum-width desktop panels | `pipeline-1440.png`, `pipeline-1280.png` |
| Pipeline field error | `pipeline-error-1440.png` |
| Training, all three sizes | `training-1440.png`, `training-1024.png`, `training-390.png` |
| Runs empty, all three sizes | `runs-empty-1440.png`, `runs-empty-1024.png`, `runs-empty-390.png` |
| Retained real CPU run, all three sizes | `runs-1440.png`, `runs-1024.png`, `runs-390.png` |
| Narrow-layout loss charts | `runs-metrics-1024.png`, `runs-metrics-390.png` |

Run-review fixtures are under `studio/test-results/run-targets-runtime-schema-0a0fd--review-and-native-controls/`: `run-review-desktop.png`, `run-review-mobile.png`, `runs-unsupported-mobile.png`. Real workflow screenshots also include `desktop-pipeline.png`, `mobile-pipeline.png`, `tablet-pipeline.png` and `mobile-artifacts.png` in the workflow test output directory. The Next.js development indicator appears in these dev-server captures; it is not Studio UI.

Keyboard checks cover panel separators/width controls, collapse focus recovery, order/insertion alternatives, field-error navigation and Run dialog focus restoration. Touch controls use 44px minimum bounds on narrow layouts; shell/navigation/run-tab dimensions and non-overlap are measured. Reduced-motion CSS is present and checked under the corresponding media preference.

### Scope and remaining gates

- No new dependencies, external fonts/assets, configuration migration, API/interface changes or operation-record changes. Existing layout sidecar fields remain unchanged. Canonical local signatures compare configuration plus explicit stage order, ignoring mapping order but preserving serialized types; they are not compared directly to server revision hashes.
- No unresolved failures in this change's scoped acceptance. All 21 implementation tasks are complete.
- Prerequisite cross-platform, packaging and operator-authorized cluster/release gates were not run or marked complete. Archival still requires establishing the prerequisite `studio-workbench` main spec first. Verification itself performed no commit, push or archive.
