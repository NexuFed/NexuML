# Tasks

## 1. Shell, source actions and checks

- [x] 1.1 Read applicable installed Next.js guidance and prerequisite Studio/backend contracts, inspect current diffs and capture the current Pipeline/Training/Runs baseline; run `npm run typecheck` and `npm test` in `studio/` and record pre-existing failures separately before edits.
- [x] 1.2 Replace the persistent source bar with entry actions and a scenario menu containing open/save/save-as actions and accurate save state; adapt `workflow.spec.ts` to verify first recipe save, file reopen, replacement cancellation, revision conflicts and sidecar preservation.
- [x] 1.3 Introduce Pipeline/Training/Runs navigation and alternate draft YAML editing, retaining existing observation content until the Runs tasks land; verify navigation preserves draft/selection/operation and invalid YAML survives view changes while edits/save/run remain blocked.
- [x] 1.4 Move undo/redo beside editing actions, add the Check menu and compact Problems surface with explicit build effects and source-bound state; verify failed checks expand diagnostics, field navigation works, semantic edits stale build evidence and navigation never runs code.
- [x] 1.5 Update `studio/README.md` for source/file actions, YAML and checks; walk the documented actions in the browser and rerun the affected workflow assertions before proceeding.

## 2. Canvas and inspector hierarchy

- [x] 2.1 Apply the viewport-filling shell, corporate typography/surface hierarchy and bounded collapsible/resizable panels using existing state and primitives; verify computed action colors, remaining canvas space, separator/width-control keyboard operation and focus recovery at 1440×1000 and 1280×800.
- [x] 2.2 Separate Structure/Components tabs, preserve search and make layer insertion show the selected destination/slot or request one; extend `authoring.spec.ts` and `containers.spec.ts` to verify click/keyboard/drag insertion, explicit before/after placement, skipped-stage warnings and no implicit first-stage fallback.
- [x] 2.3 Compact the order strip and new-layout node/stage geometry, preserve connected/required ports and add accessible optional-port disclosure and focused node navigation; verify existing container move/resize/transfer/order/undo checks, saved sidecar geometry, readable initial scale and explicit Fit/Arrange behavior.
- [x] 2.4 Add Settings/Routing inspector tabs, lightweight display-name editing and Advanced identity/layout controls; verify diagnostic navigation reveals the correct tab, named/aliased/metadata routes retain non-drag editing and existing routing/remove/undo checks pass.
- [x] 2.5 Update the Studio authoring guide for panel tabs, insertion and focused graph navigation; verify the documented keyboard-only path and capture a normal and error-state Pipeline screenshot with required labels readable.

## 3. Forms and Training sections

- [x] 3.1 Simplify existing schema controls for nullable values, fixed/automatic batch size and compact collections while preserving string/integer/custom alternatives; extend focused model/browser checks for exact serialized types, actual automatic fields, scientific numeric entry, null semantics and unrendered-field retention.
- [x] 3.2 Consolidate raw editors at component/settings-section boundaries and preserve invalid field/raw buffers across tabs/sections; verify invalid input can be left, reopened and corrected without loss, and custom factory args/kwargs remain editable without invented schemas.
- [x] 3.3 Group Training into the five designed sections, include relevant Advanced settings and link loader batch-size overrides and objectives to their actual owners; verify section navigation/error focus, loader inheritance versus override, separate trainer/backend resource fields, resume settings and lossless edit/save/reopen in existing authoring/workflow checks.
- [x] 3.4 Update the Studio configuration guide with the section layout, inheritance labels and raw fallback; verify the documented common-settings and custom-factory paths and capture desktop/mobile Training evidence.

## 4. Run review and observation

- [x] 4.1 Reorganize the existing Run dialog using backend presentation metadata and its generic fallback, with concise capacity/source/age and one emphasized next action; adapt `run-targets.spec.ts` to verify contributed backends, unknown capacity, known blockers, review invalidation, cancel preservation, duplicate prevention and keyboard focus restoration.
- [x] 4.2 Recompose Execution as Runs with Metrics/Logs/Artifacts/Configuration tabs around one selected-operation subscription; verify training/build/export selection, use with no draft, retained observations across tabs, read-only frozen settings, correct draft comparison, Open as draft confirmation and export selecting its new operation.
- [x] 4.3 Combine only the known total train/validation loss series using existing metric extraction and SVG controls; verify shared actual optimizer-step/value scales, unequal series lengths, a visible single point, missing phases and separate discoverability of accuracy/custom scalars in focused metrics checks and `live-metrics.spec.ts`.
- [x] 4.4 Adapt the run workflow and guide for available progress, logs, results, artifacts and capability-aware actions; verify reload/reselection without relaunch, stale/gap notices, log-only backends, unsupported stop/resume, frozen-source failures and existing local export behavior, with desktop/mobile Runs screenshots.

## 5. Integrated acceptance

- [x] 5.1 Run `npm run typecheck`, `npm run lint`, `npm test` and `npm run build` in `studio/`; then run the updated browser suite via `npm run test:browser` against the established isolated CPU test runtime. Verify the complete open/edit/save/check/build/train/reconnect/export flow with no live cluster launch, fabricated telemetry or external asset fetch.
- [x] 5.2 Review Pipeline, Training and Runs at 1440×1000, 1024×900 and 390×844 plus the compact 1280×800 editor; verify readable normal/error/empty/unsupported states, no document overflow, 44px touch targets, contrast, keyboard/non-drag paths, dialog focus and reduced motion. Record screenshots and remaining failures in this change's verification notes.
- [x] 5.3 Review the final diff and acceptance coverage against this change's specs; verify no new dependencies or changes to executable YAML/sidecars/API/run contracts, preserve unrelated work and predecessor release gates, and stop when these scoped checks pass.
