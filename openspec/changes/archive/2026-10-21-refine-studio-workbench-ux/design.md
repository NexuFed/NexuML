# Design

## Context

See `proposal.md` for motivation and scope. The baseline contracts are in `../add-local-nexuml-studio/specs/studio-workbench/spec.md` and `../add-discoverable-execution-backends/specs/studio-run-target-selection/spec.md`.

The current Next.js/React workbench already has one Zustand draft, schema controls, React Flow stage containers, revision-bound checks, a staged Base UI Run dialog, and observed operations. Reuse these. The local API and installed NexuML remain authoritative.

Inspection of source and existing browser-test screenshots identified these presentation seams:

- `studio.tsx` renders the source-opening bar on every view and stacks YAML above the editor. It combines Components and Structure in one sidebar and places undo/redo with diagnostics.
- `globals.css` reserves 148px above the graph and combines a 600px editor minimum with substantial surrounding chrome. The component list scrolls inside an already scrolling panel.
- `fields.tsx` recursively repeats labels, type selectors and Expert editors. Raw buffers currently belong to individual field components, so new tab boundaries must preserve them.
- `execution.tsx` mixes progress, all metrics, logs, results, artifacts and exports in one long page; its operation selector also includes builds and exports.
- `run-review.tsx` already binds discovery/preflight to staged settings and freezes the confirmed run. This redesign changes its hierarchy, not that sequence.

Audience assumption: an ML practitioner who knows model/training concepts but should not need NexuML implementation knowledge. Supplied mocks 8–9 guide editor composition, 1–4 guide restraint, 5 guides settings grouping, and 11–12 guide monitoring. The approved corporate tokens take precedence over their inconsistent colors and invented telemetry. Generic design-catalog results were not a verified workbench match.

## Goals / Non-Goals

**Goals:** make the current task, selected object, next action and diagnostic state obvious; keep common editing readable at laptop sizes; preserve all existing semantic and recovery behavior through focused presentation changes.

**Non-Goals:** a new graph engine, schema framework, component/backend registry, router architecture, dashboard database, onboarding wizard, code editor dependency, device/cost predictor, arbitrary DAG scheduling, or new backend telemetry. No new package is needed.

## Decisions

### D1. Task navigation and source actions share the existing state

Use **Pipeline**, **Training**, **Runs** as primary views. Source selection starts with **Choose a recipe** and **Open YAML**; opening a recipe remains an explicit action that can execute trusted installed Python. Once loaded, a scenario-name menu owns Open recipe, Open YAML, Save and Save as. Save on a recipe without a file prompts for a destination; an existing file saves to its current path. Preserve revision conflicts, sidecars and unsaved-draft replacement confirmation. Keep installed-library management reachable from the shell without permanently consuming a content row.

Move undo/redo beside editing actions. **Check** offers **Check fields** and **Build model**, with visible descriptions distinguishing schema validation from executing constructors/dummy forwards. **Run…** opens the existing review. These actions do not imply autosave or automatic build execution. Do not show Saved before a document has been saved; distinguish a recipe draft, unsaved changes and unapplied YAML.

The YAML control switches the editing area between visual content and the same draft's YAML. Keep Apply/Discard explicit. Switching back preserves an unapplied buffer, displays a return-to-YAML notice and keeps semantic edits/save/run blocked until resolution. Switching to Runs permits observation. Use existing menu/dialog primitives and bounded view state; no routing or project-storage layer is required.

**Alternative:** retain the permanent source bar. Rejected because opening a different document is occasional and currently displaces every task.

### D2. Pipeline is a viewport-filling editor

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ NexuML   Scenario ▾   Unsaved changes       Undo Redo  Save Check ▾ Run…  │
│ Pipeline   Training   Runs                              Libraries  YAML  │
├──────────────────┬────────────────────────────────┬──────────────────────┤
│ Structure |      │ Execution order                 │ LinearEncoder     ⋯ │
│ Components       │ 1 Encoder → 2 Decoder → 3 Loss   │ Settings | Routing  │
│                  │                                │                     │
│ Data             │  Data ──▶ Encoder               │ Output dimension    │
│ 1 Encoder        │               │                │ [ 2               ] │
│   LinearEncoder  │               ▼                │                     │
│ 2 Decoder        │            Decoder ──▶ Loss     │ Hidden dimensions   │
│   LinearEncoder  │                                │ [ 8 ]  + Add        │
│ 3 Loss           │                                │                     │
│ + Add stage      │ −  100%  +  Fit  Arrange        │ ▸ Advanced          │
├──────────────────┴────────────────────────────────┴──────────────────────┤
│ Problems (2)   Build needs rechecking                  Runtime connected │
└──────────────────────────────────────────────────────────────────────────┘
```

Annotations:

1. The shell and compact footer use content-sized rows; the editor uses the remaining `100dvh` space with shrinkable children and independent panel scrolling. Training/Runs scroll their own content. Avoid height calculations tied to a fixed number of visible bars.
2. Structure is the default left tab after loading. Components has search and installed categories, with a single list scroll region. Keep category provenance and version available without a version row on every collapsed item.
3. Add layer opens Components and displays the destination/slot near insertion. With no explicit destination, derive it from the selected stage/layer and state it before insertion; otherwise ask. Preserve before/after controls, drag insertion and skipped-stage warnings. A global browser must not silently insert into the first stage.
4. Keep the existing order strip, but compact its heading, handles and stage items into one row. It scrolls internally when needed. Structure retains keyboard/single-pointer move and transfer controls; focus/selection reveals secondary row actions as well as hover.
5. Default side widths stay close to the existing 280px/320px. Visible collapse controls and separator dragging reuse width state; retain keyboard/single-pointer width controls. Use bounded widths that leave a usable canvas and the existing single-panel mode below 1280px. Mobile exposes Structure/Components, Canvas and Properties as non-overlapping views.

**Alternative:** delete the order strip or infer order from layout. Rejected because explicit execution order is part of NexuML's model and existing authoring contract.

### D3. Compact nodes without hiding graph meaning

Reduce excessive stage-header height and default empty space, leaving order, name, layer count and skipped state readable. Keep title, short parameter summary, active ports and issues on layer cards. Connected ports and required missing inputs remain visible. Optional unconnected ports/add-input rows are disclosed by selection or a labeled routing affordance, including keyboard focus; keep edge endpoint handles stable while changing presentation.

Use approximately 14px node titles and 12px supporting text at the normal editing scale. Open at a readable scale rather than forcing all nodes into the viewport. Fit view remains an explicit overview action; selection from Structure offers a focused view of the chosen node without changing layout or execution. Keep stage/layer numbers visible and the distinction between key routing and execution order stated near the order controls.

Preserve saved positions/sizes. Apply compact geometry defaults to new/unsaved layouts and explicit Arrange; never shrink existing sidecars on load. If geometry constants change, update projection, containment, insertion and hit testing together at their existing owners. No automatic arrangement on property edits. Final key shapes remain a source-bound build summary: the API does not supply per-node shape provenance, so do not attach final-key shapes to every matching port.

**Alternative:** reproduce the mocks' fully populated node forms and inferred shapes. Rejected because the inspector already owns detailed editing and the runtime supplies limited shape evidence.

### D4. Settings and Routing are views of the same component

The inspector leads with the selected component name and **Settings / Routing** tabs. Display-name editing is a small explicit action; definition identity/version and layout controls live under Advanced. Routing owns key lists, aliases, label/metadata routes and the existing non-drag connection controls, keeping their semantics visible through readable labels. Errors reveal the appropriate tab/section and focus the exact field.

Required fields and explicitly chosen common fields appear first. Other typed settings remain available under Advanced. Provide one raw editor at each edited component/settings-section boundary instead of a raw editor for every array or mapping. Genuinely untyped factory values retain a usable structured editor; raw editing remains discoverable. Preserve invalid typed/raw/YAML input across panel/tab changes using mounted hidden panels or narrowly lifted buffers, rather than introducing another document store.

For schema controls:

- Nullable values use one explicit null/value control: **Not set / Set value** unless the schema or known field contract establishes a more specific meaning. Label null **Use default** only when it really inherits a default; known loader batch-size inheritance is **Use training batch size**. Do not display an inferred runtime default as an explicit configured value.
- Fixed training batch size is a numeric input; the existing automatic object is **Automatic**, exposing its real bounds and safety fields. Selecting a mode is explicit and undoable.
- Single usable branches render their control directly; meaningful unions keep an accessible mode choice. Use contextual labels where meaning is known, while preserving valid string/integer/custom alternatives and exact serialized types. Unknown unions keep a labeled fallback; do not coerce a numeric string into a number merely to shorten the UI.
- Lists use compact labeled rows with Add/Remove. Numeric values support direct entry, including scientific notation where permitted; do not substitute precision sliders.
- Schema titles/descriptions/constraints remain authoritative. Labels such as Learning rate can be curated for known scenario fields, but no frontend optimizer catalog, constructor introspection or fabricated defaults are introduced.

**Alternative:** per-model handcrafted forms or a schema-form dependency. Rejected because the installed catalog is extensible and the existing recursive controls are sufficient with restrained presentation rules.

### D5. Training has five navigable sections

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ NexuML   Scenario ▾   Unsaved changes                Save Check ▾ Run…   │
│ Pipeline   Training   Runs                                         YAML │
├──────────────────┬───────────────────────────────────────────────────────┤
│ Basics           │ Basics                                                │
│ Optimizer and    │ Epochs              [ 3                             ] │
│ schedule         │ Batch size          [ Fixed ▾ ] [ 32                ] │
│ Data loading     │ Learning rate       [ 0.001                         ] │
│ Execution        │ Accelerator         [ auto                          ] │
│ Checkpoints and  │ Devices             [ auto                          ] │
│ logging          │ Precision           [ 32-true                       ] │
│                  │ ▸ Advanced                                            │
├──────────────────┴───────────────────────────────────────────────────────┤
│ Problems (0)   Build not checked                       Runtime connected │
└──────────────────────────────────────────────────────────────────────────┘
```

Wireframe values are illustrative, not injected defaults. Use section navigation with one focused section, a readable form width and at most two field columns where they fit; mobile uses a labeled section selector. Keep section buffers/edits when navigating and switch sections when following a diagnostic.

Map the five sections to current fields, not a new schema: Basics contains epochs, batch size, learning rate and training accelerator/devices/precision; Optimizer and schedule contains the existing factory definitions; Data loading contains loader settings; Execution reuses backend settings; Checkpoints and logging includes checkpoint, logging, callbacks, export configuration and local resume. Keep remaining training/evaluation settings reachable under relevant Advanced sections. Objectives continue linking to the Pipeline objective inspector.

Distinguish trainer device settings from backend placement resources. Show an explicit loader batch-size override beside the training batch-size field when it takes precedence. Preserve independently configured optimizer arguments rather than merging similarly named values into one field. Factory identity and compact existing arguments are editable; custom args/kwargs remain available without fabricated constructor controls.

**Alternative:** three tall, equally weighted cards showing all settings. Rejected because vertical length and repeated nested controls obscure common adjustments.

### D6. Diagnostics guide action without occupying the page

The footer shows problem count, build state and runtime connection. Expand Problems manually or after a failed explicit check/build/apply. Group draft problems, build results and discovery diagnostics by their actual source; use only supplied severity and do not imply that every discovery issue blocks the loaded scenario. Keep meaningful new failures announced and visible even while details are collapsed. Avoid repeated live announcements for numeric updates.

Field-addressable issues navigate through Pipeline/Training and inspector/section state before focusing the control. Build messages without an address stay linked to the checked source and operation. Show **Not checked**, **Checking**, **Passed**, **Failed**, or **Needs rechecking** from existing status/signature evidence. A successful field check never implies a successful build. Frozen-run failures remain in Runs, with existing actions to inspect or open that run's settings as a draft.

**Alternative:** a permanent expanded diagnostics console or generic green Ready badge. Rejected because they consume attention or overstate evidence.

### D7. Run review and run inspection prioritize the current decision

Keep the current Run dialog and its staged copy, cancel behavior, review invalidation, native preflight and confirmation. Group fields into destination and resources using the backend's existing presentation metadata; preserve a generic form fallback for contributed backends. Capacity shows a concise source/age/completeness summary with details available, including unknown values and known blockers. Keep consequential warnings visible. Review and confirmation remain distinct, with one highlighted next action at a time; shortening copy must not remove required review or capability explanations.

```text
┌──────────────────────────────────────────────────────────────────────────┐
│ NexuML   Scenario ▾                                                     │
│ Pipeline   Training   Runs                                              │
├──────────────────────────────────────────────────────────────────────────┤
│ Run [ train · <id> ▾ ]                  Running       Stop run           │
│ <reported phase>   <reported epoch/batch progress>   Connected            │
│ Frozen configuration <revision>    Draft differs · View configuration    │
│ Metrics   Logs   Artifacts   Configuration                               │
│                                                                          │
│ Total loss                            Reported evaluation metrics        │
│ ┌──────────────────────────────┐     ┌──────────────────────────────────┐ │
│ │ Train ──  Validation - -      │     │ Values/curves when reported      │ │
│ │           observed curves    │     │                                  │ │
│ └──────────────────────────────┘     └──────────────────────────────────┘ │
│ Results                                                                  │
│ <returned evaluation values, when available>                              │
└──────────────────────────────────────────────────────────────────────────┘
```

Annotations:

1. Runs uses the current operation list/identity, including explicitly labeled build and export operations. Keep the existing selector; no new history database or guessed scenario/timestamp metadata. Runs remains accessible with no draft loaded.
2. Metrics, Logs, Artifacts and Configuration share the selected operation. Status/progress/errors remain visible across these tabs. Configuration is read-only frozen YAML with the existing Open as draft action; it is distinct from the editable draft YAML control. Fetch frozen settings through the existing operation-config read and compare draft/frozen configuration plus stage order with the same semantic comparison; never compare a local JSON signature directly to a server revision hash. An unloaded draft or unavailable frozen configuration is not labeled as matching/differing.
3. Keep the observation subscription at the selected-run boundary so changing tabs neither resubmits work nor drops collected observations. On reload the user can reselect the existing operation and replay its observations. Use backend capabilities to choose a useful initial tab (metrics for training with telemetry, logs for builds/log-only backends, artifacts for exports); do not keep changing tabs while the user reads.
4. Give `train/loss` and `val/loss` one labeled comparison chart when available: they are the framework's same weighted objective and both carry optimizer steps. Retain their actual x coordinates, share y limits, distinguish lines by label/style, and show single observations as points. Keep other metric keys separate unless compatibility is explicitly known; do not infer common units from similar names, normalize loss with accuracy, smooth, interpolate missing phases or relabel steps as epochs. Continue using existing `metricPoints` and SVG rendering.
5. Logs and structured results reuse the current observation/result renderers. Artifacts owns downloads and export actions for this operation; launching an export selects the new export operation. All available metrics remain discoverable. Technical source hashes, native references and telemetry details remain available in Configuration/details.
6. Stop, resume, native inspection and export follow actual capabilities. Unsupported telemetry gets a clear local explanation instead of empty charts. Preserve stale/disconnected observation, unknown remote ownership and local-versus-remote cancellation distinctions. No estimated completion time, GPU utilization, cost or per-node shapes without an authoritative source.

**Alternative:** a monitoring modal or a new analytics dashboard. Rejected because existing operation records and a dedicated Runs view already support the workflow, including mobile observation.

### D8. Visual hierarchy uses the existing design language

Retain `#101010` canvas/background, `#1a1c1f` panels, `#1e2024` nodes, `#e2e2e7` foreground, `#c3c6d1` secondary text, and `#188FD5`/black primary actions. Existing primary/tertiary/error tones handle selection and statuses with labels/icons. Use semantic CSS roles for repeated treatments rather than new per-component colors.

Use bundled Geist for controls/body and Montserrat for view headings/brand. Aim for 14/20 control/body text, 12/16 secondary metadata, 13/18 code and 24/32 view headings. Reserve tiny technical annotations for secondary detail, not primary navigation or settings. Keep the 4px spacing baseline and existing 4/8/12px control/card/dialog radii. Reduce decorative outlines; use surface differences and section spacing. Maintain required control/focus contrast and 4.5:1 normal-text contrast.

Use restrained interaction transitions, visible keyboard focus, labeled icons and reduced-motion behavior. Touch controls reach 44px targets. At desktop sizes separator resizing has keyboard and single-pointer alternatives; dialogs restore focus; hidden panels cannot retain invisible focus. At 1024px and 390px widths use non-overlapping panel/section views with reachable actions and no document-level horizontal overflow. A draggable canvas is intentionally pannable.

**Alternative:** adopt the mocks' neon palette, broad sliders and heavy card treatment. Rejected because corporate consistency and hierarchy address this product's needs more directly.

## Risks / Trade-offs

- **Existing in-flight work shares these files** → make focused frontend changes against the current checkout; preserve predecessor contracts and do not reset other changes or close their outstanding release tasks.
- **Collapsing controls hides important configuration or errors** → selected/required/error states stay discoverable, diagnostics open the correct section, and every editing action retains a non-drag path.
- **Tabbed editors lose local buffers** → preserve mounted state or lift only affected buffers; verify invalid-input navigation before simplifying raw editors.
- **Compact stage geometry changes placement** → preserve saved layouts, update all related geometry consumers together, and rerun existing containment/transfer checks.
- **Human labels erase union or inheritance semantics** → preserve exact types and defaults; retain generic fallback; verify automatic batch size, loader override and custom factories.
- **Paired metric charts imply unsupported equivalence** → only group the known total-loss pair on its recorded step axis; other keys retain their own scale and identity.
- **Polish expands into a framework rewrite** → reuse current primitives/models, extract only cohesive UI sections needed by this work, and stop after the specified workflow and responsive evidence pass.

## Migration Plan

1. Record the current UI baseline and run focused existing frontend checks against the current source. Read applicable installed Next.js guidance before implementation. Separate pre-existing failures from this change.
2. Implement shell and Pipeline hierarchy first, then inspector/Training disclosure, then Run review/Runs composition. Extend existing focused checks alongside the affected interactions.
3. Verify typecheck, lint, unit/model tests and production build, then the isolated real CPU browser workflow and relevant authoring/container/run-target/metrics scenarios. Capture Pipeline, Training and Runs at 1440×1000, 1024×900 and 390×844; inspect normal, error, empty and unsupported states. No live cluster launch is part of acceptance.
4. Update the Studio guide to the new labels/actions and retain honest runtime explanations. No YAML, sidecar, API or operation-record migration is needed; rollback is the previous frontend with compatible runtime/data.
5. Before archival, establish the prerequisite `studio-workbench` main spec, then apply this change's non-overlapping additions. Do not mark prerequisite cross-platform/release gates complete from this frontend verification.
