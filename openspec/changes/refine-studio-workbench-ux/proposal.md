# Proposal

## Why

Studio exposes the main NexuML workflow, but persistent setup controls, crowded panels, tiny graph labels and schema-shaped forms make routine work difficult to discover. Refine the existing interface into a readable, canvas-first workbench with focused configuration and run inspection, using the approved corporate design.

## What Changes

- Organize the shell around **Pipeline**, **Training** and **Runs**; place metrics, logs, artifacts and frozen configuration within the selected run. Keep build/export observations accessible in Runs.
- Offer **Choose a recipe** and **Open YAML** at entry, then move source switching and file actions into the scenario menu. Present YAML as an alternate editor of the same draft rather than a panel stacked above the workbench.
- Give the canvas the remaining viewport height. Separate **Structure** and **Components** into tabs, make insertion contextual, compact stage/order controls, and expose accessible panel collapse/resize controls.
- Lead the inspector with component settings and a separate **Routing** tab. Keep required connection problems visible and disclose optional ports, layout controls and raw values when relevant.
- Group Training into Basics, Optimizer and schedule, Data loading, Execution, and Checkpoints and logging. Simplify schema controls without losing supported types, inherited values or custom configuration.
- Consolidate field validation and executable build checking under **Check**; make Problems a compact, expandable, field-linked status surface. Preserve explicit code execution and current/stale build evidence.
- Clarify the existing staged Run review and prioritize actual progress and metrics in Runs, with capability-aware controls and subordinate technical details.
- Apply readable typography, restrained borders, clear surface hierarchy and responsive layouts using the existing blue/charcoal tokens and bundled fonts.

## Capabilities

### New Capabilities

- `studio-workbench`: Add task-focused navigation, progressive editing controls and run-centered inspection requirements to the same pending capability introduced by `add-local-nexuml-studio`. This capability is not yet present under `openspec/specs/`; this change uses non-overlapping ADDED requirements instead of creating a second workbench capability or copying the baseline requirements.

### Modified Capabilities

None in the current main-spec inventory. Existing Studio and backend behavior from the prerequisite changes remains binding; this proposal refines its presentation.

## Impact

- **Prerequisites:** the current `add-local-nexuml-studio` workbench and `add-discoverable-execution-backends` integration. Preserve their draft/routing/run contracts and outstanding verification tasks. Archive the baseline workbench before this additive follow-up.
- **Frontend:** `studio/components/studio.tsx`, `canvas.tsx`, `structure.tsx`, `component-browser.tsx`, `fields.tsx`, `execution-settings.tsx`, `run-review.tsx`, `execution.tsx`, and `studio/app/globals.css`; existing model helpers only where presentation needs them.
- **Verification/docs:** adapt existing model/browser checks and `studio/README.md`; record desktop, tablet and mobile evidence for the revised workflow.
- **Compatibility:** ordinary YAML, layout sidecars, API/interface version, installed discovery, execution order, backend capabilities and frozen run records stay compatible. No additional dependencies, hosted features, project database, telemetry service or ML semantics are required.
