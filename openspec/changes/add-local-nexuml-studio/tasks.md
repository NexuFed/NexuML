# Tasks

## 1. Optional API and protected local startup

- [x] 1.1 Add the `api` extra and lazy `nexuml serve` CLI entrypoint with explicit working directory, loopback binding, allowed origin, and private token configuration; verify CLI help/startup tests and a core-only installation without FastAPI imports.
- [x] 1.2 Implement runtime/interface identity and structured transport error responses; verify selected interpreter/version/directory reporting, interface mismatch handling, and missing-extra diagnostics in focused API tests.
- [x] 1.3 Protect REST and WebSockets with token, Host/Origin checks, exact-origin CORS, first-frame WebSocket authentication, limits, and secret redaction; verify unauthorized REST/upgrade/frame rejection and that credentials never enter URLs/logs/events.
- [x] 1.4 Document API-extra installation, trusted local code, protected manual startup, and the REST/WebSocket authentication contract; verify documented startup against a temporary selected installation and isolated user configuration.

## 2. Existing discovery and configuration operations

- [x] 2.1 Expose existing library sources, component/scenario registries, per-definition schemas, backend metadata, and discovery diagnostics through fresh selected-runtime discovery; verify installed entry points, local roots, a failing plugin, and CLI/API parity using isolated test libraries.
- [x] 2.2 Expose existing library root add/remove behavior without a separate catalog; verify normalization, missing-root rejection, configuration isolation, and subsequent fresh discovery after edits.
- [x] 2.3 Implement scenario resolution and lowered configuration load/validate/serialize through existing NexuML APIs, including explicit stage-order transport metadata; verify identity/version round-trips, integer-like stage names, alias dictionaries, checkpoint/evaluation/execution fields, and that validation does not compile.
- [x] 2.4 Implement authorized-root file access, symlink/traversal rejection, revision-conflict detection, atomic configuration saves, and separate layout sidecar access; verify invalid input/external changes preserve the saved file and valid saves retain CLI-readable semantics.
- [x] 2.5 Document the discovery/configuration resources, schema limitations, semantic-not-formatting fidelity, and explicit scenario-resolution trust boundary; verify examples against the actual API and tests from this group.

## 3. Existing execution and observation

- [x] 3.1 Add owned selected-interpreter subprocess calls for explicit build/train/export work and a single-active-operation guard; extract shared CLI call logic only where needed, and verify native calls, CPU/working-directory behavior, responsive status requests, build timeout, busy rejection, and existing CLI regressions.
- [x] 3.2 Delegate local training to the existing session lifecycle and configured Ray execution to its existing entrypoint, preserving resume/export/distributed restrictions; verify lifecycle order, frozen launch configuration, local checkpoint inputs, and Ray dispatch/guard errors without creating a cluster.
- [x] 3.3 Add the smallest local callback/session observation seam needed for real scalar metrics, lifecycle, logs, and terminal results; verify event values against the executed CPU fixture and prove no progress-bar parsing or duplicated fit/test loop is used.
- [x] 3.4 Record sequenced observations, launch configuration, terminal results, and artifact references under the existing resolved logs root; implement REST status/list/result and authenticated WebSocket replay with bounded queues, and verify reconnect/no-relaunch, replay gaps, `NEXUML_LOGS_ROOT`, and restart interruption/ownership-unknown handling.
- [x] 3.5 Implement confirmed local process-tree cancellation and honest unsupported remote cancellation, including owned-service shutdown behavior; verify child cleanup, grace/escalation paths, terminal state, and that driver exit is never treated as proof of remote worker cancellation.
- [x] 3.6 Expose existing checkpoint/result/export artifacts and supported export actions from explicit trained sources through authorized paths; verify returned evaluation values, absent-artifact handling, package export/reload, and unsupported format/backend diagnostics without exporting placeholder weights.
- [x] 3.7 Document build vs field checks, operation observation/reconnect, single-operation limits, cancellation/checkpoint semantics, backend telemetry limits, and export behavior; verify REST/WebSocket examples against the CPU fixture and mock Ray boundary.

## 4. Corporate frontend shell and installation selection

- [x] 4.1 Create `studio/` with stable compatible Next.js/TypeScript, Tailwind v4, selected shadcn/Base UI controls, React Flow, Zustand, TanStack Query, and committed lockfile/scripts; verify dependency installation, typecheck, lint, focused test harness, and production build.
- [x] 4.2 Implement corporate tokens, bundled Montserrat/Geist fonts, radii/spacing, and scenario-centered navigation with runtime identity display; verify `#101010` background, black-on-`#188FD5` actions, readable contrast/focus states, and zero external font requests in browser checks.
- [x] 4.3 Implement the npm launcher with explicit NexuML-executable/project-interpreter/API-attach modes, PATH fallback, readiness/error reporting, loopback services, and ownership-scoped cleanup; verify fake executable/interpreter integration, paths containing spaces, missing dependencies, attach shutdown, and that no package installation/sync occurs.
- [x] 4.4 Implement noncached protected runtime bootstrap, authenticated REST client, WebSocket first-frame authentication/reconnect, and interface checks; verify actual API connectivity, redaction, cross-origin bootstrap rejection, and no server-shared Zustand draft or build-time secret exposure.
- [x] 4.5 Document launcher options, uv-tool isolation, project environment selection, API-extra requirements, and working-directory semantics; verify the documented explicit selection paths against an installed test runtime.

## 5. Scenario graph and configuration editing

- [x] 5.1 Add existing scenario/configuration selection and actual library list/add/remove controls; verify browser workflows against real discovery/config endpoints without a Studio project entity, hardcoded production catalog, or fabricated templates.
- [x] 5.2 Implement the ordered configuration-to-graph projection for data, stages/layers, objective relationships, and evaluation, including named key ports and visual groups; verify stage/layer order, input aliases, label/metadata routing, repeated-key producers, evaluation placement, and node movement without semantic changes in focused projection tests.
- [x] 5.3 Add component insertion/removal, connection editing, explicit ordered-outline reordering, simple arrangement, fit/minimap, and drag alternatives; verify keyboard/single-pointer edit flows and that no arbitrary DAG scheduler or executable subgraph is introduced.
- [x] 5.4 Implement schema-backed inspectors with basic/advanced fields, structured factory/complex-value fallback, and field-addressable backend errors; verify installed custom definitions and arbitrary factory kwargs without fabricated schemas or dropped fields.
- [x] 5.5 Synchronize graph/inspectors/YAML through one semantic draft with undo/redo and unapplied invalid-YAML recovery; verify valid unrendered fields, unknown definitions, input errors, and undo/redo preserve user data and the last saved config.
- [x] 5.6 Add revision-checked saves, stable editor-placement mappings, and separate revision-associated layout metadata; verify save/reopen, external conflicts, layer reorder, and stale-sidecar fallback while saved YAML remains usable by the existing CLI.
- [x] 5.7 Add responsive collapsible/resizable panels, focusable diagnostics, accessible routing/outline controls, and reduced-motion behavior; verify browser workflows at desktop/tablet/mobile widths, complete keyboard operation, and no document-level horizontal overflow, then document graph order/routing and YAML fallback behavior.

## 6. Training and execution interface

- [x] 6.1 Implement existing data/training/loader/execution/checkpoint/logging configuration views without new execution concepts; verify field updates preserve all other settings and unavailable dependencies/backends remain clearly diagnosed rather than silently replaced.
- [x] 6.2 Implement distinct configuration validation and explicit build check controls with revision-associated shapes/diagnostics; verify invalid properties, build failure/timeout, and stale-shape/readiness handling after upstream edits.
- [x] 6.3 Implement launch review and submission of a frozen configuration, then execution status/metrics/logs with real backend limits; verify CPU training start, draft edits during training, authenticated reconnect/replay, stale connections, and no duplicate submissions or fabricated telemetry.
- [x] 6.4 Implement supported stop/checkpoint-resume, completed results, launch-config inspection, and artifact/export controls; verify local cancellation, unsupported remote controls, actual results, and existing checkpoint/package export behavior in browser/API integration tests.
- [x] 6.5 Document the complete configure/check/train/inspect workflow, current backend limits, and absence of hosted/browser-training features; verify the tutorial with an installed CPU-only synthetic library scenario rather than mocked production responses.

## 7. Packaged delivery

- [x] 7.1 Configure the npm package/bin and production-asset build so startup does not build source or depend on checkout/cwd, with writable runtime/cache data outside packaged assets; verify `npm pack` contents exclude secrets/dev trees and an install into a clean directory serves assets/fonts offline.
- [ ] 7.2 Add packaged-launch smoke coverage for Linux, macOS, and Windows, including executable/interpreter modes, paths with spaces, REST/WebSocket connection, and owned/attached shutdown; verify the package-installed test matrix rather than only development-server tests.
- [x] 7.3 Verify Python wheel/sdist include the API but not frontend dependencies/build assets, and add uv-tool/project-install smoke checks using the built Python distribution; verify optional extras, actual runtime identity/discovery, and existing core-only CLI commands.
- [x] 7.4 Document release-accurate Python/npm installation, supported Node/Python/OS requirements, PATH guidance, offline asset behavior, and troubleshooting; verify every documented normal-user launch without a repository clone and prepare release artifacts without publishing them automatically.

## 8. End-to-end acceptance

- [x] 8.1 Run the packaged CPU workflow end to end: installed discovery → config/graph edit → save/reload → field check → explicit build → training → browser reconnect → results/checkpoint/package export; verify equivalent CLI configuration, real observed metrics, retained artifacts, and unchanged environment/package files.
- [ ] 8.2 Run frontend type/lint/tests/build/package checks and the relevant existing Python CLI/config/discovery/training/export regressions; verify all capability scenarios have evidence, inspect corporate desktop/mobile screenshots and accessibility behavior, and confirm no unrequested hosted, browser-training, or parallel ML framework code was added.
