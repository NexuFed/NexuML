# Design

## Context

See `proposal.md` for motivation and the three capability specs for acceptance behavior. The user explicitly requires a thin NexuML UI, REST and WebSockets through FastAPI, existing Python installations including uv tools, and corporate styling rather than a replacement design system.

Relevant existing boundaries:

- `src/nexuml/core/types.py`: `ScenarioSpec` owns data, ordered pipeline stages/layers, training, evaluation, logging, checkpoint, exports, and local/Ray execution. There is no general-purpose graph scheduler or required project entity.
- `src/nexuml/core/registry.py`, `scenario_registry.py`, and `discovery.py`: typed components/scenario factories and installed entry-point/configured-root library discovery. `LibraryConfig` already owns the persistent local roots.
- `src/nexuml/core/serialization.py` and `config.py`: registered components persist as `type`, `version`, and `params`; ordinary `model_dump()` is not a substitute for this boundary.
- `src/nexuml/core/compiler.py`: compilation follows stage/layer order and materializes components to run dummy forwards for shapes. Discovery and Python scenario resolution also import trusted code.
- `src/nexuml/training/lightning.py`: `NexuSession.run()` owns fit, validation, post-train fitting, and test; checkpoint restore and logging already exist.
- `src/nexuml/cli/main.py`: current resolve/build/train/export/library/backend operations. `src/nexuml/execution/ray.py` owns Ray behavior, including explicit distributed-semantic guards.
- `src/nexuml/core/log_paths.py`: preserve `NEXUML_LOGS_ROOT` behavior. HTTP file permissions must not redefine the framework's dataset/log path rules.

There is currently no frontend package, HTTP/WebSocket interface, or durable API observation mechanism. Any operation handles/event recording added here are transport bookkeeping, not new NexuML domain entities.

## Goals / Non-Goals

**Goals:**

- Make the same typed configuration usable from CLI, Python, and a visual editor without semantic translation into a new graph format.
- Keep all ML behavior and validation on the selected installed NexuML side; let Next.js own only presentation, draft editing, layout, and connection handling.
- Deliver one complete local workflow: choose an existing scenario/configuration, edit, check, train, observe, and inspect/export available results.
- Isolate executable work from the API event loop and retain enough observation data to reconnect and inspect finished invocations.

**Non-Goals:**

- No Studio project database, component registry, execution-provider abstraction, scheduler, training loop, or package/environment manager.
- No arbitrary DAG scheduling, executable nested subgraphs, code regeneration from YAML, or universal Python constructor introspection.
- No managed clusters, SaaS authentication/authorization, payment, Supabase deployment, collaboration, browser training, or tuning UI in this change.
- No promise of observing pre-existing CLI training processes, recovering training automatically after server restart, or stopping remote Ray workers without existing provider support.

## Decisions

### D1. Split presentation from the installed framework

Add the frontend as `studio/` and the optional Python transport as `src/nexuml/api/`. Both remain in this repository, but ship independently through npm and the Python distribution.

```text
npm launcher                       selected existing NexuML installation
    │                                          │
    ├── starts Next.js production UI           └── nexuml serve
    │                                                │
    └── opens browser                                ▼
             │                                 FastAPI, loopback
             ├── REST: settings/actions/status ──────┤
             └── WebSocket: observed events ─────────┤
                                                     ▼
                                             existing NexuML calls
                                             in owned subprocesses
```

The browser calls FastAPI directly on a separate loopback port. Next.js supplies protected runtime connection bootstrap information but does not proxy ML operations or implement a second API. Explicit origin restrictions cover both transports. This avoids relying on WebSocket upgrade proxying through ordinary Next.js Route Handlers or requiring a custom Next.js server.

**Alternative rejected:** a Next.js-owned training service or a single custom Node server that duplicates framework behavior. The Python installation already owns the dependencies and domain APIs.

### D2. Select an existing installation, never provision one

Add the proposed CLI command `nexuml serve` with working-directory, loopback port, allowed-origin, and private-token configuration. The command imports API dependencies lazily.

Add `nexuml[api]` with FastAPI, Uvicorn, a supported WebSocket implementation, and psutil for cross-platform descendant ownership/exit confirmation; include it in the existing aggregate `all` extra without making it mandatory for core installs. Server dependencies must be installed by the user into the same environment as NexuML.

Proposed user-facing launch contract:

| Mode | Explicit selection | How the launcher starts Python |
|---|---|---|
| uv tool/global executable | `nexuml-studio --nexuml /path/to/nexuml .` | Invoke that executable's new `serve` command directly |
| Existing project environment | `nexuml-studio --python /path/to/.venv/bin/python .` | Invoke that interpreter with `-m nexuml.cli.main serve` |
| Existing local API | `nexuml-studio --api http://127.0.0.1:8000 --api-token-file /private/path .` | Verify and attach; do not own or stop its process |

When no explicit selection is supplied, use a `nexuml` executable on PATH; display the resolved installation in the UI. Do not silently prefer or merge a detected project environment. Do not invoke `uv run` without no-sync/no-install protections; the initial launcher needs no uv subprocess at all. Use argument-array process spawning, not shell command interpolation.

Installation examples for the future release:

```text
uv tool install "nexuml[api]"
uv tool install "nexuml[api,library]"       # when the base library is wanted
uv tool install --with my-library "nexuml[api]"
uv add "nexuml[api,library]"               # existing uv-managed project
uv pip install "nexuml[api,library]"       # existing non-project environment
npm install -g @nexufed/nexuml-studio
nexuml-studio .
```

These are proposed interfaces, not commands currently supplied by this branch. Installation documentation must use the actual release's package index/version constraints; it must not imply that an unpublished package or PyPI release already exists. Windows examples use the corresponding executable paths.

Missing NexuML/API dependencies or unsupported interface versions produce explicit setup errors. A runtime handshake reports the actual NexuML version, Python executable, working directory, and interface major version. It is a compatibility check, not a generic plugin/negotiation framework.

**Alternative rejected:** npm bootstrapping Python or locating uv internals and importing from their directories. Executable/interpreter selection already handles uv isolation correctly.

### D3. Expose existing operations through a small REST surface

The API accepts portable lowered configurations and uses `restore_model_data`, the existing typed models, and `ResolvedConfig` for validation/serialization. REST response models describe transport results; they do not replace the existing configuration.

Proposed resource groups:

| Resource | Behavior |
|---|---|
| `/api/v1/runtime` | Runtime identity, interface version, actual backend support/dependency diagnostics |
| `/api/v1/libraries` | Existing library sources; add/remove normalized local roots |
| `/api/v1/registry` | Component identity/schema/provenance and discovery diagnostics |
| `/api/v1/scenarios` | Existing scenario names; explicit resolution of a selected factory/config/trusted local file |
| `/api/v1/config/*` | Load, validate, serialize, and revision-checked save of ordinary NexuML configurations |
| `/api/v1/build`, `/api/v1/train` | Start an existing executable operation and return its observation handle |
| `/api/v1/operations/*` | List/status/result, supported stop/resume requests, launch configuration, events and available artifacts |
| `/api/v1/export` | Existing supported export call from an explicit completed source/checkpoint, never untrained placeholder weights |
| `/api/v1/operations/{id}/events` | WebSocket observation and reconnect from an event sequence |

Use field-addressable error locations plus stable transport error categories for validation, missing dependencies, conflicts, busy, timeout, and execution failures. Do not expose a REST endpoint for arbitrary shell commands or Python snippets.

Where an existing CLI branch contains necessary shared behavior not yet exposed as a callable helper, extract only that branch's call logic into the nearest responsible NexuML module and have both interfaces use it. Do not wrap Typer handlers or parse Rich terminal output as an API. Preserve local/Ray dispatch, checkpoint resume restrictions, provenance, and export limitations.

Read component forms from each registry definition's JSON schema. Plain base-model OpenAPI schemas alone cannot enumerate every dynamically discovered definition. Return per-definition schemas and reference metadata explicitly. For generic `NnModule`/factory kwargs, offer structured values, not speculative constructor schemas.

Catalog reads use freshly instantiated discovery in a short-lived selected-runtime process, so configured local-root edits do not depend on a persistent process-wide registry or a new cache. A later UI reread uses the ordinary existing discovery path. Configuration checks can import definitions but never implicitly compile.

**Alternative rejected:** a Studio-specific catalog/schema system, stale serialized discovery cache, or reimplementation of the existing library CLI policies.

### D4. Observe calls without building another execution system

Run executable build/training/export work in subprocesses launched from the API's `sys.executable`, with the selected working directory and normal NexuML environment. The parent holds the invocation handle and publishes observations while continuing to serve lightweight requests.

Allow one resource-consuming executable operation per API instance. Reject a second with HTTP 409/busy; do not add a queue, distributed scheduler, or concurrency framework. This is an intentional v1 resource ceiling; annotate the implemented guard with a `ponytail:` comment stating that concurrent resource allocation is the upgrade boundary.

For local execution, call `NexuSession` and inject a small observation callback through the existing callback construction seam. Add only the narrow session observation seam needed if public callbacks cannot represent lifecycle boundaries; do not rewrite the sequence in `run()`. Use existing scalar results/loggers and process output for logs, not parsing progress bars. Convert supported tensor scalars before transport; do not serialize live modules/tensors into JSON.

For configured Ray execution, call the existing `run_ray` boundary. Driver logs, final results, and existing backend diagnostics are required; live worker metrics are displayed only when genuinely available. Do not introduce a worker telemetry platform or bypass the current stateful-evaluation/post-train-fit guards.

The operation handle is a correlation ID for an API-started existing invocation. Store a frozen ordinary configuration, a small observation/status sidecar, and sequenced JSONL observations under the existing resolved logs root (for example `.experiments/.studio/operations/<id>/`). Use these files to list API-observed invocations, replay events, and inspect terminal results; no SQLite project/run catalog is needed. Preserve existing logger/checkpoint/export destinations and `NEXUML_LOGS_ROOT` rather than moving domain artifacts into a new format.

An event envelope contains invocation ID, sequence, timestamp, kind, and payload. Kinds cover real lifecycle progress, scalar metrics, log messages, artifact references, error, and terminal result. Record bounded-size events before broadcast, bound in-memory subscriber queues, throttle repetitive progress, and report replay gaps/backpressure explicitly. Validate/whitelist artifact roots before exposing downloads.

Disconnecting the browser only detaches observation. On server restart, existing terminal records remain inspectable. Recorded nonterminal invocations become interrupted/ownership-unknown; do not infer live ownership from a reused PID or claim automatic recovery. Resuming uses existing checkpoint semantics as a new invocation, with a clear source reference.

Local stop interrupts the owned process group/tree, waits a bounded grace period, then terminates its owned descendants if necessary. Confirm stop before publishing cancellation. Do not promise a new checkpoint. Do not offer Ray remote stop until the existing provider boundary supplies actual cancellation; killing its driver is not evidence that remote training ended. Report unknown remote ownership after lost control/shutdown.

**Alternative rejected:** running training in an async HTTP handler or FastAPI background task, writing another fit/test loop, promising universal cancellation, or adding a provider-neutral job service.

### D5. Protect the local boundary explicitly

Both Next.js and FastAPI bind to loopback. The launcher generates a cryptographically random session token and passes it privately to owned child processes; it never includes the token in CLI arguments, URL parameters, `.env` files in the working directory, logs, or compiled assets. Manual/API-attach startup accepts a user-supplied environment/private-file token.

Next.js's noncached, same-origin bootstrap route supplies runtime connection information to its own browser session after Host/Origin/Fetch-Metadata checks. Nothing secret uses build-time `NEXT_PUBLIC_*` variables. Browser REST requests authenticate with a bearer token held in memory. Allow only the actual Studio origin for API CORS; no wildcard origins or local-network exposure.

For WebSockets, check Host and Origin before accepting the upgrade, then require the session token in a short-deadline first authentication frame before subscribing or sending any invocation data. Reject invalid/missing authentication, impose frame-size limits, and never log that frame. An initial frame avoids leaking credentials in WebSocket URLs and accounts for browser WebSocket clients not supporting arbitrary Authorization headers. Keep state-changing actions in authenticated REST.

Transport filesystem operations resolve and check the working-directory/authorized-artifact-root boundary, reject symlink escapes, and use revision-checked atomic writes for configuration. Framework data sources can still refer to trusted explicit dataset paths, and existing library-root management can still authorize roots elsewhere; neither grants unrestricted HTTP filesystem browsing.

Local discovery, scenario factories, generic imports, and build checks run trusted code. State that boundary in setup/help and require explicit resolution/build actions. Process separation supplies cancellation and fault isolation, not a sandbox for hostile Python. Hosted isolation is a different future change.

**Alternative rejected:** treating localhost as authentication, wildcard CORS, passing reusable credentials in query strings, or claiming a subprocess makes customer Python safe.

### D6. Project the ordered specification, not a new DAG

React Flow is the renderer and interaction layer. The editable semantic document remains NexuML's lowered configuration; do not save React Flow nodes/edges as the executable pipeline.

Projection rules:

- Data nodes refer to `data.source`/`data.datasets`; their inspectors edit existing source, target, split, loader, preprocessing, and input-shape fields.
- Stage groups and layer nodes refer to the ordered `pipeline.stages` placements. Group collapse is visual, not executable composition.
- Loss/metric relationships refer to existing pipeline key outputs and `training.loss_keys`/`metric_keys`. Do not invent loss nodes unsupported by the installed catalog.
- Evaluation nodes refer to `evaluation.algorithms` and their existing feature/label/axis contracts, not fabricated layer placements.
- Ports represent existing keys. Edges refer to the applicable preceding producer in execution order; preserve input alias dictionaries, label routing, and overwrites. When routing is not fully renderable, retain editable key/metadata fields and show the limitation rather than inventing connectivity.
- Moving a node only edits layout. Explicit ordered-outline controls edit stage/layer order; invalid forward references remain diagnosable and never become implicitly topologically sorted.

Stage insertion order is semantic. Return an ordered stage-name view with configuration transport metadata, render from that order rather than JavaScript object enumeration, and restore the mapping in that order before validating/saving. Test integer-like stage names as well as ordinary names. This is representation of existing order, not a new execution specification.

Each open editor keeps stable local selection IDs mapped to existing placements. Save only placement/layout information in a sibling `*.studio.json` sidecar, associated with a semantic configuration revision. Update placement mappings on explicit edits; ignore stale layout metadata after unmatched external edits and fall back to a simple stage-ordered layout. Do not insert editor IDs into `LayerSpec` or misuse persistent array indices after a reorder.

Keep one draft shared by graph/inspectors/YAML. Preserve all valid unrendered fields. Invalid YAML remains a separate unapplied buffer until corrected; graph edits are unavailable while applying them would discard that buffer. Matching semantic hashes bind build checks to their source revision. Serialization preserves semantics, not original YAML comments/formatting or Python source code.

Start with deterministic stage-column arrangement, fit view, and a minimap. Avoid an automatic layout dependency unless real branching examples demonstrate that simple layout is insufficient.

**Alternative rejected:** a new graph IR/topological compiler or executable subflow framework. Neither is required to configure the current NexuML model.

### D7. Build a focused, accessible workbench

Navigation follows NexuML concepts: choose a discovered scenario or existing config, then use **Pipeline**, **Training**, **Execution**, and **Artifacts** views. Library/environment controls are discovery/settings views, not separate Studio entities. A directory breadcrumb is context only.

```text
┌──────────────────────────────────────────────────────────────────────┐
│ NexuML Studio / working directory / scenario    Saved   Check   Run… │
├──────────────────────────────────────────────────────────────────────┤
│ Pipeline     Training     Execution     Artifacts              YAML │
├──────────────┬───────────────────────────────────┬───────────────────┤
│ Components   │                                   │ Properties        │
│ Search…      │ Source → Stage → Model             │ Basic / Advanced  │
│              │                 ├→ Objective      │ Key routing       │
│ Discovered   │                 └→ Evaluation     │ Build diagnostics │
│ libraries    │                                   │                   │
│              │ Zoom / Fit / Arrange / Outline    │                   │
├──────────────┴───────────────────────────────────┴───────────────────┤
│ Problems · check revision · selected NexuML runtime · connection     │
└──────────────────────────────────────────────────────────────────────┘
```

The screenshot supplied in the discussion is the composition baseline; shipping must not depend on Stitch authentication or externally hosted assets.

Interaction decisions:

- Begin with discovered scenario recipes or imported config, not mandatory blank-canvas authoring.
- Keep detailed properties in the right inspector; node cards show name, relevant parameter summary, named ports, available shapes, and explicit diagnostic state.
- Put epochs, optimizer, scheduler, batch/loader overrides, precision, devices, callbacks/logging, checkpoint, and existing execution settings in Training, not disconnected canvas nodes.
- Build is an explicit executable check; schema errors appear beside fields and in a focusable Problems list. Unknown/stale shapes remain labeled; no decorative readiness claims.
- Run review displays the exact source revision, existing execution selection, training settings, and output references. Launch switches to actual observation without mutating the draft.
- Execution shows observed lifecycle, scalar curves, logs, results, and available controls. Plot only recorded metrics, identify stale connections/replay gaps, and do not animate fake tensor flow. A lightweight plotting component is sufficient; no GPU plotting requirement.
- Add/connect/reorder through visible menus/controls and an ordered outline as well as pointer dragging. Maintain undo/redo, visible focus, accessible names, understandable tab order, and reduced motion.
- Desktop panels resize/collapse without covering focus. Below the supplied 1280px desktop breakpoint, switch to tabbed/single-panel canvas/outline/inspector presentation; mobile prioritizes monitoring and configuration forms. Canvas panning is intentional; document-level horizontal overflow is not.

**Alternative rejected:** an overview dashboard as the main entry, putting every scalar setting on the canvas, or requiring precision dragging to complete a workflow.

### D8. Use the approved corporate tokens and a modern stable stack

Use the latest compatible stable versions at implementation time, commit the lockfile, and keep the existing Python dependency constraints intact. No canary dependency is necessary.

| Concern | Choice and boundary |
|---|---|
| Application | Next.js App Router and TypeScript; interactive workbench as a client boundary, not the entire server shell |
| Styling | Tailwind CSS v4 semantic theme variables; no default palette replacement |
| Controls | shadcn/ui with Base UI primitives; copy only required components and restyle them |
| Graph | `@xyflow/react`; custom corporate nodes, application-owned routing rules |
| Client draft | Zustand per editor/provider, narrow subscriptions; no shared server-global store |
| Server data | TanStack Query for catalog/status/results, separate from draft state; update from authenticated WebSocket events |
| Python transport | FastAPI/Uvicorn/WebSockets in the optional `api` extra |
| Checks | Existing pytest conventions; frontend type/lint/build checks and focused component/browser tests |

Do not add Auth.js, Supabase, Three.js, WebGL, WebGPU, a PWA, an animation framework, or a generic schema-form framework for this slice. Use ordinary controlled inputs for supported schema fields and structured/YAML fallback for complex values.

The user's correction overrides the attachment's conflicting background prose/tokens. Keep the approved action pair separate from the tonal `primary` token:

| Token/role | Value |
|---|---|
| Background / surface / surface-dim | `#101010` |
| Surface lowest | `#0c0e12` |
| Surface low / ordinary panels | `#1a1c1f` |
| Surface container | `#1e2024` |
| Surface high | `#282a2e` |
| Surface highest / variant | `#333539` |
| Surface bright | `#38393d` |
| On surface / on background | `#e2e2e7` |
| On surface variant | `#c3c6d1` |
| Outline / outline variant | `#8d919b` / `#434750` |
| Action / on action | `#188FD5` / `#000000` |
| Primary / on primary | `#a7c8ff` / `#003061` |
| Primary container / on primary container | `#0a3d74` / `#83a9e7` |
| Secondary / on secondary | `#93ccff` / `#003351` |
| Secondary container / on secondary container | `#2a98de` / `#002c47` |
| Tertiary / on tertiary | `#44dfab` / `#003827` |
| Tertiary container / on tertiary container | `#004632` / `#00be8d` |
| Error / on error | `#ffb4ab` / `#690005` |
| Error container / on error container | `#93000a` / `#ffdad6` |

Use existing primary/tertiary/error tones for selection, successful checks, and errors, with text/icons. Preserve the attachment's inverse/fixed token families where needed rather than inventing additional accents. White-on-action is not permitted for ordinary button labels: black-on-`#188FD5` is approximately 5.93:1, whereas white is approximately 3.54:1. Check hover, selected, focus, and muted text separately.

Typography follows the supplied scale: Montserrat headline XL 48/56 at 700, LG 32/40 at 600 (mobile 24/32), MD 24/32 at 600; Geist body LG 18/28, MD 16/24, SM 14/20; labels 14/16 at 600; technical/code text 13/18 at 500. Use tabular figures for changing numeric values. Bundle required font files so normal installed/offline use does not fetch Google Fonts.

Preserve the 4px spacing baseline, 24px gutters, 16px mobile margins, 4px control radius, 8px card radius, and 12px modal radius. Use the supplied 768px/1280px breakpoints. The editor intentionally fills the viewport; 1440px content limits apply to noncanvas views. Avoid pill-shaped primary actions, heavy shadows, animated edge decoration, or large glass surfaces; blur is reserved for overlays.

**Alternative rejected:** importing a generated dashboard palette or choosing experimental graphics libraries as evidence of modernity.

### D9. Package production assets and prove the published path

The npm package `@nexufed/nexuml-studio` supplies the `nexuml-studio` launcher, compiled production Next.js assets, local fonts, and required runtime dependencies. Resolve assets relative to the package, not the user's cwd; pass the selected working directory to NexuML explicitly. Store mutable application runtime/cache data outside immutable package assets.

Build once during release, not on user startup. Prefer normal prebuilt Next.js production output plus npm-installed runtime dependencies over copying a Linux-built standalone dependency tree into a supposedly cross-platform package. If standalone output is selected during implementation, build/package its platform-dependent assets correctly and explicitly include `public` and `.next/static`; the output mode alone is not a portability guarantee.

Exclude source-checkout secrets, test fixtures, Python environments, and development dependency trees from npm artifacts. Keep Python wheel/sdist builds unchanged except for the API modules/extra. Existing frontend-free CLI commands remain lazily independent of FastAPI.

Add scripts for frontend typecheck, lint, focused tests, browser tests, build, and package smoke validation. Validate `npm pack` output installed into a clean directory with both a uv-tool runtime and an existing project runtime. Run launcher/process/REST/WebSocket/asset checks on Linux, macOS, and Windows. A CPU-only synthetic installed-library scenario is the end-to-end training acceptance case; real cluster/GPU tests are optional and must not be silently started by ordinary CI.

**Alternative rejected:** shipping a development checkout, asking users to build Next.js, assuming platform-specific traced dependencies work everywhere, or claiming success from `npm run dev` alone.

## Risks / Trade-offs

- **Ordered pipeline vs arbitrary graph expectations** → State execution order in the outline and preserve existing key overwrites; no implicit DAG semantics.
- **Arbitrary schemas and factory kwargs** → Support common field types with explicit structured/YAML fallback and backend validation; preserve unrendered fields.
- **Trusted Python import/build side effects** → Explicit trusted-code boundary, subprocess isolation/timeouts, and no sandbox claims.
- **uv tool isolation or mismatched installations** → Explicit executable/interpreter selection and runtime identity handshake; no package installation by the launcher.
- **Frontend/API release skew** → Small interface-major handshake and actionable mismatch diagnostics; no compatibility aliases or migration framework.
- **Live callback coverage, especially Ray** → Show only observed data and expose limited telemetry/cancellation rather than duplicating backend behavior.
- **Server failure and remote ownership uncertainty** → Persist terminal observations; classify unknown active work honestly and never equate driver exit with remote cancellation.
- **Large metrics/log streams** → Throttle progress, bound individual events/subscriber memory, replay from disk, and batch/limit visible chart/log updates without dropping source data silently.
- **Brand attachment inconsistencies** → The user's corrected background/action pair is authoritative; other supplied corporate tokens are retained with explicit contrast checks.
- **Existing in-flight release/config/Ray work** → Additive API/frontend integration only, tests around any narrow shared call extraction, and no edits to other OpenSpec changes.

## Migration Plan

1. Add the optional API extra and lazy CLI server path; verify core-only installations still work.
2. Implement REST parity/security and observation around the current installed framework APIs, with focused tests before frontend integration.
3. Implement the corporate workbench against the real local API; complete the CPU workflow before package release.
4. Package and validate the production npm artifact and both installation modes on supported operating systems, then document release-accurate installation commands.
5. Existing configs, libraries, and CLI workflows need no migration. Rollback means stopping/removing the new frontend/API extra or returning to the existing CLI. Layout/observation sidecars are separate from executable configs and can be ignored without data conversion.
