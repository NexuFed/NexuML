# Proposal

## Why

NexuML already owns scenario configuration, component libraries, pipeline compilation, evaluation, and execution, but users must operate those capabilities through Python or the CLI. A corporate-design local Studio should make the same capabilities discoverable and editable without creating a second ML framework or requiring a hosted account.

## What Changes

- Add an npm-distributed Next.js Studio with a node-based scenario editor, schema-backed inspectors, training configuration, and execution/results views.
- Keep NexuML's existing definitions, registries, ordered pipeline stages, TensorDict routing, serialization, and execution backends authoritative. Library controls expose existing library sources and operations; they do not create Studio-specific libraries, projects, or execution providers.
- Add a thin FastAPI interface to NexuML: REST for existing operations and WebSockets for their progress, metrics, logs, and completion. Long-running calls execute outside the HTTP event loop and reuse existing NexuML lifecycles.
- Add a NexuML CLI server entrypoint. Studio uses an explicitly selected existing NexuML installation, including an isolated `uv tool install` environment or a project environment; npm never installs or manages Python/NexuML.
- Make FastAPI/server dependencies an optional `api` extra, with explicit installation guidance and a clear error when the selected installation lacks them.
- Apply the supplied Nexus Edge design: `#101010` background, Montserrat/Geist typography, restrained tonal surfaces, and primary actions using `#188FD5` with black text.
- Support lossless configuration editing, explicit build checks, execution observation across browser reconnects, supported cancellation, and inspection of existing result/artifact outputs.
- Refine the same workbench into graph-first authoring: categorized drag/drop components, removable key connections, typed nested forms, field-focused errors, native observed progress and in-place terminal rendering; retain JSON/YAML as explicit expert tools.
- Keep v1 local and single-user. Hosted identities/rights, payment, Kubernetes provisioning, browser ONNX training, collaboration, arbitrary DAG execution, and a tuning UI are non-goals. Existing configured execution backends remain NexuML-owned; no cluster management is added.

## Capabilities

### New Capabilities

- `nexuml-local-api`: Local REST/WebSocket transport over existing NexuML discovery, configuration, compilation, execution, and artifact behavior, with authenticated access and lightweight operation observation.
- `studio-workbench`: Corporate-design node editor and training/results interface that preserves NexuML configuration semantics and accessible interaction.
- `studio-local-distribution`: npm installation/launch of Studio against user-installed NexuML, including project and uv-tool environments without a source checkout or automatic Python provisioning.

### Modified Capabilities

None. Existing library discovery/management, scenario semantics, and execution requirements remain unchanged; the new interfaces consume them.

## Impact

- Python: new `src/nexuml/api/` transport code, a lazy CLI server command, optional dependency metadata, and narrowly scoped observation hooks/shared call helpers only where existing Python APIs cannot yet serve both CLI and HTTP callers.
- Frontend: a new `studio/` Next.js/TypeScript npm package using Tailwind CSS v4, shadcn/ui with Base UI, React Flow, Zustand, and TanStack Query. UI libraries are implementation aids, not replacements for NexuML concepts.
- Delivery: npm package build/pack validation, supported-OS installation smoke checks, focused backend/browser tests, and installation/API/Studio documentation. Python wheel/sdist packaging remains independent of frontend assets.
- Existing in-flight typed-config, release, and Ray work must be respected; this change uses the current typed definitions and existing Ray limitations rather than reverting or redesigning them.
