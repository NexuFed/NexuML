# NexuML Studio (unreleased)

Local, single-user Next.js workbench over **your installed NexuML**. Ordinary NexuML
YAML remains the execution configuration; React Flow is only its ordered editor view.
No accounts, payments, Python provisioning, cluster creation, or browser training.

## Install built artifacts

These packages are not yet published. Use the wheel and npm tarball built from this
change; do not assume the public registry versions already contain Studio/API:

```sh
uv tool install '/path/to/nexuml-0.2.1-py3-none-any.whl[api]'
# Install the separately built base/custom library into that same selected tool environment.
npm install -g /path/to/nexufed-nexuml-studio-0.1.0.tgz
nexuml-studio --nexuml /path/to/nexuml /path/to/work
```

After releases containing this interface are published, normal package-name installation
will be `uv tool install 'nexuml[api,library]'` and
`npm install -g @nexufed/nexuml-studio`. Core commands do not require the API extra.

Requirements: Node >=22.13, Python >=3.12 supported by the installed NexuML, and a
modern browser. The launcher supports Linux/macOS/Windows; cross-platform smoke coverage
is included, but a local Linux pass is not proof that unrun macOS/Windows jobs passed.

## Choose an existing runtime

```sh
nexuml-studio .                             # existing nexuml on PATH
nexuml-studio --nexuml '/path to/nexuml' .   # uv tool/global executable directly
nexuml-studio --python '/project/.venv/bin/python' '/working directory'
```

Windows: `nexuml-studio --python 'C:\project\.venv\Scripts\python.exe' 'C:\work'`.
Paths with spaces are argument-array spawned, not interpreted as shell commands.
An explicit Python interpreter invokes `-m nexuml.cli.main serve`; a uv-tool executable
is invoked directly. Do not select an unrelated shell Python for an isolated uv tool.
Custom libraries and the `api` extra must exist in **that installation**.

The directory is filesystem context, not a Studio project. Both services bind to
`127.0.0.1`; random unused ports are selected by default. Set `--port 3000 --api-port 8000`
to choose explicit ports. `--no-open` suppresses automatic browser opening. Runtime
identity is displayed, and an interface mismatch is an error, not a silent fallback.

Attach to an already running API with its private token file and exact configured
Studio origin:

```sh
nexuml-studio --port 3000 --api http://127.0.0.1:8000 --api-token-file /private/token .
```

The attached API must use this directory and allow `http://127.0.0.1:3000`. It is never
stopped by the launcher. Owned services are stopped on Ctrl-C. The token stays out of
arguments, URLs, build output and working-directory `.env` files. POSIX token files need
mode 600; on Windows restrict the ACL. Bootstrap is uncached and exact-origin/Host/
Fetch-Metadata protected; the browser keeps credentials only in memory.

No build, npm/uv/pip installation, dependency sync or internet access occurs on ordinary
startup. Prebuilt UI/assets/fonts ship in the npm package. Next's runtime/cache uses a
private temporary copy, not writable package assets. The directory is removed at clean
shutdown; after a crash, stale `nexuml-studio-*` temporary directories can be removed
after confirming their services have stopped. Training records remain under NexuML logs.

## Configure → check → train → inspect

1. Resolve an installed scenario explicitly, or open existing YAML in the selected
   working directory. Resolution imports/calls **trusted Python**, not sandboxed code.
2. Use Pipeline to inspect data, stage/layer order, key ports, objectives and evaluation.
   Browse installed components by kind/module categories (for example Pipeline layers →
   Models → Vision). Drag one onto a stage to append it to that stage, or use its add button.
   Insert from actual installed definition schemas. Use Training for optimizer/scheduler,
   loader, execution, logging/callbacks/checkpoints and exports.
3. **Structure** and the canvas **Execution order** strip control the native ordered stages
   and layers. Drag their order handles onto insertion markers, or use move buttons and
   Properties → **Move / transfer layer** for keyboard/single-pointer editing. Canvas
   coordinates never schedule execution. Drag a stage header to move it with its children;
   resize a selected stage from its lower-right control or Properties → **Container size**.
   Layers move freely inside their stage without reordering. Moving a layer into another
   highlighted stage previews an explicit append; dropping outside every stage or pressing
   Escape restores it. Transfer controls/Structure markers also allow explicit insertion slots.
   Skipped-stage previews warn that moved layers will not execute, without changing skip settings.
   Add a stage from Structure's **Stage** item (drag or click) or **Add stage**: give it a unique
   name and review insertion after the selected stage, otherwise last. Empty stages offer
   **Drop a layer here / Add layer**. New-layer drops outside stages ask for a destination;
   click insertion uses the visible destination/slot controls. Movement/resizing are layout-only;
   transfers/reorders change ordinary configuration and make previous build evidence stale.
   Repeated keys connect to their most recent preceding producer. Alias dictionaries,
   label domains and metadata routing stay editable; dataset metadata/runtime-only keys
   are diagnosed rather than fabricated into a DAG. Buttons/connection selects/position
   fields provide drag alternatives. Connect an empty + Feature/Label/Loss/Metric input, name outputs
   in Properties, or reconnect an existing edge. Select an edge and choose Remove connection
   (or Delete/Backspace); connection selects offer the same removal. Missing required routes
   remain visible and prevent launch. If removing an evaluator route would restore its runtime
   default, reroute it or explicitly disable that evaluator instead. Display names are visual
   sidecar data, never tensor names or execution settings. Use Undo/Redo for routing, names,
   completed layout gestures and atomic transfers/order edits. Cancelled gestures create no entry.
4. Schema controls handle nested models, nullable/union modes, lists, mappings, enums and
   installed component choices. Execution/checkpoint/logging settings are ordinary forms.
   Expand **Expert** for raw JSON or use the complete YAML editor; genuinely untyped factory
   kwargs retain that escape hatch and no constructor schema is invented.
   Valid unrendered fields are retained. Unapplied/invalid YAML is a retained buffer and
   blocks graph edits until Apply or deliberate Discard. Comments/formatting are not preserved.
5. Check fields validates without compiling. Build check explicitly runs constructors and
   dummy forwards in the selected interpreter. Displayed final key shapes are source-bound
   and become stale after semantic edits. Save uses revision conflicts and a separate
   `*.studio.json` position/size/name sidecar; an external semantic edit invalidates old placement metadata.
   Matching older sidecars without container sizes use defaults. Membership/order stay in ordinary YAML.
   Click a field-addressable problem to open and focus its inspector/settings field. Runtime
   errors without a field address remain linked to their frozen source, not guessed draft fields.
   Failed operations offer **Open frozen settings as draft** to edit their actual launch settings,
   including after a reload; unsaved draft replacement requires confirmation and never changes the run.
6. Run reviews the frozen source, settings and output references. Optional local Trainer
   resume reviews the checkpoint-derived scenario (trusted Python input). Confirm once;
   later draft edits do not change this run. One resource-consuming operation is allowed.
7. Execution shows native phase/epoch/batch progress, real scalar curves, readable results,
   process logs and reconnect state. Preparation/downloads without counters use indeterminate
   activity; carriage-return and supported ANSI progress updates overwrite terminal lines.
   These logs are not parsed into training counters. Raw output remains downloadable. Reload
   and select the existing operation; it is never relaunched. Visible history is bounded
   to 2,000 events, with a notice; charts separately retain 2,000 metric observations
   so verbose logs do not evict their samples. Full observations/logs remain on disk.
   Training loss updates during the first epoch from actual batch callbacks (at most
   four updates/second plus first/final batches). Charts use optimizer steps, and a
   single measurement is a visible point rather than an invisible line. Validation/test
   values appear after their phase completes, excluding the preliminary sanity check.
   Pipeline → **Objectives & metrics** edits `training.loss_keys` (weighted losses) and
   `training.metric_keys` (outputs to log). NexuML also reports their weighted loss total
   as `train/loss`, `val/loss`, and `test/loss`. Metrics must be produced by pipeline
   layers: the CIFAR ClassificationMetrics layer computes accuracy/F1 in validation/test,
   not training. Scalar evaluator results can also appear when evaluation finishes.
   Charts reflect the observed run's frozen configuration, not current draft settings.
8. Stop is available for owned local processes, not Ray workers. It confirms tree exit,
   not a new checkpoint. Artifacts downloads use authorized paths. Export requires a
   completed local source and an actual Trainer checkpoint, not random placeholder weights.

Ray delegates to existing NexuML guards/cluster connection and has driver logs/final
results only. No remote-stop, automatic recovery, hosted permissions/collaboration,
browser ONNX training or Kubernetes provisioning is implied.

## Troubleshooting

- **Executable missing:** select `--nexuml`/`--python` explicitly or correct PATH.
- **Missing API dependencies:** install the matching `api` extra yourself in the selected
  installation. The launcher will not upgrade it.
- **403/attach failure:** match `--origin`, Studio port, directory and private token.
  `localhost` and `127.0.0.1` are distinct origins.
- **409 save:** an external file changed. Reload or save to a new path; no overwrite occurs.
- **Busy:** wait for or stop the active local build/train/export. There is no queue.
- **Ownership unknown after restart:** inspect retained logs/artifacts. Resume only from
  an actual checkpoint, as a new invocation; remote worker termination is not assumed.
- **Missing custom fields:** inspect the registered schema and use structured/YAML fallback.
- **Unconnected evaluator:** installed schemas may declare parameter/default input routes and
  dataset label outputs. Dynamic grouping, dataset metadata and undeclared custom contracts
  remain runtime-only; Studio never constructs an evaluator or loads data just to draw an edge.

## Development and release checks

```sh
npm ci
npm run typecheck
npm run lint
npm test
npm run build
npm pack
```

Build/pack is a release task, never a normal-user startup task. API/Python checks and
installed-package/browser smoke instructions live in the repository's local API guide.
Do not publish either artifact automatically. Dependencies are locked; ESLint uses v9
because the stable Next plugins do not yet accept v10's peer range.
