# NexuML Studio 0.1.0 — Linux Developer Preview (unreleased)

Local, single-user Next.js workbench over **your installed NexuML**. Ordinary NexuML
YAML remains the execution configuration; React Flow is only its ordered editor view.
No accounts, payments, Python provisioning, cluster creation, or browser training.

## Linux quickstart

Prerequisites: **Python >=3.12**, **Node >=22.13**, **uv**, and a modern browser.

```sh
uv tool install "nexuml[api,library]"
npm install -g @nexufed/nexuml-studio
nexuml-studio .
```

Studio opens a local browser interface automatically. The welcome screen identifies the
connected NexuML version and Python; **Runtime details** shows the executable and working
directory. NexuML executes training in that Python environment, not in the browser.
Libraries must be installed in the **same environment**. Nothing is installed or synced at
startup; after installation, the example and local interface work offline.

To use an existing project environment instead of the tool installation:

```sh
nexuml-studio --python /path/to/project/.venv/bin/python .
```

Keep the terminal running; **Ctrl-C** stops Studio and its owned local processes. Closing
the browser does not stop training. Run outputs remain under `.experiments/` in the selected
working directory, including Studio records in `.experiments/.studio/operations/`;
`NEXUML_LOGS_ROOT` can relocate them. Runs → Artifacts shows actual output paths.
Saved YAML stays at the path you choose.

This is a **Linux developer preview**. macOS/Windows are not release-validated.
Backend and frontend releases are independent; core NexuML does not need the optional API.
The registry commands above apply once the preview is published. Until then, use the
built candidates below; public registry versions must not be assumed to contain this work.

## First training run

1. Click **Try an example**. Studio resolves the installed
   `synthetic-linear-ae-reconstruction` recipe, then validates an ordinary draft set to
   **one CPU epoch**. Data is generated locally; no dataset download or service is needed.
   Resolution calls trusted installed Python, but **does not start training**.
2. Inspect the editable graph in **Pipeline**. **Training** holds normal execution settings;
   **Save** writes portable NexuML YAML. The installed recipe's defaults are unchanged.
3. Click **Run…**, inspect Local execution, choose **Review selection**, then **Run on Local**.
4. **Runs** shows real **Metrics**, **Logs**, **Artifacts**, and the frozen **Configuration**.
   Reload and select the same operation to inspect it again without relaunching.

If the example is missing, install `nexuml-library` in the selected environment and restart
Studio, or use **Choose a recipe** / **Open YAML**. Studio never installs it for you.

### Testing unreleased artifacts

These packages are not yet published. Use the wheel and npm tarball built from this
change; do not assume the public registry versions already contain Studio/API:

```sh
uv tool install '/path/to/nexuml-0.2.1-py3-none-any.whl[api]'
# Install the separately built base/custom library into that same selected tool environment.
npm install -g /path/to/nexufed-nexuml-studio-0.1.0.tgz
nexuml-studio --nexuml /path/to/nexuml /path/to/work
```

Use a modern browser. Linux project/tool installations are checked in CI. macOS/Windows
jobs are temporarily disabled at maintainer request; portability acceptance remains open.

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

To change installations, stop Studio and select a different `--nexuml` or `--python`.
Updates are explicit user actions: for example `uv tool upgrade nexuml`, or install the
desired `nexuml[api,library]` version into your project environment. Studio never runs these.

Install a packaged custom library into the same environment: use
`uv tool install --with my-nexuml-library 'nexuml[api,library]'` when installing the tool,
or `uv pip install --python /project/.venv/bin/python my-nexuml-library` for a project.
For trusted source roots, use Libraries → Add root or the selected installation's
`nexuml library add /path/to/library`; install its dependencies yourself first.

### Compatibility

Studio **0.1.x** supports stable NexuML **>=0.2.1,<0.3.0** with **API interface 2**.
Python >=3.12 remains NexuML's requirement. Package versions and the interface version
are separate: supported backend patch releases do not require a Studio version bump.
Unknown minors, prereleases and interfaces are rejected before the workbench/catalog
opens. Errors name the selected/required version; choose or explicitly update the backend.
The launcher and runtime details report the actual NexuML version, Python version and interpreter.

The directory is filesystem context, not a Studio project. By default both services bind to
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
mode 600; on Windows restrict the ACL. Bootstrap is uncached and browser Host/Origin/
Fetch-Metadata checked; these headers are **not authentication for arbitrary HTTP clients**.
Local users/processes are trusted. The browser keeps credentials only in memory.

No build, npm/uv/pip installation, dependency sync or internet access occurs on ordinary
startup. Prebuilt UI/assets/fonts ship in the npm package. Next's runtime/cache uses a
private temporary copy, not writable package assets. Only npm's installed Next/React
runtime packages are linked, using Windows junctions where needed. Standalone output was
reviewed but traces build-host Sharp/libvips native binaries; excluding/repackaging those
would add portability assumptions rather than simplify this single npm tarball.
No Linux-built dependency tree is shipped. The directory is removed at clean
shutdown; after a crash, stale `nexuml-studio-*` temporary directories can be removed
after confirming their services have stopped. Training records remain under NexuML logs.

## Authenticated HTTPS website mode

Local/offline mode needs no proxy, account or network access. To serve the same Studio
through an existing authenticated HTTPS reverse proxy, opt in to one exact public origin
and a **separate private proxy credential** (not the Python API token):

```sh
umask 077
python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$HOME/.nexuml-studio-proxy-token"
nexuml-studio --port 3000 --public-origin https://studio.example \
  --proxy-token-file "$HOME/.nexuml-studio-proxy-token" --no-open /path/to/work
```

`NEXUML_STUDIO_PUBLIC_ORIGIN`, `NEXUML_STUDIO_HOST` and `NEXUML_STUDIO_PROXY_TOKEN_FILE`
supply the same settings; CLI flags take precedence. The UI defaults to `127.0.0.1`
**even when hosted**. A non-loopback bind requires an explicit `--host` (for example a
private interface), public origin and proxy credential, and emits a security warning.
Host must be an
IPv4 address; the public origin must be exact HTTPS, with no trailing slash, credentials,
path, query or fragment. The printed/opened address is the public URL. Local access at
`http://127.0.0.1:3000` cannot bypass the proxy credential in hosted mode.

Configure NetBird (or your existing proxy) with **one `/` target to the UI port**,
**Pass Host Header enabled**, WebSocket forwarding, HTTPS and authentication/access
restrictions. After authenticating a trusted operator, the proxy must **overwrite**
`X-NexuML-Studio-Proxy` with the private file's value on every upstream HTTP/WebSocket
request. Never accept a client-supplied value or expose this credential in browser code,
URLs, logs or responses. Configure this only on the server-side proxy. If your NetBird
version cannot inject a private upstream header, place a header-injecting proxy on its
restricted target or keep Studio local; identity/forwarded headers alone are insufficient.
Do not publish the Python API port or add an API proxy target. Firewall any non-loopback
UI port to the proxy host only; do not allow arbitrary private-network clients. Anyone admitted to
Studio can obtain its runtime credential and invoke trusted local Python. This is a
trusted-operator website, not a public or multi-tenant service. Studio does not configure
NetBird or provide a login system.

Browser HTTP and WS/WSS use `/api/v1` on the browser's own origin. The Next server forwards
to the runtime-selected loopback API, preserving browser bearer/first-frame WebSocket
authentication, replacing upstream Host/Origin with the configured loopback values and
discarding cookies and forwarded-proxy metadata. Bootstrap never discloses the API URL.
Actual Host, optional Origin and Fetch Metadata are checked; forwarded host/proto headers
cannot grant access. In hosted mode the proxy credential is required for bootstrap and
API traffic even with spoofed local/public Host/Origin headers; it is stripped before
forwarding to Python. A public origin alone is never a login or an authentication boundary.
The Python API still binds to loopback and validates its original
local origin, token and interface version **2**. Attach mode still requires the API to
allow the **local** UI origin, not the public HTTPS origin.

For development, export the runtime settings and configure the Next listen host explicitly:

```sh
export NEXUML_STUDIO_ORIGIN=http://127.0.0.1:3000
export NEXUML_STUDIO_API=http://127.0.0.1:8000
export NEXUML_STUDIO_TOKEN="$(cat /private/token)"
export NEXUML_STUDIO_PUBLIC_ORIGIN=https://studio.example
export NEXUML_STUDIO_PROXY_TOKEN="$(cat "$HOME/.nexuml-studio-proxy-token")"
export NEXUML_STUDIO_HOST=127.0.0.1
npm run dev -- --hostname "$NEXUML_STUDIO_HOST" --port 3000
```

Start the selected API separately with local `--origin http://127.0.0.1:3000`. HMR allows
only the explicit public hostname plus loopback, not a wildcard. Keep the same proxy
access restrictions in development. Public origin and API port are runtime settings;
there is no public API URL baked into the build.

## Configure → check → train → inspect

1. At entry, **Choose a recipe** or **Open YAML** in the selected working directory.
   Recipe resolution imports/calls **trusted Python**, not sandboxed code. After loading,
   the scenario-name menu contains source switching, Save and **Save as…**. First Save of
   a recipe asks for a destination. Saved means configuration and separate layout persisted;
   unsaved changes, an unsaved recipe and unapplied YAML have distinct states.
2. Use Pipeline to inspect data, stage/layer order, key ports, objectives and evaluation.
   The left panel starts on **Structure**. Switch to **Components** to search installed
   definitions by kind/module categories (for example Pipeline layers →
   Models → Vision). Drag one onto a stage to append it to that stage, or use its add button.
   Insert from actual installed definition schemas. Use Training's **Basics**, **Optimizer
   and schedule**, **Data loading**, **Execution**, and **Checkpoints and logging** sections.
   Trainer accelerator/devices are in Basics, separate from backend placement resources.
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
   click insertion uses the visible destination/slot controls, or requests a destination if
   neither one nor a stage/layer selection exists. It never silently chooses the first stage.
   Switching Structure/Components retains search. **Focus selected** centers a Structure
   selection at a readable scale; **Fit view** provides an explicit overview. **Arrange**
   explicitly resets layout; opening a matching sidecar retains its positions/sizes.
   Desktop panels have collapse buttons, draggable/keyboard separators and Panels width
   controls; narrower layouts select Structure/Components, Canvas or Properties separately.
   Movement/resizing are layout-only;
   transfers/reorders change ordinary configuration and make previous build evidence stale.
   Repeated keys connect to their most recent preceding producer. Alias dictionaries,
   label domains and metadata routing stay editable; dataset metadata/runtime-only keys
   are diagnosed rather than fabricated into a DAG. Buttons/connection selects/position
   fields provide drag alternatives. Connect an empty + Feature/Label/Loss/Metric input, name outputs
   in Properties, or reconnect an existing edge. Select an edge and choose Remove connection
   (or Delete/Backspace); connection selects offer the same removal. Missing required routes
   remain visible and prevent launch. If removing an evaluator route would restore its runtime
   default, reroute it or explicitly disable that evaluator instead. Display names are visual
   sidecar data, never tensor names or execution settings. Properties leads with **Settings**;
   **Routing** contains key lists, aliases, label/metadata routes and connection selects.
   **Edit display name** and **Advanced layout** hold visual-only controls. Optional empty
   ports open on selection or **Routing ports**; connected/required ports remain visible.
   Use Undo/Redo beside Save for routing, names,
   completed layout gestures and atomic transfers/order edits. Cancelled gestures create no entry.
4. Schema controls handle nested models, nullable/union modes, lists, mappings, enums and
   installed component choices. Execution/checkpoint/logging settings are ordinary forms.
   Required/common controls appear before **Advanced**. Nullable controls use **Not set /
   Set value**; training batch size uses **Fixed / Automatic** with actual schema fields.
   Data loading's **Use training batch size** stores null (inheritance), not a copied number;
   Basics links to an explicit loader override. Numeric entry accepts scientific notation.
   **Raw configuration** edits a settings section; **Advanced definition and raw component**
   edits a component. Genuinely untyped factory args/kwargs retain raw JSON and no constructor
   schema is invented. Invalid field/raw buffers survive tabs/sections until corrected.
   Valid unrendered fields are retained. Unapplied/invalid YAML is a retained buffer and
   blocks semantic edits, Save and Run until Apply or deliberate Discard. YAML is an alternate
   editor, not a stacked panel. Leaving it retains the buffer and a **Return to YAML** notice;
   Runs remains available. Comments/formatting are not preserved.
5. **Check → Check fields** validates without compiling. **Check → Build model** explicitly runs constructors and
   dummy forwards in the selected interpreter. Displayed final key shapes are source-bound
   and become stale after semantic edits. Save uses revision conflicts and a separate
   `*.studio.json` position/size/name sidecar; an external semantic edit invalidates old placement metadata.
   Matching older sidecars without container sizes use defaults. Membership/order stay in ordinary YAML.
   The compact **Problems** footer expands manually or after a failed check/build/apply.
   Field success does not imply build success; semantic edits show **Needs rechecking**.
   Click a field-addressable problem to reveal and focus its inspector tab/Training section. Runtime
   errors without a field address remain linked to their frozen source, not guessed draft fields.
   Failed operations offer **Open frozen settings as draft** to edit their actual launch settings,
   including after a reload; unsaved draft replacement requires confirmation and never changes the run.
6. **Run…** stages backend/target/context/namespace/template/resources from the selected
   installation's schemas, including contributed backends. Refresh capacity is read-only;
   sourced node/resource/quota snapshots show age, partial/unknown information and observed
   zero separately. Capacity is advisory, not reserved; restricted node reads do not mean
   create permission is denied. **Review selection** checks semantics/access/admission and
   captures exact settings; **Run on …** confirms once. Changes invalidate review, template
   changes require renewed review, and Cancel discards staged edits. Accepted execution
   edits form one undoable ordinary draft edit without saving the file automatically.
   Optional local Trainer resume reviews the checkpoint-derived scenario (trusted Python
   input). Later draft edits do not change the run. One resource-consuming operation is allowed.
7. **Runs** selects retained training, build and export operations, even with no draft.
   **Metrics · Logs · Artifacts · Configuration** share that operation and its subscription;
   tab changes never relaunch. Status, errors and supported progress remain visible.
   Configuration is read-only frozen YAML; **Open frozen settings as draft** explicitly
   replaces the draft after unsaved-replacement confirmation. Draft matches/differs compares
   configuration and stage order, not server hashes; unavailable sources make no comparison.
   Runs shows native phase/epoch/batch progress, real scalar curves, readable results,
   process logs and reconnect state. Preparation/downloads without counters use indeterminate
   activity; carriage-return and supported ANSI progress updates overwrite terminal lines.
   These logs are not parsed into training counters. Raw output remains downloadable. Reload
   and select the existing operation; it is never relaunched. Visible history is bounded
   to 2,000 events, with a notice; charts separately retain 2,000 metric observations
   so verbose logs do not evict their samples. Full observations/logs remain on disk.
   Training loss updates during the first epoch from actual batch callbacks (at most
   four updates/second plus first/final batches). Charts use optimizer steps, and a
   single measurement is a visible point rather than an invisible line. Only **train/loss**
   and **val/loss** share one Total loss chart with actual step/value scales and distinct
   solid/dashed styles. Accuracy and custom scalars keep separate scales. Validation/test
   values appear after their phase completes, excluding the preliminary sanity check.
   Pipeline → **Objectives & metrics** edits `training.loss_keys` (weighted losses) and
   `training.metric_keys` (outputs to log). NexuML also reports their weighted loss total
   as `train/loss`, `val/loss`, and `test/loss`. Metrics must be produced by pipeline
   layers: the CIFAR ClassificationMetrics layer computes accuracy/F1 in validation/test,
   not training. Scalar evaluator results can also appear when evaluation finishes.
   Charts reflect the observed run's frozen configuration, not current draft settings.
8. Stop is available for owned local processes and verified Job/PyTorchJob identities, not
   direct RayCluster or RayJob work. Native Stop confirms UID-bound foreground deletion;
   a local driver exit is not worker cancellation. Local Stop confirms tree exit,
   not a new checkpoint. Runs → Artifacts downloads use authorized paths. Export requires a
   completed local source and an actual Trainer checkpoint, not random placeholder weights.
   Export selects its new operation; the source training record stays selectable. Unsupported
   telemetry/stop/resume is explained rather than simulated. Native log-only runs open Logs.

Ray delegates to existing NexuML guards/cluster connection and has driver logs/final
results only. Native jobs display actual pending/terminal state and available bounded
log tails, without fake scalar metrics/progress. External artifact URIs are references,
not authorized local downloads. API shutdown/restart leaves unverified remote work
ownership-unknown; **Inspect native reference** never resubmits it. See Compatibility
above and the Python docs' native-job template/handoff
guide before an operator-authorized cluster launch; no cluster smoke coverage is implied.
No automatic recovery, hosted collaboration, browser training or provisioning is added.

## Troubleshooting

- **Executable missing (terminal, before browser opens):** install with
  `uv tool install "nexuml[api,library]"`; use `uv tool update-shell` and reopen your shell if
  `nexuml` is not on PATH. Or select `--nexuml`/`--python` explicitly.
- **Missing API dependencies:** install the matching `api` extra yourself in the selected
  installation, for example `uv tool install --reinstall "nexuml[api,library]"` for a tool,
  or `uv pip install --python /project/.venv/bin/python "nexuml[api,library]"` for a project.
  Read the terminal diagnostic and selected executable; the launcher will not upgrade it.
- **Unsupported version:** Studio 0.1.x needs NexuML >=0.2.1,<0.3.0 and interface 2.
  Select a compatible Python with `--python` or explicitly reinstall a supported version.
- **Wrong Python / missing library:** check Runtime details. Install custom packages into
  that interpreter (see Choose an existing runtime above), not your unrelated shell
  Python. Discovery errors are available under Problems → Runtime and discovery details.
- **Missing working directory:** select an existing directory; Studio does not create one.
- **Backend exits or cannot start:** read its terminal output before the launcher error.
  The selected executable and exit status are reported; fix that environment, then restart.
- **Connection lost in the browser:** the draft is retained, but runtime/status values may
  be stale. Check the launcher terminal, then **Retry connection**. If the launcher was
  restarted, reload the page for its new session. Retained runs remain inspectable on disk.
- **403/attach failure:** match `--origin`, Studio port, directory and private token.
  `localhost` and `127.0.0.1` are distinct origins.
- **Hosted bootstrap/observations fail:** preserve the public Host header, match the exact
  configured HTTPS origin, inject the private proxy credential and enable WebSocket forwarding.
  Direct hosted listener access deliberately returns 401. Do not solve this by relaxing
  API origins or trusting `X-Forwarded-Host`/`X-Forwarded-Proto`.
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
npm pack  # prepack builds once and includes LICENSE
# Explicitly prepare an isolated backend from the matching Python wheels first.
STUDIO_TEST_PYTHON=/path/to/test/python npm run test:package -- /path/to/built-studio.tgz
```

Build/pack is a release task, never a normal-user startup task. API/Python checks and
installed-package/browser smoke instructions live in the repository's local API guide.
Do not publish either artifact automatically. Dependencies are locked; ESLint uses v9
because the stable Next plugins do not yet accept v10's peer range.

The repository's `CI` workflow runs on trusted repository branch pushes (including
`main` and `dev`), never fork PR events on persistent institute runners. Maintainers
must review external contributions before bringing them onto a trusted branch.
Studio static checks run once; one production npm tarball and the validated Python
wheels are uploaded as immutable artifacts. Linux project/uv-tool jobs test those exact
files, including cold installation, offline runtime and real CPU browser workflows.
macOS/Windows matrix entries are temporarily disabled, not claimed as tested. All feed the
single required `CI` gate. Package installation can download dependencies; installed
startup, fonts and browser transport remain offline checks. Each matrix job retains its
HTML browser report, smoke log and screenshots for seven days. Traces remain disabled
because they can contain runtime credentials.
The protected npm workflow publishes that CI artifact without rebuilding it; see
the repository's [release guide](https://github.com/NexuFed/NexuML/blob/main/docs/development/release.md).
