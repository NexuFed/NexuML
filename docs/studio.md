# NexuML Studio

Studio is a local, single-user workbench over your installed NexuML. YAML remains the
execution configuration; training runs in the selected Python environment, not the browser.
Libraries and configurations execute trusted Python, **not sandboxed code**.

## Install and launch

Install Python, Node.js/npm and a modern browser. Create or activate your project's
Python environment, then run:

```sh
pip install "nexuml[api,library]"
npm install -g @nexufed/nexuml-studio
nexuml-studio .
```

Studio opens the browser and reports the connected runtime. **Runtime details** shows
its Python executable and working directory. If that environment is not on PATH:

```sh
nexuml-studio --python /path/to/project/.venv/bin/python .
```

No installation, upgrade, dependency sync or internet access occurs at startup.
After installation, the local interface and synthetic example work offline.
Keep the terminal running: **Ctrl-C** stops Studio and its owned local processes;
closing the browser does not stop training.

## First training run

1. Click **Try an example**. The installed `synthetic-linear-ae-reconstruction` recipe
   becomes an editable **one-epoch CPU** draft with locally generated data and no downloads.
   Resolution calls trusted installed Python but does **not** train or change recipe defaults.
2. Inspect **Pipeline** and **Training**. **Save** asks for a YAML destination. You can
   also start with **Choose a recipe** or **Open YAML** in the working directory.
3. Use **Check → Check fields**, then **Check → Build model**. Field validation does
   not compile; Build executes constructors and dummy forwards. Semantic edits make
   previous build results stale.
4. Click **Run…**, review Local execution, then **Review selection → Run on Local**.
   Launch review captures exact settings; changing them requires another review.
5. In **Runs**, inspect **Metrics**, **Logs**, **Artifacts** and frozen **Configuration**.
   Tabs never relaunch work. Reload and select the retained operation to inspect it again.

If the example is missing, install the base library in the selected environment and
restart. Studio never installs it for you. Run outputs and records remain under
`.experiments/`, including `.experiments/.studio/operations/`, unless relocated with
`NEXUML_LOGS_ROOT`. Artifacts shows actual paths; YAML stays where you save it.

### Edit and inspect safely

- Add installed definitions from **Components**; edit their settings/routing in Properties.
  **Structure** and **Execution order** control stage/layer order. Canvas positions do not
  schedule execution. Move buttons and connection selects provide non-drag alternatives.
- Connections use the most recent preceding producer of a key. Missing required routes
  block launch; dynamic metadata stays runtime-only. Display names and positions live in a
  separate `*.studio.json` sidecar, not executable YAML. External semantic edits invalidate
  old layout metadata. Undo/Redo covers completed configuration and layout edits.
- YAML is an alternate editor. Invalid or unapplied text blocks semantic edits, Save and
  Run until Apply or deliberate Discard. Comments/formatting are not preserved. A save
  conflict never overwrites an externally changed file; reload or save to a new path.
- **Problems** links field-addressable errors to their controls. Runtime errors remain
  tied to frozen launch settings. **Open frozen settings as draft** restores those settings
  after confirmation if it would replace unsaved work; it does not change the run.
- Metrics are observed values, not inferred from progress bars. `training.loss_keys`
  selects weighted losses; `training.metric_keys` selects outputs actually produced by
  layers. Total losses use `<phase>/loss`; validation/test values appear after finalization.
  Display history is bounded, with a truncation notice; raw observations/logs stay on disk.

### Execution limits

One build/train/export operation may consume resources at a time; there is no queue.
Local Stop confirms process-tree exit, not a new checkpoint. Resume requires an actual
trusted Trainer checkpoint; export requires completed local training and its checkpoint.
Loading checkpoints can execute Python.

Contributed backends use the selected installation's schemas. Capacity is read-only,
advisory and never reserved. Ray exposes driver logs/results, not invented worker telemetry
or cancellation. Native jobs expose actual status and available log tails; only
Job/PyTorchJob support UID-verified Stop. External artifact URIs are references, not
authorized local downloads. After an API restart, unverified remote work is
`ownership_unknown`; inspecting a native reference never resubmits it. Read
[native job limits](how-to/training-backends/jobs.md) before an operator-authorized launch.
Studio does not provision clusters or recover/resume runs automatically.

## Choose a runtime

```sh
nexuml-studio .                            # nexuml on PATH
nexuml-studio --nexuml '/path to/nexuml' .  # isolated tool/global executable
nexuml-studio --python '/project/.venv/bin/python' '/working directory'
```

On Windows, select the project's `.venv\Scripts\python.exe`. Paths with spaces are
spawned as arguments, not shell commands. Install the `api` extra and all component
libraries in **that installation**. Do not select an unrelated shell Python for a uv tool.
To use an isolated tool environment:

```sh
uv tool install "nexuml[api,library]"
nexuml-studio --nexuml /path/to/nexuml .
```

For a packaged custom library, add `--with my-nexuml-library` to that tool installation,
or install it into the selected project interpreter with
`uv pip install --python /project/.venv/bin/python my-nexuml-library`.
For trusted source roots use **Libraries → Add root**; see
[local library roots](how-to/library-cli.md) for dependency and discovery rules.
To change installations, stop Studio and launch with another executable. Updates are
explicit user actions. Unsupported runtimes are rejected before the workbench opens;
read the launcher diagnostic rather than assuming frontend/backend releases advance together.

### Ports and attached APIs

Both services bind to loopback and select unused ports by default. Set
`--port 3000 --api-port 8000` for explicit ports; `--no-open` suppresses browser opening.
The directory must already exist and is filesystem context, not a Studio project.

To attach to a [manually started API](how-to/local-api.md):

```sh
nexuml-studio --port 3000 --api http://127.0.0.1:8000 --api-token-file /private/token .
```

The API must use the same directory and allow exactly `http://127.0.0.1:3000`.
An attached API is never stopped by the launcher. Keep tokens outside the working
directory, private (mode 600 on POSIX; restricted ACL on Windows), and out of arguments,
URLs, logs and `.env` files. Browser credentials stay in memory. Local users/processes
are trusted; Host/Origin/Fetch-Metadata checks do not authenticate arbitrary HTTP clients.

## Existing authenticated HTTPS reverse proxy

Local mode needs no proxy or account. Hosted mode requires an authenticated HTTPS proxy
and a **separate private proxy credential**, never the Python API token:

```sh
umask 077
python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$HOME/.nexuml-studio-proxy-token"
nexuml-studio --port 3000 --public-origin https://studio.example \
  --proxy-token-file "$HOME/.nexuml-studio-proxy-token" --no-open /path/to/work
```

The credential needs at least 32 random bytes and the private-file permissions above.
`NEXUML_STUDIO_PUBLIC_ORIGIN`, `NEXUML_STUDIO_HOST` and
`NEXUML_STUDIO_PROXY_TOKEN_FILE` supply the same settings; CLI flags take precedence.
The listener stays on **127.0.0.1**, even when hosted. A non-loopback bind requires
explicit `--host` with an IPv4 address, public origin and proxy credential, and emits a
warning. The public origin must be exact HTTPS without credentials, trailing slash,
path, query or fragment. The Python API stays loopback-only; its allowed origin remains
the **local UI origin**, including in attach mode.

Configure NetBird or your existing proxy outside Studio:

- One `/` target to the **UI port**, **Pass Host Header enabled**, HTTPS, WebSocket
  forwarding, and authentication/access restrictions for trusted operators.
- After authentication, **overwrite** `X-NexuML-Studio-Proxy` with the private credential
  on every upstream HTTP/WebSocket request. Never accept a client-supplied value or expose
  it in browser code, URLs, logs or responses. If the proxy cannot inject private headers,
  use a restricted header-injecting target proxy or keep Studio local.
- No API proxy target or externally reachable API port. Firewall any non-loopback UI
  listener to the proxy host only, not arbitrary private-network clients.

Direct hosted bootstrap/API access requires the proxy credential even with spoofed
local/public Host/Origin. There is no unauthenticated local bypass. Browser HTTP and
WS/WSS use same-origin `/api/v1`; Next forwards to the runtime-selected loopback API,
preserving bearer/first-frame authentication, rewriting upstream Host/Origin and
stripping cookies, forwarded metadata and the proxy credential. Bootstrap never exposes
the private API URL. Forwarded Host/Proto or a public origin alone cannot establish trust.

This is **trusted-operator hosting, not public/multi-tenant SaaS**: anyone admitted can
obtain the runtime credential and execute local Python. Studio adds no login system and
does not configure NetBird. Development needs the same external restrictions.

### Hosted development

After [contributor installation](development/install.md), start the API separately with
the local `--origin http://127.0.0.1:3000`, then from `studio/`:

```sh
export NEXUML_STUDIO_ORIGIN=http://127.0.0.1:3000
export NEXUML_STUDIO_API=http://127.0.0.1:8000
export NEXUML_STUDIO_TOKEN="$(cat /private/token)"
export NEXUML_STUDIO_PUBLIC_ORIGIN=https://studio.example
export NEXUML_STUDIO_PROXY_TOKEN="$(cat "$HOME/.nexuml-studio-proxy-token")"
export NEXUML_STUDIO_HOST=127.0.0.1
npm run dev -- --hostname "$NEXUML_STUDIO_HOST" --port 3000
```

HMR allows only loopback and the configured public hostname, not a wildcard.
Public origin/API port are runtime settings, not a public API URL baked into the build.

## Troubleshooting

- **Executable/API/library missing:** check the terminal diagnostic and Runtime details.
  Install the needed extra/library into the selected interpreter, then restart; for uv
  tools use `uv tool update-shell` if the executable is missing from PATH.
- **Connection lost:** the draft stays, but status may be stale. Check the terminal, then
  **Retry connection**. Reload after launcher restart to receive the new session.
- **Attach rejected:** match directory, token, Studio port and exact API `--origin`.
  `localhost` and `127.0.0.1` are different origins.
- **Hosted bootstrap/observations fail:** check Host preservation, exact HTTPS origin,
  private-header injection and WebSocket forwarding. Do not relax API origins or trust
  forwarded headers to bypass authentication.
- **Stale temporary directories after a crash:** remove `nexuml-studio-*` directories only
  after confirming their services stopped. Run records remain under NexuML's logs root.

For contributor checks and package/browser smoke commands, see
[Development workflow](how-to/development.md).
