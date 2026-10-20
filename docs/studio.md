# NexuML Studio — Linux Developer Preview

NexuML Studio runs over your selected, already-installed NexuML. See the
[Studio guide](https://github.com/NexuFed/NexuML/blob/main/studio/README.md) for installation, editing, training and runtime selection.
YAML is authoritative; Studio does not install or synchronize Python environments.

## Linux quickstart

Prerequisites: **Python >=3.12**, **Node >=22.13**, **uv**, and a modern browser.

```sh
uv tool install "nexuml[api,library]"
npm install -g @nexufed/nexuml-studio
nexuml-studio .
```

Studio opens a local browser interface. Its welcome screen shows the actual connected
NexuML/Python versions; **Runtime details** shows the Python executable and working directory.
Training runs in that selected Python environment. Install all base/custom libraries in
the **same environment**, not an unrelated shell Python. To select an existing project:

```sh
nexuml-studio --python /path/to/project/.venv/bin/python .
```

Keep the terminal running. **Ctrl-C** stops Studio and its owned local processes; closing
the browser does not stop training. Outputs remain under `.experiments/` in the working
directory, with Studio records in `.experiments/.studio/operations/`, unless relocated
using `NEXUML_LOGS_ROOT`. Runs → Artifacts shows actual paths. YAML saves where you choose.
After installation the local interface and synthetic example work offline; no packages
are installed, upgraded or synchronized at startup.

**Linux developer preview:** macOS/Windows are not release-validated. Registry commands
apply once compatible preview releases are published; for unreleased candidates, install
the separately built wheels/npm tarball described in the Studio guide.
Studio 0.1.x supports stable NexuML
>=0.2.1,<0.3.0 (Python >=3.12) with **API interface 2**. Backend and frontend package
versions are independent; supported backend patches do not force simultaneous releases.
An incompatible backend is rejected before the workbench opens, with the required version.

## First training run

1. Click **Try an example** to resolve the installed synthetic autoencoder into an editable
   **one-epoch CPU** draft. It generates data locally, with no downloads. Resolution executes
   trusted installed Python; loading the example does **not** start training.
2. Inspect/edit it in **Pipeline** and **Training**. **Save** writes ordinary NexuML YAML.
3. Click **Run…** → **Review selection** → **Run on Local** to confirm execution.
4. In **Runs**, inspect real **Metrics**, **Logs**, **Artifacts** and frozen **Configuration**.
   Reload and select the retained operation to inspect it again, without retraining.

**Choose a recipe** and **Open YAML** remain available. If the example or a custom library
is missing, install it in the selected Python environment and restart Studio. Trusted
source roots can also use Libraries → Add root or `nexuml library add`.
For startup errors, read the launcher terminal; for a later disconnect, the UI retains
your draft and offers **Retry connection**. Reload after restarting the launcher.
See the [Studio guide](https://github.com/NexuFed/NexuML/blob/main/studio/README.md#troubleshooting)
for installation/error recovery, custom libraries, attached APIs and execution backends.

## Local/offline default

```sh
nexuml-studio --port 3000 --api-port 8000 /path/to/work
```

Both processes bind to loopback by default. The browser opens `http://127.0.0.1:3000`
and sends HTTP/WebSocket traffic through that same origin. Prebuilt UI startup requires
no internet access or dependency installation. The private API never needs browser access.
The launcher stops only services it owns; an attached API is never stopped.

## Existing authenticated HTTPS reverse proxy

```sh
nexuml-studio --port 3000 --public-origin https://studio.example \
  --proxy-token-file /private/studio-proxy-token --no-open /path/to/work
```

The listener stays on **127.0.0.1**, including with `--public-origin` (or
`NEXUML_STUDIO_PUBLIC_ORIGIN`). Non-loopback access requires explicit `--host` (or
`NEXUML_STUDIO_HOST`), an exact HTTPS public origin and a private proxy credential,
and emits a warning. The file must contain at least 32 random bytes, be mode 600 on
POSIX (restricted ACL on Windows), and differ from the Python API token. It can also
be selected with `NEXUML_STUDIO_PROXY_TOKEN_FILE`. CLI flags take precedence.
Origins have no credentials, trailing slash,
path, query or fragment. The API always remains loopback-only; its configured origin
remains `http://127.0.0.1:<UI port>`, including in attach mode.

Configure the proxy, including NetBird, outside Studio:

- One `/` target to the **UI port**, with **Pass Host Header enabled**.
- HTTPS, WebSocket forwarding, and authentication/access restrictions for trusted operators.
- After authentication, overwrite `X-NexuML-Studio-Proxy` with the private credential on
  every upstream HTTP/WebSocket request. Never forward a client-supplied value. If the
  proxy cannot inject private headers, use a restricted header-injecting target proxy
  or keep Studio local; forwarded identity headers do not replace authentication.
- No API target or externally reachable API port.
- Firewall any non-loopback UI port to the proxy host only. Direct hosted bootstrap/API
  access, including with spoofed local/public Host/Origin, is rejected without the proxy
  credential. There is no unauthenticated local bypass in hosted mode.

All browser API requests and observations use the public origin; HTTPS uses WSS.
The Next proxy forwards to the runtime-selected API port. It preserves browser bearer
authorization and WebSocket first-frame auth, rewrites upstream Host/Origin to their
local values, and strips cookies and forwarded-proxy metadata. Bootstrap returns the
browser origin and an in-memory runtime credential, never the private API address.
Host, optional Origin and Fetch Metadata must match configured settings; forwarded
Host/Proto do not establish trust. **Host/Origin checks are not authentication for arbitrary
HTTP clients.** The private proxy credential authenticates the hosted boundary and is
stripped upstream; Python token/origin/interface validation is unchanged.

This is trusted-operator hosting, **not public/multi-tenant SaaS**. Anyone admitted to
Studio can obtain its runtime credential and execute trusted local Python. Studio adds
no account system and does not mutate NetBird settings. Production and development need
the same external access restrictions. For dev/HMR commands, see the Studio guide; the
allowlist includes only `127.0.0.1` and the explicitly configured public hostname.
