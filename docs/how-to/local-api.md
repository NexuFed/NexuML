# Local Studio API (unreleased)

The API is an optional interface over the **selected installed NexuML**. It does not
install Python, libraries, GPU dependencies, or environments. No account is needed.
The Studio frontend is delivered separately; these instructions cover manual API startup.

## Installation

Use a distribution built from this change until a release containing the `api` extra
is published. Install that wheel with the matching extra in your chosen environment:

```bash
uv tool install '/path/to/nexuml-0.2.1-py3-none-any.whl[api]'
# Or, in an existing project environment:
uv pip install '/path/to/nexuml-0.2.1-py3-none-any.whl[api]'
```

After a release containing the API is published, the package-name forms are:

```bash
uv tool install 'nexuml[api]'
uv tool install 'nexuml[api,library]'  # includes the base component library
uv add 'nexuml[api,library]'          # existing uv project
uv pip install 'nexuml[api,library]'  # existing active environment
```

Do not assume that `python` on PATH belongs to a uv-tool installation. Invoke that
tool's `nexuml` entrypoint. For a project installation, invoke its existing interpreter
directly (`.venv/bin/python`, or `.venv\Scripts\python.exe` on Windows):

```bash
/path/to/nexuml serve --directory /path/to/work --port 8000 --origin http://127.0.0.1:3000
/path/to/.venv/bin/python -m nexuml.cli.main serve --directory /path/to/work
```

Set a random token first. This POSIX example uses a private file outside the working
directory; the token is never a command-line argument or URL:

```bash
umask 077
python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$HOME/.nexuml-api-token"
nexuml serve --directory . --token-file "$HOME/.nexuml-api-token"
```

Alternatively set `NEXUML_API_TOKEN` in the server's environment. A file must be private
(`chmod 600` on POSIX; restrict its ACL on Windows). Keep it out of version control.
Startup fails explicitly if the selected installation lacks server dependencies.
No install or environment sync runs on startup. Existing nonserver commands do not
import FastAPI.

## Trust and authentication

- The server binds only to `127.0.0.1`. Host headers must name a loopback host.
- `--origin` must be an exact loopback HTTP origin, without a slash/path. Only that
  browser origin is allowed. `localhost` and `127.0.0.1` are different origins.
- REST requires `Authorization: Bearer <token>`. Optional `X-NexuML-Interface: 1`
  detects a mismatched client. Runtime identity is at `GET /api/v1/runtime`.
- WebSocket upgrades require the exact allowed Origin. The first text frame must
  be `{"token":"<token>"}` within five seconds and at most 4096 bytes. No data is
  sent before authentication; invalid access closes with policy code 1008.
- CORS is not authentication. Browser closure is not cancellation.
- Discovery imports trusted local Python. Resolving a scenario calls its factory;
  an explicit build runs component constructors and dummy forwards. Process separation
  is **not a sandbox**. Do not use untrusted libraries or configurations.

The authenticated OpenAPI contract is available at `/openapi.json`; do not put the
token in a browser URL. REST errors use `error.code`, `error.message`, and optional
`error.fields` (`loc`, `message`, `type`) without echoing request inputs.

## Discovery and configuration

| Route | Existing operation |
| --- | --- |
| `GET /api/v1/registry` | Fresh component/scenario discovery, schemas, sources, diagnostics and backend availability |
| `GET /api/v1/scenarios` | Discovered scenario names and diagnostics |
| `GET /api/v1/libraries` | Installed packages and existing configured roots |
| `POST /api/v1/libraries` | Add `{"path":"..."}` through existing `LibraryConfig` |
| `DELETE /api/v1/libraries` | Remove an explicitly configured root |
| `POST /api/v1/scenarios/{name}/resolve` | Call the selected registered scenario factory, without compilation |
| `POST /api/v1/scenarios/resolve-file` | Explicit trusted Python file within the selected directory |
| `POST /api/v1/config/validate` | Validate exactly one of `yaml` text or portable `data` |
| `POST /api/v1/config/load` | Load `{"path":"..."}` within the selected directory |
| `POST /api/v1/config/save` | Validate and save `path`, `base_revision`, and config input |
| `POST /api/v1/config/layout/load` | Read optional `path + ".studio.json"` sidecar |
| `POST /api/v1/config/layout/save` | Save `path`, `base_revision`, `semantic_revision`, `layout` |

Responses contain NexuML's ordinary portable `data`, `yaml`, `stage_order` and a
`semantic_revision`. Send `stage_order` back with JSON edits: JavaScript can reorder
integer-like object keys, but pipeline stage order is executable semantics. The API
restores that mapping order before model validation. Node positions never belong in it.

Forms consume each installed component's actual schema. Generic Python factory kwargs
remain structured values; the API does not invent constructor fields/defaults.
Validation may import definitions, but does not compile or run a dummy forward.
YAML comments/formatting and Python source formatting are not preserved; configuration
meaning, identities/versions, key aliases, evaluation and execution settings are.

Load returns a file `base_revision`; save requires that revision (null only for a new
file). External changes yield HTTP 409 `conflict` without overwriting either version.
Invalid YAML/fields yield HTTP 422 and leave saved files unchanged. Layout is saved
separately and should be applied only when its semantic revision matches the config.

Transport reads/writes reject traversal and symlink escapes outside the selected working
directory. Explicitly adding an external library root permits normal trusted discovery,
not arbitrary HTTP file browsing. Dataset/log path policies remain NexuML's.

## Build, train and observe

`POST /api/v1/build` accepts the same configuration input, plus optional `timeout`
(seconds; default 120). It **executes** the existing compiler, constructors and dummy
forwards. Successful results contain final key shapes, not fabricated per-layer shapes.
`POST /api/v1/train` uses the existing local `NexuSession.run()` lifecycle or existing
Ray entrypoint. Both return HTTP 202 with an operation `id`. One build/train/export
operation may be active; a second receives HTTP 409 `busy`. There is no queue.

Local resume accepts `trainer_checkpoint`, an authorized trusted `.ckpt` file. Checkpoint
loading can execute Python. `POST /api/v1/train/prepare` resolves the actual checkpoint
scenario for launch review; the checkpoint's scenario, not later draft edits, is authoritative.
Ray Trainer checkpoint resume is rejected; Ray retains its own recovery semantics.

| Resource | Behavior |
| --- | --- |
| `GET /api/v1/operations` | Persisted API-started invocation snapshots |
| `GET /api/v1/operations/{id}` | Actual state, frozen source revision, results, artifact references |
| `GET /api/v1/operations/{id}/config` | Frozen ordinary launch configuration |
| `POST /api/v1/operations/{id}/cancel` | Confirm stop of owned local process tree; no promised checkpoint |
| `GET /api/v1/operations/{id}/artifacts/{index}` | Existing authorized artifact download |
| `POST /api/v1/export` | `source_id`, `kind`, `output`; requires completed local training and its actual checkpoint |

Export kinds are existing `train_package`, `safetensors`, and `onnx` (requires installed
ONNX dependencies). Missing files/dependencies are explicit errors. Exports never
compile a fresh pipeline and call its random weights "trained". Configured exports after
training use that session's actual pipeline. Download paths must resolve within the
working directory or the existing resolved logs root; references outside those roots
remain references, not an unrestricted download facility.

Observe at `ws://127.0.0.1:<api-port>/api/v1/operations/<id>/events?after=<sequence>`.
Authenticate with the first frame described above (no token in the URL). The API sends
a `snapshot`, then sequenced `progress`, `metrics`, `log`, `replay_gap`, and `terminal`
events. Local scalar values come from actual Lightning callbacks. Logs are actual process
output, not parsed progress bars. Ray shows driver logs and returned metrics only; remote
worker telemetry/cancellation is not invented. Existing Ray evaluation/post-training-fit
compatibility guards still apply, and the API never provisions a cluster.

Disconnect only detaches observation. Reconnect from the last sequence; it does not
relaunch training. Individual frames are bounded; awaited sends apply backpressure
directly from the disk replay, rather than an unbounded subscriber queue. Oversized or
missing history is reported as a replay gap. Raw logs remain available on disk.
An oversized snapshot is preceded by `replay_gap` and carries `partial: true`; fetch
the complete status from `GET /api/v1/operations/{id}` rather than replacing a full
operation with the reduced snapshot.

Records live under `resolve_logs_root('.experiments/.studio/operations')`, honoring
`NEXUML_LOGS_ROOT`.
They are sidecars for existing invocations, not a second experiment/project database.

API shutdown stops confirmed owned local work. It cannot claim Ray workers stopped
when the driver exits. After a restart, terminal records remain inspectable; prior active
records become `ownership_unknown`. No PID-based adoption or automatic resume occurs.
