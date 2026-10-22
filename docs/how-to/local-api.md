# Local Studio API

The optional API runs over the selected installed NexuML. For the browser workbench,
use [Studio](../studio.md); this page covers manual startup and transport rules.
It does not provision Python, dependencies or environments.

## Start manually

Create or activate your chosen Python environment and install:

```sh
pip install "nexuml[api,library]"
```

Set a random token in a private file **outside the working directory**. This POSIX
example keeps it out of command arguments and URLs:

```sh
umask 077
python -c 'import secrets; print(secrets.token_urlsafe(32))' > "$HOME/.nexuml-api-token"
nexuml serve --directory . --port 8000 --origin http://127.0.0.1:3000 \
  --token-file "$HOME/.nexuml-api-token"
```

Alternatively set `NEXUML_API_TOKEN` in the server environment. Token files need mode
600 on POSIX or a restricted Windows ACL; keep them out of version control. Invoke a
uv tool's own `nexuml` executable, not an unrelated shell Python. A project interpreter
can invoke `-m nexuml.cli.main serve` with the same options. See
[runtime selection](../studio.md#choose-a-runtime) for isolated-tool installation.

## Authentication and REST reference

- The API binds only to `127.0.0.1`; Host must name a loopback host. `--origin` is an exact
  loopback HTTP origin without a slash/path. `localhost` and `127.0.0.1` differ.
- REST requires `Authorization: Bearer <token>`. Fetch **`/openapi.json` with that
  header** for request/response schemas and routes; never put the token in a URL.
  `GET /api/v1/runtime` reports the selected runtime/interface. Clients may send
  `X-NexuML-Interface` using the reported interface value; a mismatch returns 409.
- Catalog entries keep parameter `schema` separate from native `io` bindings and fixed
  labels. Unknown contracts remain unresolved; discovery does not build components.
  Studio and Python must support the same interface; neither updates packages automatically.
- Errors expose `error.code`, `error.message` and optional `error.fields`
  (`loc`, `message`, `type`), without echoing request inputs.
- CORS/Host/Origin checks are not authentication for arbitrary HTTP clients. Discovery,
  scenario resolution, builds and checkpoint loading execute trusted Python; process
  separation is **not a sandbox**. Only use trusted libraries/configurations/checkpoints.

### Configuration and conflicts

Portable responses include `data`, `yaml`, `stage_order` and `semantic_revision`.
Send `stage_order` back with JSON edits: integer-like JavaScript object keys can reorder,
but pipeline order is executable semantics. Layout is a separate `*.studio.json`
sidecar and applies only to matching semantic revisions; it never controls membership
or execution order. YAML comments/formatting are not preserved.

Load returns `base_revision`; save requires it (null only for a new file). External
changes yield 409 `conflict` without overwriting either version. Invalid YAML/fields
yield 422 and leave saved files unchanged. Transport file access rejects traversal and
symlink escapes outside the working directory. Adding an external trusted library root
permits discovery, not arbitrary HTTP browsing. Existing dataset/log path policies remain.

## WebSocket observation and replay

Connect to `ws://127.0.0.1:<api-port>/api/v1/operations/<id>/events?after=<sequence>`.
The upgrade requires the exact allowed Origin. Within five seconds send the first text
frame `{"token":"<token>"}` (at most 4096 bytes). No data precedes authentication;
invalid access closes with policy code 1008. Never put credentials in the URL.

The API sends a `snapshot`, then sequenced `progress`, `metrics`, `log`, `replay_gap`
and `terminal` events. Reconnect from the last sequence; disconnect/reload only detaches
observation and **never relaunches or cancels training**. Frames are bounded and awaited
sends apply backpressure from disk replay. Missing/oversized history produces a replay
gap. An oversized snapshot follows `replay_gap` with `partial: true`; fetch the full
`GET /api/v1/operations/{id}` status instead of replacing it with the reduced snapshot.
Raw logs and observations remain on disk even when the UI display is truncated.

## Operation lifecycle and limits

One build/train/export operation may be active; another receives 409 `busy`, with no
queue. Builds run constructors/dummy forwards. Local training uses `NexuSession.run()`;
metrics come from actual callbacks, not parsed logs or fabricated progress. Resume
reviews the trusted checkpoint-derived scenario; later draft edits do not alter a run.
Export requires completed local training and its actual checkpoint, never fresh random
weights. Downloads are limited to authorized paths under the working directory or
resolved logs root; external artifact URIs remain references.

Records live under `.experiments/.studio/operations/`, honoring `NEXUML_LOGS_ROOT`.
Shutdown stops confirmed owned local work, without promising a checkpoint. After restart,
terminal records remain inspectable and prior active records become `ownership_unknown`;
there is no PID adoption or automatic resume. A Ray driver exit is not proof that workers
stopped. Native shutdown detaches observation without claiming remote cancellation.
Only Job/PyTorchJob advertise UID-verified foreground Stop; RayCluster/RayJob do not.
Inspecting a native reference never resubmits it. See
[execution modes](training-backends/index.md) and [native job limits](training-backends/jobs.md)
before a separately authorized cluster launch; the API never provisions a cluster.
