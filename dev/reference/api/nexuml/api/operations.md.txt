# nexuml.api.operations

*module*

Own and observe API-started calls, not a second ML execution engine.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L1-L1)

## nexuml.api.operations.MAX_EVENT

*attribute* · *module attribute*

```python
MAX_EVENT = 65500
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L26-L26)

## nexuml.api.operations.TERMINAL

*attribute* · *module attribute*

```python
TERMINAL = {'succeeded', 'failed', 'cancelled', 'timeout', 'ownership_unknown'}
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L25-L25)

## nexuml.api.operations.Operations

*class*

```python
class Operations
```

One local invocation owner with disk-backed, sequenced observation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L98-L469)

### nexuml.api.operations.Operations.active

*attribute* · *instance attribute*

```python
active: tuple[str, subprocess.Popen] | None = None
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L108-L108)

### nexuml.api.operations.Operations.lock

*attribute* · *instance attribute*

```python
lock = threading.RLock()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L107-L107)

### nexuml.api.operations.Operations.root

*attribute* · *instance attribute*

```python
root = settings.directory / resolve_logs_root('.experiments/.studio/operations').resolve()
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L103-L105)

### nexuml.api.operations.Operations.settings

*attribute* · *instance attribute*

```python
settings = settings
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L102-L102)

### nexuml.api.operations.Operations.stopping

*attribute* · *instance attribute*

```python
stopping: dict[str, str] = {}
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L110-L110)

### nexuml.api.operations.Operations.threads

*attribute* · *instance attribute*

```python
threads: list[threading.Thread] = []
```

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L109-L109)

### nexuml.api.operations.Operations.append

*method*

```python
def append(folder: Path, kind: str, payload: dict) -> None
```

Record bounded observations before clients can replay them.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L158-L195)

### nexuml.api.operations.Operations.cancel

*method*

```python
def cancel(identity: str) -> dict
```

Stop confirmed owned local work only.

**Returns**

- (`dict`) — Current snapshot; terminal cancellation is published after confirmed exit.

**Raises**

- `ApiError` — For unsupported remote stop or unknown ownership.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L399-L441)

### nexuml.api.operations.Operations.close

*method*

```python
def close() -> None
```

Stop owned local calls on API shutdown; never claim remote cancellation.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L443-L453)

### nexuml.api.operations.Operations.folder

*method*

```python
def folder(identity: str) -> Path
```

Locate an observed invocation safely.

**Returns**

- (`Path`) — Existing observation directory.

**Raises**

- `ApiError` — For an invalid or missing invocation identity.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L118-L132)

### nexuml.api.operations.Operations.inspect

*method*

```python
def inspect(identity: str) -> dict
```

Explicitly inspect a stored remote identity without resubmitting.

**Returns**

- (`dict`) — The native observation, independent of local process ownership.

**Raises**

- `ApiError` — When no native identity was recorded.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L455-L469)

### nexuml.api.operations.Operations.list

*method*

```python
def list() -> builtins.list[dict]
```

List API-observed invocations, not unrelated CLI processes.

**Returns**

- (`builtins.list[dict]`) — Persisted snapshots, newest first.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L142-L156)

### nexuml.api.operations.Operations.observe

*method*

```python
def observe(folder: Path, process: subprocess.Popen, timeout: float | None) -> None
```

Drain actual logs/callbacks and publish the real terminal result.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L297-L397)

### nexuml.api.operations.Operations.start

*method*

```python
def start(kind: str, payload: dict, *, timeout: float | None = None) -> dict
```

Freeze a validated config and call the existing framework in an owned process.

**Returns**

- (`dict`) — Running invocation snapshot.

**Raises**

- `ApiError` — If another resource-consuming call is active or startup fails.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L201-L295)

### nexuml.api.operations.Operations.status

*method*

```python
def status(identity: str) -> dict
```

Read actual persisted status.

**Returns**

- (`dict`) — Invocation snapshot, including its frozen source revision and capabilities.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L134-L140)

## nexuml.api.operations.stop_process_tree

*function*

```python
def stop_process_tree(process: subprocess.Popen) -> None
```

Stop only an owned process group and wait for local execution to exit.

**Raises**

- `ApiError` — If the OS cannot confirm stopping the owned process tree.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L36-L95)

## nexuml.api.operations.write_json

*function*

```python
def write_json(path: Path, data: dict) -> None
```

Atomically publish a status snapshot for reconnect/restart readers.

[View source](https://github.com/NexuFed/NexuML/blob/171c58d7e9a055ea6e378a6ff987baaded5ee009/src/nexuml/api/operations.py#L29-L33)
