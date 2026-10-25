# nexuml.api.worker

*module*

Call existing framework APIs in a fresh process, never parse terminal output.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L1-L1)

## nexuml.api.worker.catalog

*function*

```python
def catalog() -> dict
```

Read authoritative registries/schemas and existing discovery diagnostics.

**Returns**

- (`dict`) — Existing library/component/scenario metadata and runtime dependency availability.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L64-L121)

## nexuml.api.worker.configuration

*function*

```python
def configuration(payload: dict) -> ResolvedConfig
```

Restore portable typed configuration, preserving explicit stage order.

**Returns**

- (`ResolvedConfig`) — Existing validated configuration model, without compiling it.

**Raises**

- `ValueError` — If the representation or ordered stage view is invalid.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L17-L45)

## nexuml.api.worker.dispatch

*function*

```python
def dispatch(action: str, payload: dict) -> dict
```

Route transport calls to existing NexuML APIs.

**Returns**

- (`dict`) — Portable worker result.

**Raises**

- `ValueError` — For unknown operations (there is no arbitrary code endpoint).

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L124-L225)

## nexuml.api.worker.document

*function*

```python
def document(config: ResolvedConfig) -> dict
```

Return editable existing configuration plus representation metadata.

**Returns**

- (`dict`) — Portable config, ordered stage names, ordinary YAML and semantic revision.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L48-L61)

## nexuml.api.worker.execute

*function*

```python
def execute(action: str, payload: dict) -> dict
```

Delegate execution and exports to the existing session/backend lifecycle.

**Returns**

- (`dict`) — Actual results and existing-format artifact references.

**Raises**

- `ValueError` — When an unsupported backend/export combination is requested.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L228-L286)

## nexuml.api.worker.main

*function*

```python
def main() -> None
```

Read a private request and write a plain response, not terminal logs.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/src/nexuml/api/worker.py#L289-L330)
