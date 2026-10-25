# nexuml.core.log_paths

*module*

Utilities for resolving log and artifact output paths.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/log_paths.py#L1-L1)

## nexuml.core.log_paths.LOGS_ROOT_ENV

*attribute* · *module attribute*

```python
LOGS_ROOT_ENV = 'NEXUML_LOGS_ROOT'
```

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/log_paths.py#L9-L9)

## nexuml.core.log_paths.resolve_logs_file_uri

*function*

```python
def resolve_logs_file_uri(uri: str, *, env_var: str = LOGS_ROOT_ENV) -> str
```

Resolve relative file URIs under logs root; preserve absolute/remote URIs.

**Returns**

- (`str`) — Resolved ``file://`` URI with absolute path, or the original *uri*
- (`str`) — unchanged if it is absolute or remote.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/log_paths.py#L30-L42)

## nexuml.core.log_paths.resolve_logs_root

*function*

```python
def resolve_logs_root(path: str | Path, *, env_var: str = LOGS_ROOT_ENV) -> Path
```

Prefix relative log paths with ``NEXUML_LOGS_ROOT`` when configured.

**Returns**

- (`Path`) — Resolved absolute or relative path.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/log_paths.py#L12-L22)

## nexuml.core.log_paths.resolve_logs_root_str

*function*

```python
def resolve_logs_root_str(path: str | Path, *, env_var: str = LOGS_ROOT_ENV) -> str
```

Return the logs root path resolved to a string.

[View source](https://github.com/NexuFed/NexuML/blob/1e2e9e39fea3b6a25da3752c55398e43e00d3830/src/nexuml/core/log_paths.py#L25-L27)
