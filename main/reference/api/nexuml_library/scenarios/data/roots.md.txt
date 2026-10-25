# nexuml_library.scenarios.data.roots

*module*

Shared data-root resolution for scenario data builders.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/data/roots.py#L1-L1)

## nexuml_library.scenarios.data.roots.resolve_data_root

*function*

```python
def resolve_data_root(data_root: str, env_var: str = 'NEXUML_DATA_ROOT') -> Path
```

Resolve a dataset path against the optional global data root.

Absolute paths are used as-is. Relative paths are resolved under
``NEXUML_DATA_ROOT`` when set. Otherwise, relative paths are returned
unchanged and remain relative to the current working directory.

**Returns**

- `Path` (`Path`) — Resolved dataset path.

[View source](https://github.com/NexuFed/NexuML/blob/b9fb6586f015db459cb8881e6feb51b84b3f03e0/library/src/nexuml_library/scenarios/data/roots.py#L9-L27)
