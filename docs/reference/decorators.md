# Discovery Decorators

All decorators live in `nexuml.core.discovery`.

| Decorator | Required object | Python use | Registry kind |
|---|---|---|---|
| `@layer(name, version="1")` | `LayerDefinition` subclass | `LayerSpec(component=Definition(...))` | `layer` |
| `@data_source(name, version="1")` | `DataSourceDefinition` subclass | `DataSpec(source=Definition(...))` | `data_source` |
| `@eval_algorithm(name, version="1")` | `EvalAlgorithmDefinition` subclass | `EvalAlgorithmSpec(algorithm=Definition(...))` | `eval_algorithm` |
| `@loader_backend(name, version="1")` | `LoaderBackendDefinition` subclass | `LoaderSpec(backend=Definition(...))` | `loader_backend` |
| `@scenario(name)` | Callable returning `ScenarioSpec` | CLI scenario lookup | separate scenario registry |

Component decorators validate the role immediately and attach explicit kind, name, and version metadata. Concrete definition fields are ordinary Pydantic fields and are visible through `model_json_schema()`.

```python
from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source


@data_source("my_dataset", version="1")
class MyDataset(DataSourceDefinition):
    root: str

    def build(self):
        return _MyDatasetRuntime(root=self.root)
```

Python imports and constructs `MyDataset(root="/data")`. At a serialization boundary, the registry lowers it to:

```yaml
type: my_dataset
version: '1'
params:
  root: /data
```

No concrete module path is persisted. Restoration performs an exact kind/name/version lookup followed by Pydantic validation.

Discovery scans built-ins, installed `nexuml.libraries` entry points, and configured local roots on each CLI run. Duplicate identities within the same `(kind, name, version)` fail registration; failures are collected without stopping unrelated modules.

## See Also

- [Decorators and discovery](../learn/decorators-and-discovery.md)
- [Registry inspection](registry.md)
- [Add a custom layer](../how-to/custom-layer.md)
