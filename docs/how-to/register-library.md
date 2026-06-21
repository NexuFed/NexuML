# Register a library

NexuML discovers external libraries at runtime via Python entry-points. Any installed package that declares a `nexuml.libraries` entry-point is loaded automatically.

## 1. Structure your library

```
my_library/
├── pyproject.toml
└── src/
    └── my_library/
        ├── __init__.py
        ├── layers/
        │   └── my_layer.py
        ├── data/
        │   └── my_dataset.py
        ├── evaluation/
        │   └── my_eval.py
        └── scenarios/
            └── my_scenario.py
```

## 2. Declare the entry-point

In `my_library/pyproject.toml`:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

The entry-point value must be the **importable package name**. NexuML scans the whole package tree for decorated elements. There is no `register()` function requirement — decorators on classes and functions inside the package are sufficient.

## 3. Install and verify

```bash
uv pip install -e /path/to/my_library
nexuml library list       # shows installed libraries
nexuml registry list      # shows all registered elements
```

## CLI library management

For libraries not yet installable as packages:

```bash
nexuml library add /path/to/my_library    # add a path-based library root
nexuml library list                        # list all known library roots
```

See [Managing local library roots](library-cli.md).

## See also

- [`nexuml.core.discovery`](../reference/api/nexuml/core/discovery.md)
- [Library discovery explanation](../explanation/library-discovery.md)
- [Custom library end-to-end tutorial](custom-library.md)
