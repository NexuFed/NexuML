# Registry inspection

The NexuML registry stores discovered layers, data sources, scenarios, and eval algorithms. Use `nexuml registry list` to inspect what is available in the current environment.

## Commands

### List by kind

```bash
nexuml registry list layers       # PipelineLayer subclasses
nexuml registry list data         # data source classes
nexuml registry list scenarios    # scenario functions
nexuml registry list eval         # evaluation algorithm classes
```

### Verbose mode — show discovery errors

```bash
nexuml registry list layers --verbose
nexuml registry list layers -v
```

In verbose mode, any modules that failed to import during scanning are shown with their error messages. This is the primary tool for diagnosing missing or broken library elements.

## Reading the output

```
              Registered Layers
┏━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━┓
┃ Type Key         ┃ Module                              ┃ Constructor Params ┃
┡━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━┩
│ linear_encoder   │ nexuml_library.layers.feature...    │ input_dim, ...     │
└──────────────────┴─────────────────────────────────────┴────────────────────┘
```

- **Type Key** — the string used in `LayerSpec(type_key=...)` or `DataSpec(source_type=...)`
- **Module** — the Python module where the decorator was applied
- **Constructor Params** — inferred from the class `__init__` signature

## Troubleshooting missing elements

### Element not shown

1. Check the library is loaded:

   ```bash
   nexuml library list
   nexuml registry list layers --verbose
   ```

2. Confirm the library root is registered (for local libraries):

   ```bash
   nexuml library add /path/to/my_library
   ```

3. Confirm the class/function is decorated:

   ```python
   from nexuml.core.discovery import layer

   @layer("my_layer")   # ← required
   class MyLayer(PipelineLayer): ...
   ```

4. Confirm the module is importable. Import errors are captured during discovery and shown in `--verbose` output. Common causes:
   - Missing optional dependency
   - Syntax error in the module
   - Circular import

### Element shown but ignored at runtime

If a layer appears in `registry list` but `nexuml resolve` complains it is unknown, the registry was populated in a different process. Discovery runs fresh per process — no cache.

### Empty registry

If the registry is completely empty, `nexuml_library` is likely not installed. Install the base library from the repository (see [Install NexuML](../start/install.md)).

## Discovery is per-run

NexuML rescans all packages on every invocation. There is no persistent discovery cache. Adding a new element to an already-registered library package makes it available on the next run without any `library add` command.

## Implementation map

- `src/nexuml/core/discovery.py` — `scan_all`, `Scanner`, `DiscoveryError`
- `src/nexuml/core/registry.py` — layer registry
- `src/nexuml/core/scenario_registry.py` — scenario registry
- `src/nexuml/cli/main.py` — `registry list` command

## See also

- [Discovery decorators](decorators.md)
- [Library discovery](../explanation/library-discovery.md)
- [Managing local roots](../how-to/library-cli.md)
