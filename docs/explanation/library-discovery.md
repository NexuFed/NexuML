# Library discovery

NexuML discovers layers, data sources, scenarios, and evaluation algorithms at startup by scanning Python packages for objects decorated with `@layer`, `@data_source`, `@scenario`, or `@eval_algorithm`. There is no persistent discovery cache — every NexuML process performs a fresh scan.

## Discovery sources

NexuML scans three sources in order:

1. **Built-in library** — `nexuml_library`, if installed. This is the public allow-list of open, reusable components.
2. **Entry-point packages** — packages that declare a `nexuml.libraries` entry point
3. **Local roots** — directories registered with `nexuml library add`

Proprietary components are not in `nexuml_library`; they live in the internal package at `external/`.

```
nexuml starts
    │
    ▼
scan nexuml_library (if installed)
    │
    ▼
scan nexuml.libraries entry points
    │
    ▼
scan local roots (~/.config/nexuml/libraries.json)
    │
    ▼
registries populated (layers, data sources, scenarios, eval algorithms)
    │
    ▼
nexuml resolve my-scenario → resolved ✓
```

## How scanning works

`scan_all()` in `nexuml.core.discovery` walks each package with `pkgutil.walk_packages`, imports every module, and inspects it for objects carrying the `__nexuml_discovered__` attribute (set by the decorators). Import failures are recorded as `DiscoveryError` entries and never abort the scan — one broken module does not hide everything else.

## Entry-point discovery

A library package advertises itself via the `nexuml.libraries` entry-point group in `pyproject.toml`:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

The entry-point value must be the importable package name. NexuML scans the whole package tree. There is no `register()` function requirement — decorators on classes/functions inside the package are sufficient.

Install the library:

```bash
uv pip install --link-mode=copy -e /path/to/my_library
```

Verify:

```bash
nexuml library list
nexuml registry list layers
```

## Local root discovery

For libraries not yet installable as packages:

```bash
nexuml library add /path/to/my_library
```

Roots are stored in `~/.config/nexuml/libraries.json`. The library must contain a Python package (a directory with `__init__.py`) either at the root or under a `src/` subdirectory.

See [Managing local library roots](../how-to/library-cli.md).

## No persistent cache

Discovery runs fresh on every NexuML invocation. If you add a new layer to an already-registered library, it is available immediately on the next run without re-running any `library add` command.

## Resilience

- A module that fails to import is recorded as a `DiscoveryError` and skipped.
- A duplicate key within the same registry kind raises a `ValueError` at registration time.
- Use `nexuml registry list layers --verbose` to see import failures.

## Implementation map

- `src/nexuml/core/discovery.py` — `scan_all`, `Scanner`, decorators, `LibraryConfig`, `discover_library_packages`
- `src/nexuml/core/registry.py` — layer registry
- `src/nexuml/core/scenario_registry.py` — scenario registry

## See also

- [Discovery decorators](../reference/decorators.md)
- [Add a custom layer](../how-to/custom-layer.md)
- [Register a library](../how-to/register-library.md)
- [Managing local roots](../how-to/library-cli.md)
- [Registry inspection](../reference/registry.md)
