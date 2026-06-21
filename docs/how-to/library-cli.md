# Managing local library roots

`nexuml library` manages a persistent list of local Python package roots that NexuML discovers at startup. Use this when developing a library that is not yet installed as a package.

## Prerequisites

- NexuML installed (`uv sync`)
- A local library directory with a Python package (containing `__init__.py`)

## Commands

### Add a local root

```bash
nexuml library add /path/to/my_library
```

NexuML stores the absolute path in `~/.config/nexuml/libraries.json`. On every subsequent run, the path is added to `sys.path` and the package is scanned for decorated elements.

### List current roots

```bash
nexuml library list
```

Shows installed entry-point libraries and configured local roots.

### Remove a root

```bash
nexuml library delete /path/to/my_library
```

Removes the path from `~/.config/nexuml/libraries.json`. The library is no longer loaded in future runs.

## Discovery behavior

Local roots are rescanned on every NexuML run — there is no persistent discovery cache. If you add a new layer to an existing local library, it appears in the registry immediately on the next run without running `library add` again.

## Verifying after adding

```bash
nexuml library list
nexuml registry list layers      # shows layers from the local library
nexuml registry list scenarios   # shows scenarios from the local library
```

## Example — developing a custom library

```bash
# Scaffold a library
mkdir -p my_library/src/my_library/layers
touch my_library/src/my_library/__init__.py
touch my_library/src/my_library/layers/__init__.py

# Register it
nexuml library add my_library

# Verify
nexuml library list
nexuml registry list layers --verbose
```

## Config file location

Local roots are stored at `~/.config/nexuml/libraries.json`:

```json
{
  "roots": [
    "/home/user/projects/my_library"
  ]
}
```

## Implementation map

- `src/nexuml/core/discovery.py` — `LibraryConfig`, `discover_local_packages`
- `src/nexuml/cli/main.py` — `library add`, `library delete`, `library list` commands

## See also

- [Register a library](register-library.md) — entry-point distribution
- [Library discovery](../explanation/library-discovery.md)
- [Registry inspection](../reference/registry.md)
