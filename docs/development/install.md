# Development install

Use this guide if you are contributing to NexuML or want to modify its source code. Normal users should use the [standard install](../start/install.md) instead.

## Prerequisites

- Python
- [uv](https://docs.astral.sh/uv/) installed
- Git

## Clone and install

```bash
git clone https://github.com/NexuFed/NexuML.git
cd NexuML

# Includes dev/docs tools; excludes DALI
uv sync --extra all --extra dev
source .venv/bin/activate

# Install the base library as an editable package
uv pip install --link-mode=copy -e "./library"
```

## Verify

```bash
nexuml --help
nexuml registry list scenarios
```

## Tests, linting, and type checking

```bash
# Run tests
pytest tests/ -v

# Type check
ty check

# Lint and format
ruff check .
ruff format .
```

## Serve docs locally

```bash
DISABLE_MKDOCS_2_WARNING=true mkdocs serve
```

## Dependency management

```bash
uv add <package>                 # runtime dependency
uv add --optional dev <package>  # development extra
uv sync --extra all --extra dev  # sync contributor dependencies
uv lock                         # regenerate the lockfile
```

## OpenChamber worktrees

After you trust [setup](https://docs.openchamber.dev/repository-config/), OpenChamber
waits for an editable worktree-local `.venv` (`all`/`dev`, including the base library)
and Studio's locked Node dependencies. Requires uv, Python, Node.js and npm; excludes
DALI and Playwright browser downloads. Activate `.venv` for terminal commands.

Find project actions in Work Status or the command palette. **Format Python** writes
files; check actions do not. Full tests need their GPU/data prerequisites.
Start **Dev backend**, then **Dev UI** at `http://127.0.0.1:3200`.
Ports 8000/3200 support one pair; change the actions for parallel worktrees.

## Studio development

Install the frontend dependencies:

```bash
npm --prefix studio ci
```

Follow [manual API startup](../how-to/local-api.md#start-manually), then in another terminal:

```bash
NEXUML_STUDIO_TOKEN="$(cat "$HOME/.nexuml-api-token")" \
NEXUML_STUDIO_API=http://127.0.0.1:8000 \
NEXUML_STUDIO_ORIGIN=http://127.0.0.1:3000 \
npm --prefix studio run dev -- --port 3000
```

Open `http://127.0.0.1:3000`. Keep credentials out of URLs and source files;
external proxies require [hosted development](../studio.md#hosted-development).

## Next

See [Development workflow](../how-to/development.md) for coverage, CI and package checks,
or [Installation](../start/install.md#nvidia-dali) for DALI.
