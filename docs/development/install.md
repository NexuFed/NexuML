# Development install

Use this guide if you are contributing to NexuML or want to modify its source code. Normal users should use the [standard install](../start/install.md) instead.

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- Git

## Clone and install

```bash
git clone https://github.com/NexuFed/NexuML.git
cd NexuML

# Install all dependencies including dev extras
uv sync --all-extras
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
ruff check src/ tests/
ruff format src/ tests/
```

## Serve docs locally

```bash
DISABLE_MKDOCS_2_WARNING=true mkdocs serve
```

## Dependency management

```bash
uv add <package>        # add runtime dependency
uv add --dev <package>  # add dev dependency
uv sync                 # install all deps from uv.lock
uv lock                 # regenerate uv.lock
```
