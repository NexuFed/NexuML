# Install NexuML

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed

## Install into your own project

You do **not** need to clone the NexuML repository to use it. Install directly from GitHub into your own project environment.

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the core framework
uv pip install "git+https://github.com/NexuFed/NexuML.git"

# Install the base library (built-in scenarios and layers)
uv pip install "git+https://github.com/NexuFed/NexuML.git#subdirectory=library"
```

## Verify the install

```bash
nexuml --help
```

You should see the `nexuml` command with subcommands `resolve`, `build`, `train`, `export`, `smoke`, `registry`, and others.

## Verify the base library scenarios

```bash
nexuml registry list scenarios
```

You should see `cifar-resnet` and other scenarios from the base library.

## Set environment variables (optional)

```bash
export NEXUML_DATA_ROOT=/path/to/datasets
export NEXUML_LOGS_ROOT=/path/to/logs
```

When `NEXUML_LOGS_ROOT` is set, training logs and checkpoints are written there instead of `.experiments/` in the current directory.

## Next step

[Train CIFAR ResNet](train-cifar-resnet.md) — run your first model end-to-end.

---

## Development install

If you plan to modify NexuML itself, use the [development install](../development/install.md) instead.
