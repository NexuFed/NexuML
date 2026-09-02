# Install NexuML

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed

## Install into your own project

You do **not** need to clone the NexuML repository to use it. Create or activate your project environment, then install the core framework from PyPI.

```bash
# Create and activate a virtual environment
uv venv
source .venv/bin/activate

# Install the core framework and CLI
uv pip install nexuml

# Optional: add bundled components and scenarios
uv pip install "nexuml[library]"
```

The core framework and CLI work without `nexuml-library`. Install the `library` extra when you want the bundled reusable layers, data sources, evaluation algorithms, and scenarios used by the tutorials.

## Verify the install

```bash
nexuml --help
```

You should see the `nexuml` command with subcommands `resolve`, `build`, `train`, `export`, `smoke`, `registry`, and others.

## Verify the base library scenarios

If you installed `nexuml[library]`, run:

```bash
nexuml registry list scenarios
```

You should see `cifar-resnet` and other scenarios from the base library.

## CUDA and DALI

The default installation resolves PyTorch from the public Python package index. For a specific CUDA build, install the matching PyTorch packages from the [official PyTorch selector](https://pytorch.org/get-started/locally/) before installing NexuML.

NVIDIA DALI is a separate, Linux-only extra served from NVIDIA's package index:

```bash
uv pip install "nexuml[dali]" --index https://pypi.nvidia.com
```

DALI is intentionally not included by `nexuml[all]` because its availability depends on the host platform and CUDA setup.

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

If you plan to modify NexuML itself, use the [development install](../development/install.md) for clone, `uv sync --all-extras`, and editable-library instructions.
