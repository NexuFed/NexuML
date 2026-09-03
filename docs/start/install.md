# Install NexuML

## Prerequisites

- Python 3.12+
- `uv` or another Python package installer

You do **not** need to clone the NexuML repository to use it.

## Recommended installation

Install NexuML together with the reusable base library used by the built-in examples:

```bash
uv venv
source .venv/bin/activate
uv pip install "nexuml[library]"
```

This installs two distributions:

- `nexuml` — the framework, CLI, compiler, data/runtime infrastructure, evaluation runtime, and integrations.
- `nexuml-library` — reusable datasets, model/loss/metric components, evaluation definitions, and example scenarios discovered through the `nexuml.libraries` entry point.

Verify the installation:

```bash
nexuml --help
nexuml registry list scenarios
nexuml backend list
```

## Core-only installation

If your project supplies all library components itself, install only the framework:

```bash
uv pip install nexuml
```

A core-only environment intentionally does not contain the built-in `nexuml_library` scenarios.

## Optional integrations

Core integrations are installed only when requested. Examples:

```bash
uv pip install "nexuml[tracking]"
uv pip install "nexuml[tuning]"
uv pip install "nexuml[export]"
uv pip install "nexuml[ray]"
uv pip install "nexuml[s3]"
```

`nexuml[all]` installs the normal user-facing core integrations and the base library, but intentionally excludes development tooling and DALI.

The base library has additional feature extras for optional datasets/models/evaluation tooling when a component needs them, for example `nexuml-library[audio]`, `nexuml-library[data]`, `nexuml-library[pretrained]`, and `nexuml-library[eval]`.

## Container image

The GHCR image provides the complete development and training environment when you do not want to assemble the Python and CUDA dependencies yourself. Pull a release using its explicit NexuML and CUDA versions:

```bash
docker pull ghcr.io/nexufed/nexuml:0.2.0-cuda12.8.1
```

Run it with NVIDIA GPU access:

```bash
docker run --rm -it --gpus all ghcr.io/nexufed/nexuml:0.2.0-cuda12.8.1
```

This Linux AMD64 image contains NexuML 0.2.0, Python 3.13, Ubuntu 24.04, CUDA 12.8.1 development tools, both `nexuml` and `nexuml-library`, and every optional extra from both workspace packages. It is substantially larger than the normal Python installation.

GPU execution requires a compatible NVIDIA host driver and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Building or starting the image without GPU access remains possible, but CUDA workloads require `--gpus all` or a narrower Docker GPU selection.

Release tags use `<nexuml-version>-cuda<cuda-version>`. `edge-cuda12.8.1` tracks `main`, while `sha-<short-sha>-cuda12.8.1` identifies one immutable source revision. Always specify a tag: NexuML does not publish an unqualified `latest` image. GHCR (`ghcr.io/nexufed/nexuml`) is the supported registry; Docker Hub mirroring is not enabled.

## CUDA and PyTorch

The default NexuML installation resolves PyTorch from the public package index. If you need a particular CUDA build, install the matching PyTorch packages using the official PyTorch installation instructions before installing NexuML.

NexuML does not encode a project-specific CUDA wheel index into the published package metadata.

## NVIDIA DALI

DALI is a separate Linux/platform-specific integration. Install it only on a compatible environment:

```bash
uv pip install "nexuml[dali]" --index https://pypi.nvidia.com
python -c "import nvidia.dali"
```

`nexuml backend list data-loader` lists the registered `DaliLoader` definition, but it is a **catalog command**, not an import/driver health check. A scenario that selects DALI validates the actual optional dependency when the loader runtime is built.

See [Data loading](../how-to/data-loading.md) for Torch/DALI/tensor-shard selection.

## Environment roots

Two optional environment variables provide convenient roots:

```bash
export NEXUML_DATA_ROOT=/path/to/datasets
export NEXUML_LOGS_ROOT=/path/to/logs
```

See [Environment roots](../reference/environment.md) for the exact resolution rules.

## Next

Continue with [Your first scenario](train-cifar-resnet.md).

If you plan to modify NexuML itself, use the [development install](../development/install.md) instead of the PyPI workflow above.
