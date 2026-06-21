<div align="center">

<img alt="NexuML" src="https://www.nexufed.ai/assets/logo-long-Ceach6Dp.png" width="800px" style="max-width: 100%;">

<br/>

<img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square&logo=python">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-compatible-ee4c2c?style=flat-square&logo=pytorch&logoColor=white">
<img alt="Status" src="https://img.shields.io/badge/status-preview-orange?style=flat-square">
<img alt="Install" src="https://img.shields.io/badge/install-from%20GitHub-black?style=flat-square&logo=github">

<!-- <img alt="Tests" src="https://img.shields.io/github/actions/workflow/status/NexuFed/NexuML/tests.yml?branch=main&style=flat-square&label=tests">
<img alt="Docs" src="https://img.shields.io/badge/docs-online-blue?style=flat-square">
<img alt="PyPI" src="https://img.shields.io/pypi/v/YOUR_PACKAGE_NAME?style=flat-square&logo=pypi"> -->

<br/>
<br/>

**The pipeline-based PyTorch framework.**

A modular deep learning pipeline framework built on PyTorch Lightning and TensorDict. Define training scenarios declaratively, compile them into type-safe pipelines, train with Lightning, and export portable model packages — all from a single composable API.

</div>

# NexuML

[Documentation](https://nexufed.github.io/NexuML/)

## Features

- **Registry-based layer discovery** — layers self-register by decorator; the compiler resolves them by key
- **TensorDict pipelines** — all data flows as named tensors through staged forward passes
- **Declarative scenarios** — compose data, model, training, and evaluation specs in pure Python
- **YAML config export/reload** — full pipeline reproducibility without pickle
- **Lightning training** — gradient accumulation, mixed precision, callbacks, all built in
- **Portable export** — ship a directory with `state_dict.pt + config.yaml + metadata.json`
- **CLI** — resolve, build, train, tune, export, and smoke-test from the terminal
- **Automatic pipeline diagrams** — `nexuml build` generates a Mermaid flowchart of your pipeline architecture

## Install

```bash
uv pip install --link-mode=copy -e ".[dev,all]"
uv pip install --link-mode=copy -e "./library"
```

Or with uv sync:

```bash
uv sync --all-extras
source .venv/bin/activate
```

### Serve the docs locally
```bash
uv run mkdocs serve
```

## Public library allow-list

The public `nexuml_library` package contains only the open, reusable core: public data loaders, layers, training defaults, and the scenarios under `scenarios/vision/`, `scenarios/asd/`, and `scenarios/tune/`.

Add your own libraries to the `external` folder to be also available in the automatic docs.

## Quickstart

```bash
export NEXUML_DATA_ROOT=/mnt/local
export NEXUML_LOGS_ROOT=/mnt/logs    # optional

scenario=cifar-resnet
nexuml resolve $scenario                  # compile spec → configs/$scenario.yaml
nexuml build configs/$scenario.yaml       # inspect layers, shapes, parameter counts
nexuml train $scenario --max-epochs=50
nexuml export $scenario --checkpoint      # portable state_dict.pt + config.yaml
```

## Roadmap

- Harden and improve the codebase
- Support distributed training backends
- Add a UI
- Extend the library