<div align="center">

<img alt="NexuML" src="https://www.nexufed.ai/assets/logo-long-Ceach6Dp.png" width="800px" style="max-width: 100%;">

<br/>

<img alt="Python" src="https://img.shields.io/badge/python-3.12%2B-blue?style=flat-square&logo=python">
<img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-compatible-ee4c2c?style=flat-square&logo=pytorch&logoColor=white">
<img alt="Status" src="https://img.shields.io/badge/status-alpha-orange?style=flat-square">
<img alt="PyPI" src="https://img.shields.io/pypi/v/nexuml?style=flat-square&logo=pypi">

<br/>
<br/>

**Composable ML pipelines for reproducible experiments.**

NexuML is a modular PyTorch framework for building machine-learning systems from reusable, typed components connected through explicit TensorDict keys. A `ScenarioSpec` describes data, model pipelines, training, evaluation, logging, export, and execution in one place and can be persisted as validated YAML.

[Documentation](https://nexufed.github.io/NexuML/) · [Hands-on tutorials](https://github.com/NexuFed/NexuMLTutorial) · [PyPI](https://pypi.org/project/nexuml/)

</div>

## Why NexuML?

ML projects often start as a clean notebook and gradually collect project-specific dataset code, model wiring, training loops, evaluation scripts, and export logic. Copying a project template improves the folder structure, but it still duplicates implementations and makes experiments harder to reproduce or extend.

NexuML separates **reusable implementations** from **experiment composition**. Datasets, model blocks, evaluation algorithms, and loader backends can live in libraries; scenarios assemble those pieces into explicit pipelines without reimplementing the training lifecycle for every project.

## Core ideas

- **Typed component definitions** — Python scenarios construct concrete Pydantic definitions directly; stable registry identities are used for discovery and persisted YAML.
- **TensorDict pipelines** — named tensors flow through ordered stages using explicit `keys_in` and `keys_out` contracts.
- **Declarative scenarios** — `ScenarioSpec` composes data, pipeline, training, evaluation, logging, checkpoint, export, and execution configuration.
- **One Lightning lifecycle** — PyTorch Lightning owns the training loop; local and Ray execution reuse the same NexuML session semantics.
- **Pluggable data paths** — PyTorch, NVIDIA DALI, and tensor-shard loaders plus dataset export to NumPy, mmap, Torch, TensorDict memmap, WebDataset, and tensor shards.
- **Post-training evaluation** — typed evaluation definitions materialize stateful algorithms while fitted pipeline layers can perform post-train processing before test.
- **Portable model artifacts** — export a compiled pipeline with weights, resolved configuration, metadata, and dependency information.
- **CLI workflow** — inspect registries and backends, resolve scenarios, build pipelines, train, tune, export datasets, and package models.

## Install

For most users, install the framework together with the reusable base library:

```bash
uv pip install "nexuml[library]"
```

Install only the framework and CLI when you want to provide all components yourself:

```bash
uv pip install nexuml
```

NVIDIA DALI, Ray, tracking, tuning, S3, and export integrations are optional. See the [installation guide](https://nexufed.github.io/NexuML/start/install/) before adding platform-specific extras.

## First look

With the base library installed, inspect a real scenario without starting a training job:

```bash
nexuml registry list scenarios
nexuml resolve cifar-resnet
nexuml build configs/cifar-resnet.yaml
```

This shows the central NexuML flow: a Python scenario is resolved to a reproducible configuration and then materialized into a validated TensorDict pipeline. Continue with [Get started](https://nexufed.github.io/NexuML/start/) for training requirements.

## Learn by building

The [NexuML Tutorial repository](https://github.com/NexuFed/NexuMLTutorial) is the home for complete hands-on projects. It builds an external NexuML library from scratch rather than hiding the framework behind finished built-in components.

The learning path starts with MNIST library basics, then adds file-backed Speech Commands audio with native DALI loading and demonstrates pipeline composition by swapping a CNN encoder for a Transformer while reusing the rest of the system.

> **Version note:** the tutorial repository evolves independently from NexuML. NexuML 0.2 uses typed component definitions and rejects the legacy selector/parameter-bag syntax, so use a tutorial revision compatible with the NexuML version you install.

## Documentation

- **[Get started](https://nexufed.github.io/NexuML/start/)** — install NexuML and inspect your first scenario.
- **[Tutorials](https://nexufed.github.io/NexuML/tutorials/)** — complete hands-on projects and the tutorial compatibility note.
- **[Guides](https://nexufed.github.io/NexuML/how-to/)** — accomplish a specific task such as training, tuning, exporting, or adding a component.
- **[Concepts](https://nexufed.github.io/NexuML/explanation/)** — understand the architecture, TensorDict data flow, definitions, discovery, and scenarios.
- **[Reference](https://nexufed.github.io/NexuML/reference/)** — exact CLI, configuration, backend, decorator, and Python API information.

## Extending NexuML

External libraries can provide their own typed layers, data sources, evaluation algorithms, loader backends, and scenarios through the `nexuml.libraries` entry-point group or a local library root. See [Build a custom library](https://nexufed.github.io/NexuML/how-to/custom-library/) for the package structure and component contracts.
