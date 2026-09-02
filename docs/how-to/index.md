# Guides

Guides answer a specific "how do I ...?" question. They are not intended to be read in order. If you are learning NexuML for the first time, use the [Tutorials](../tutorials.md) instead.

## Core workflow

- [CLI lifecycle](cli-lifecycle.md) — how resolve, build, train, export, and smoke fit together.

## Scenarios and configuration

- [Define a scenario](define-scenario.md) — compose data, pipeline, training, and evaluation.
- [Run scenarios](run-scenarios.md) — registered scenarios, resolved YAML, and trusted Python files.
- [Trusted scenario files](scenario-file.md) — local Python experiment files and provenance snapshots.

## Training and evaluation

- [Train a model](train.md) — run the Lightning lifecycle and apply common overrides.
- [Evaluate a model](evaluate.md) — pipeline metrics, post-train layers, and evaluation algorithms.
- [Checkpoints](checkpoints.md) — distinguish Lightning resume from selective weight loading.
- [Automatic batch size](auto-batch-size.md) — probe CUDA batch sizes at runtime.

## Data

- [Choose a data loader](data-loading.md) — Torch, DALI, and tensor-shard loading.
- [Export a dataset](export-dataset.md) — persist raw or partially processed dataset views for reuse.

## Experimentation

- [Tracking and logging](tracking.md) — TensorBoard, MLflow, DVCLive, and diagrams.
- [Optuna tuning](tune.md) — hyperparameter and structural search.

## Model export

- [Export a model package](export.md) — package, reload, inference, and alternative weight formats.

## Execution

- [Execution modes](training-backends/index.md) — local vs Ray placement while keeping one NexuML lifecycle.
- [Ray execution](training-backends/ray.md) — existing clusters, Ray Jobs, distributed strategies, and shared data.

## Extending NexuML

- [Build a custom library](custom-library.md) — package structure and the definition/runtime split.
- [Add a custom layer](custom-layer.md) — direct `nn_module(...)` vs registered `LayerDefinition`.
- [Add a custom data source](custom-data-source.md) — `DataSourceDefinition` → `NexuDataset`.
- [Add a custom eval algorithm](custom-eval-algorithm.md) — `EvalAlgorithmDefinition` → `EvalAlgorithm`.
- [Register a library](register-library.md) — distribute components through the `nexuml.libraries` entry point.
- [Manage local library roots](library-cli.md) — develop a library without installing it first.

For exact command flags and Pydantic/API fields, use [Reference](../reference/index.md) rather than copying reference tables into task guides.
