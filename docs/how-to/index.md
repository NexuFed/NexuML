# How-to guides

Task-oriented recipes for practitioners who know what they want to achieve. Each page follows a consistent structure: prerequisites, runnable example, expected output, and links to reference.

## Core workflow

- [CLI lifecycle](cli-lifecycle.md) — the full resolve → build → train → export command flow
- [Run scenarios](run-scenarios.md) — all ways to execute a scenario (name, config, scenario file)
- [Train a model](train.md) — run Lightning training with options and overrides
- [Evaluate a model](evaluate.md) — post-training evaluation algorithms
- [Export a model package](export.md) — create and reload portable inference packages
- [Checkpoints](checkpoints.md) — resume, fine-tune, and selective checkpoint loading

## Scenario authoring

- [Define a scenario](define-scenario.md) — compose data, model, training, and evaluation specs
- [Trusted scenario files](scenario-file.md) — local Python file workflow and security notes
- [Write a custom composed scenario](custom-scenario.md) — full tutorial

## Data

- [Export a dataset](export-dataset.md) — export feature/label views to disk

## Library extension

- [Add a custom layer](custom-layer.md) — implement and register a `PipelineLayer` with `@layer`
- [Add a custom data source](custom-data-source.md) — implement and register a dataset with `@data_source`
- [Add a custom eval algorithm](custom-eval-algorithm.md) — implement and register eval logic with `@eval_algorithm`
- [Register a library](register-library.md) — distribute via `nexuml.libraries` entry-point
- [Manage local library roots](library-cli.md) — `nexuml library add/delete/list` for development
- [Custom library end-to-end](custom-library.md) — full tutorial building a library package

## Optimization

- [Optuna tuning](tune.md) — hyperparameter search with `nexuml tune`
- [Automatic batch size](auto-batch-size.md) — CUDA memory probing for optimal batch size

## Advanced operations

- [Tracking and logging](tracking.md) — TensorBoard, DVCLive, MLflow integration
- [Autoresearch](autoresearch.md) — Claude-driven iterative experiment loops
- [Docker & Kubernetes](deploy.md) — containerised training and scheduling
