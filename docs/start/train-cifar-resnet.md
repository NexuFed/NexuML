# Your first NexuML scenario

The goal of this page is to show the core lifecycle without introducing custom component code yet. The `cifar-resnet` scenario comes from the optional `nexuml-library` package.

**Prerequisite:** install `nexuml[library]` as described in [Installation](install.md).

## 1. Discover scenarios

```bash
nexuml registry list scenarios
```

`cifar-resnet` should appear in the output. Scenario functions are recipes discovered from installed libraries; normal Python composition still imports concrete component definitions directly.

## 2. Resolve the scenario

```bash
nexuml resolve cifar-resnet
```

This evaluates the Python scenario, validates its `ScenarioSpec`, compiles its component graph, and writes the stable configuration to:

```text
configs/cifar-resnet.yaml
```

The YAML contains stable component identities and validated parameters rather than live Python objects.

## 3. Build the pipeline

```bash
nexuml build configs/cifar-resnet.yaml
```

`build` restores the typed definitions, materializes their runtime objects, propagates TensorDict shapes through the pipeline, and reports the compiled stages. If diagram output is enabled by the scenario, it also writes the configured Mermaid diagram.

At this point you have exercised the most important architectural boundary without starting a training job:

```text
Python ScenarioSpec → resolved YAML → typed definitions → CompiledPipeline
```

## 4. Train the scenario

Training uses the loader selected by the scenario's `DataSpec`. `LoaderSpec` defaults to the portable PyTorch loader, so the standard library installation is enough for scenarios that leave the loader implicit. Run a short training job:

```bash
nexuml train cifar-resnet --max-epochs 1
```

NexuML uses the same `NexuSession` lifecycle for the run: fit → validate → post-train fitting → test. Logging, checkpoints, and model exports are controlled by the scenario instead of by hard-coded quickstart paths. When a checkpoint callback omits `dirpath`, Lightning writes checkpoints under the active logger's run directory, or under the trainer's `default_root_dir` when no logger is configured.

Scenarios that need NVIDIA DALI must select `DaliLoader()` explicitly and install the optional integration. See [Data loading](../how-to/data-loading.md) and [Checkpoints](../how-to/checkpoints.md).

## What you learned

- libraries expose discoverable scenario recipes and typed component definitions;
- `ScenarioSpec` composes the complete experiment;
- `resolve` creates a reproducible persisted configuration;
- `build` materializes and validates the TensorDict pipeline;
- `train` delegates the actual training lifecycle to Lightning.

## Next

Do not continue by reading every reference page. Build something yourself in the [NexuML Tutorials](../tutorials.md), then return to the [Guides](../how-to/index.md) for individual tasks.
