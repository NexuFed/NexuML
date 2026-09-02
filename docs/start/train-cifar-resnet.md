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

## 4. Train when the configured loader is available

Training uses the loader selected by the scenario's `DataSpec`. In the current 0.2 configuration, `LoaderSpec` defaults to the DALI definition, so install DALI on a compatible Linux environment before training a scenario that leaves the loader implicit:

```bash
uv pip install "nexuml[dali]" --index https://pypi.nvidia.com
python -c "import nvidia.dali"
```

Then run a short training job:

```bash
nexuml train cifar-resnet --max-epochs 1
```

NexuML uses the same `NexuSession` lifecycle for the run: fit → validate → post-train fitting → test. Logging, checkpoints, and model exports are controlled by the scenario instead of by hard-coded quickstart paths.

For your own portable scenarios, explicitly select `TorchLoader()` unless you intend to require DALI. See [Data loading](../how-to/data-loading.md).

## What you learned

- libraries expose discoverable scenario recipes and typed component definitions;
- `ScenarioSpec` composes the complete experiment;
- `resolve` creates a reproducible persisted configuration;
- `build` materializes and validates the TensorDict pipeline;
- `train` delegates the actual training lifecycle to Lightning.

## Next

Do not continue by reading every reference page. Build something yourself in the [NexuML Tutorials](../tutorials.md), then return to the [Guides](../how-to/index.md) for individual tasks.
