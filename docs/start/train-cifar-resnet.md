# Train CIFAR ResNet

This tutorial walks you through the complete NexuML workflow using the built-in `cifar-resnet` scenario: resolve a config, build and inspect the pipeline, train with Lightning, and export a portable model package.

**Prerequisite:** [Install NexuML](install.md) first.

## 1. List available scenarios

```bash
nexuml registry list scenarios
```

You should see `cifar-resnet` in the list. This scenario is provided by the base library (`nexuml_library`) you installed.

**What this means:** NexuML discovers scenarios from installed packages. Each entry was registered with `@scenario("key")`. See [Decorators and discovery](../learn/decorators-and-discovery.md).

## 2. Set data root (CIFAR will download here)

```bash
export NEXUML_DATA_ROOT=~/nexuml-data
```

CIFAR-10 is about 170 MB. The first run downloads it; subsequent runs use the cache.

## 3. Resolve the scenario

```bash
nexuml resolve cifar-resnet
```

**What happens:** NexuML calls the `cifar-resnet` scenario function, which builds a `ScenarioSpec` object, then serialises it to a reproducible YAML config.

**Artifact:** `configs/cifar-resnet.yaml` — the resolved configuration for this scenario.

## 4. Build and inspect the pipeline

```bash
nexuml build configs/cifar-resnet.yaml
```

**What happens:** NexuML compiles each `LayerSpec` into a concrete `PipelineLayer`, validates tensor key contracts (`keys_in`/`keys_out`), and reports layer names, tensor shapes, and parameter counts.

**Artifact:** Pipeline summary printed to the terminal. If `logging.diagram` is enabled in the config, a Mermaid diagram is also written to your logs directory.

## 5. Train

```bash
nexuml train cifar-resnet --max-epochs=2
```

**What happens:** NexuML compiles the pipeline, wraps it in a PyTorch Lightning `LightningModule`, and runs the training loop. Progress and metrics appear in the terminal via Lightning's default progress bar.

**Artifacts:**
- Checkpoints under `.experiments/lightning_logs/version_0/checkpoints/`
  (or `$NEXUML_LOGS_ROOT/lightning_logs/...` if you set that variable)
- TensorBoard logs in the same directory

!!! tip "Use `--max-epochs=2` for the first run"
    CIFAR-10 trains well for demonstration at 2 epochs. Remove the flag or increase it for real experiments.

## 6. Export the model

After training, export the checkpoint to a portable package:

```bash
nexuml export cifar-resnet --checkpoint .experiments/lightning_logs/version_0/checkpoints/last.ckpt
```

**What happens:** NexuML reconstructs the pipeline from the scenario, loads the checkpoint weights, and writes a self-contained export package.

**Artifact:** `exported_model/` directory containing the pipeline and weights.

!!! note "Checkpoint path"
    Adjust the checkpoint path if Lightning created `version_1` or later. Check `.experiments/lightning_logs/` for the actual version directory.

## 7. Smoke test (optional)

The `smoke` command runs the full pipeline in a single call, useful for verifying an install:

```bash
nexuml smoke cifar-resnet --max-epochs=2
```

This does resolve → build → train → export → reload → infer in sequence and reports any failures.

## What you've learned

| Step | Command | Concept introduced |
|------|---------|-------------------|
| List scenarios | `registry list scenarios` | Discovery, registered scenarios |
| Resolve | `resolve cifar-resnet` | `ScenarioSpec` → YAML config |
| Build | `build configs/cifar-resnet.yaml` | Pipeline compilation, layer contracts |
| Train | `train cifar-resnet` | Lightning training loop |
| Export | `export cifar-resnet --checkpoint ...` | Portable model package |

## Next steps

- [Mental model](../learn/mental-model.md) — understand the full `ScenarioSpec` → resolve → build → train lifecycle
- [Scenarios](../learn/scenarios.md) — learn how scenarios are structured and how to write your own
- [CLI reference](../reference/cli.md) — all commands and flags
- [Export a model package](../how-to/export.md) — reload and run inference from the exported package
