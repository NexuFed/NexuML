# Coming from Lightning

If you already use PyTorch Lightning, NexuML should feel familiar. This page maps Lightning concepts to NexuML equivalents.

## Concept mapping

| PyTorch Lightning | NexuML equivalent | Notes |
|---|---|---|
| `LightningModule` | `PipelineLayer` (one per stage) | NexuML composes many layers into a single compiled module |
| `LightningDataModule` | `DataSpec` + registered data source | Data source resolved by `source_type` key |
| `Trainer(...)` | `TrainingSpec` | Optimizer, scheduler, max epochs declared in spec |
| `ModelCheckpoint` callback | `CheckpointLoadSpec` | Resume from checkpoint using `--trainer-checkpoint` |
| Training script | `ScenarioSpec` + `nexuml train` | CLI replaces the training script |
| `pl.Trainer.fit(model, datamodule)` | `nexuml train cifar-resnet` | Same Lightning loop, NexuML wires the pieces |
| Experiment YAML / Hydra config | `nexuml resolve` → `configs/<name>.yaml` | Generated from Python spec, then version-controllable |

## What NexuML adds

NexuML does not replace Lightning — it wraps it. The Lightning `Trainer` still runs the training loop. What NexuML adds:

1. **A layer-contract system.** Each `LayerSpec` declares `keys_in` and `keys_out` as TensorDict keys. NexuML validates that upstream layers provide what downstream layers expect, catching shape and key errors before training begins.

2. **A registry.** Layers, data sources, scenarios, and eval algorithms are registered with decorators (`@layer`, `@data_source`, `@scenario`, `@eval_algorithm`). The registry is scanned from installed packages at runtime.

3. **Composable specs.** A `ScenarioSpec` assembles data, pipeline, training, evaluation, and export into one object that serialises to YAML.

4. **CLI first-class commands.** `resolve`, `build`, `train`, `export`, `smoke`, `tune` replace hand-written training scripts.

## What stays the same

- The Lightning training loop is unchanged.
- You can still use all Lightning callbacks, loggers, and plugins via `CallbackSpec` / `LoggingSpec`.
- PyTorch layer implementations (`nn.Module`) are unchanged; NexuML wraps them in a `PipelineLayer`.

## Translating a Lightning training script

**Before (plain Lightning):**

```python
model = ResNetClassifier(num_classes=10)
datamodule = CIFAR10DataModule(batch_size=64)
trainer = pl.Trainer(max_epochs=10)
trainer.fit(model, datamodule)
```

**After (NexuML scenario):**

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec

@scenario("cifar-resnet")
def cifar_resnet() -> ScenarioSpec:
    return ScenarioSpec(
        name="cifar_resnet",
        data=...,       # DataSpec referencing registered CIFAR data source
        pipeline=...,   # PipelineSpec with LayerSpec entries
        training=...,   # TrainingSpec with optimizer / max_epochs
        evaluation=..., # EvaluationSpec
    )
```

```bash
nexuml train cifar-resnet --max-epochs=10
```

The scenario function replaces the script; the CLI replaces `trainer.fit(...)`.

## Next

- [Mental model](mental-model.md) — the full resolve → build → train lifecycle
- [Scenarios](scenarios.md) — deep dive into `ScenarioSpec` fields
- [Decorators and discovery](decorators-and-discovery.md) — how the registry works
