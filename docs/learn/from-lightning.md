# Coming from PyTorch Lightning

NexuML does not replace Lightning's training loop. It adds a typed experiment description, TensorDict pipeline composition, reusable component libraries, persistence, and execution placement around it.

## Concept mapping

| PyTorch / Lightning | NexuML |
| --- | --- |
| `torch.nn.Module` blocks | runtime `PipelineLayer` objects, usually created from typed `LayerDefinition` values or `nn_module(...)` |
| one model graph | `CompiledPipeline` with ordered stages and explicit TensorDict key contracts |
| custom `LightningModule` | `NexuLightningModule`, created by `NexuSession` around the compiled pipeline |
| `LightningDataModule` | `NexuDataModule`, built from `DataSpec` and a typed loader backend |
| `Trainer(...)` training settings | `TrainingSpec` plus `CallbackSpec` / `LoggingSpec` |
| `ModelCheckpoint(...)` | `CallbackSpec(type="checkpoint", params={...})` |
| `Trainer.fit(..., ckpt_path=...)` | `nexuml train ... --trainer-checkpoint PATH` for full Lightning-state resume |
| loading pretrained weights | `CheckpointLoadSpec` for selective weight reuse/freeze policies |
| training script | `ScenarioSpec` + `nexuml train` |
| hand-maintained experiment YAML | `nexuml resolve` creates validated YAML from the Python scenario |

## The key difference

A Lightning project commonly wires model construction, data, trainer configuration, evaluation, and paths in one training script. NexuML makes those choices explicit and reusable:

```python
@scenario("my-experiment")
def my_experiment() -> ScenarioSpec:
    return ScenarioSpec(
        name="my-experiment",
        data=...,
        pipeline=...,
        training=...,
        evaluation=...,
        logging=...,
        callbacks=...,
        execution=...,
    )
```

The scenario is configuration, not the mutable runtime. Layer/data/evaluation definitions are materialized only when the pipeline or data runtime is built.

## What stays Lightning-native

- Lightning still performs fit, validation, test, precision/device handling, callbacks, logging integration, and checkpoint resume.
- `training.strategy` selects the Lightning strategy. Under Ray execution, NexuML maps supported values to Ray's official Lightning strategies rather than implementing its own distributed training loop.
- `CallbackSpec` can represent known callback aliases or importable Lightning callbacks.

## What NexuML adds

### Explicit TensorDict contracts

Instead of passing one positional tensor through an opaque model, pipeline blocks declare the named values they consume and produce:

```python
LayerSpec(
    component=MyEncoder(width=128),
    keys_in=["waveform"],
    keys_out=["embedding"],
)
```

This makes intermediate representations reusable by later heads, losses, metrics, evaluation, and export paths.

### Definitions instead of constructor bags

Python imports the actual component definition:

```python
component=MyEncoder(width=128)
```

The decorator identity is used for discovery and YAML restoration; normal Python composition does not look components up by string.

### One scenario across execution environments

`ScenarioSpec.execution` controls placement. Local execution and Ray execution use the same NexuML session lifecycle; execution is not a second model API.

## Checkpoints: do not mix these concepts

There are three separate operations:

1. **Create checkpoints** with a checkpoint `CallbackSpec`.
2. **Resume a Lightning run** with `--trainer-checkpoint`, restoring trainer/optimizer/scheduler state.
3. **Reuse model weights** with `CheckpointLoadSpec`, optionally selecting or freezing parameters without resuming the old trainer state.

See [Checkpoints](../how-to/checkpoints.md) for examples.

## Next

- [Mental model](mental-model.md)
- [Scenarios](scenarios.md)
- [Architecture](../explanation/architecture.md)
- [Train a model](../how-to/train.md)
