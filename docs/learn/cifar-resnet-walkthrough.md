# CIFAR ResNet walkthrough

This page walks through the `cifar-resnet` scenario — the canonical example in the base library — to show how a real scenario is structured and how each `ScenarioSpec` field connects to the CLI lifecycle.

## Where it lives

The scenario is defined in the base library:

```
library/src/nexuml_library/scenarios/vision/cifar_resnet.py
```

It is registered as `cifar-resnet` by the `@scenario` decorator.

## The scenario function

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import ScenarioSpec

@scenario("cifar-resnet")
def cifar_resnet(
    dataset: str = "cifar10",
    download: bool = True,
    resnet_type: str = "resnet18",
    pretrained: bool = False,
    lr: float = 1e-3,
    batch_size: int = 64,
    max_epochs: int = 10,
) -> ScenarioSpec:
    return ScenarioSpec(
        name="cifar_resnet",
        pipeline=resnet_classifier(num_classes=10, resnet_type=resnet_type, ...),
        training=default_training(lr=lr, batch_size=batch_size, max_epochs=max_epochs, ...),
        data=cifar10_data(download=download),
        evaluation=classification_evaluation(),
    )
```

The function accepts parameters with defaults. During `resolve`, NexuML calls the function with defaults (or overrides from `--override`) and serialises the resulting `ScenarioSpec` to YAML.

## Data

`cifar10_data(...)` returns a `DataSpec` with `source_type` set to the key registered by the CIFAR data source. The data source handles downloading, train/val/test splitting, and batching.

The resolved config stores all data parameters so training is reproducible from the YAML alone.

## Pipeline

`resnet_classifier(...)` returns a `PipelineSpec` whose `stages` list contains `LayerSpec` entries:

| `type_key` | `keys_in` | `keys_out` | Role |
|---|---|---|---|
| `resnet18` (or variant) | `{"x": "image"}` | `["features"]` | Backbone feature extraction |
| `classification-head` | `{"x": "features"}` | `["logits", "classification_loss"]` | Class scores + loss |

Each `type_key` resolves to a registered `@layer` class. `keys_in` and `keys_out` define the TensorDict key contracts that NexuML validates at `build` time.

## Training

`default_training(...)` returns a `TrainingSpec`:

- `loss_keys={"classification_loss": 1.0}` — the `classification_loss` TensorDict key, weighted 1.0
- `metric_keys=["accuracy", "f1"]` — metrics tracked during training and evaluation
- `max_epochs=10` — overridable with `--max-epochs` flag

## Evaluation

`classification_evaluation()` returns an `EvaluationSpec` that runs after training. It uses a registered evaluation algorithm that computes classification metrics on the test split.

## CLI lifecycle for this scenario

```bash
# Discover the scenario
nexuml registry list scenarios

# Resolve to config
nexuml resolve cifar-resnet
# → configs/cifar-resnet.yaml

# Build and validate the pipeline
nexuml build configs/cifar-resnet.yaml

# Train (2 epochs for quick demo)
nexuml train cifar-resnet --max-epochs=2
# → .experiments/lightning_logs/version_0/checkpoints/last.ckpt

# Export trained model
nexuml export cifar-resnet --checkpoint .experiments/lightning_logs/version_0/checkpoints/last.ckpt
# → exported_model/

# Or run the full pipeline as a smoke test
nexuml smoke cifar-resnet --max-epochs=2
```

## Override parameters without editing code

```bash
# Use ResNet-50 instead of ResNet-18
nexuml train cifar-resnet --max-epochs=5 --override resnet_type=resnet50

# Use CIFAR-100
nexuml train cifar-resnet --override dataset=cifar100
```

## What to read next

- [Mental model](mental-model.md) — the full lifecycle explained
- [Scenarios](scenarios.md) — how to write your own `ScenarioSpec`
- [Decorators and discovery](decorators-and-discovery.md) — how `@scenario` and `@layer` work
- [Define a scenario](../how-to/define-scenario.md) — step-by-step how-to for writing a scenario
- [Write a custom composed scenario](../how-to/custom-scenario.md) — full end-to-end tutorial
