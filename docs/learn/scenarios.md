# Scenarios

A scenario is a Python function that returns a `ScenarioSpec`. Python scenarios use concrete, typed component definitions directly. Registry names appear only when a resolved scenario is serialized to YAML.

## Example

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import LayerSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml_library.layers.model.linear_encoder import LinearEncoder
from nexuml_library.scenarios.data.synthetic import synthetic_vector_data


@scenario("my-autoencoder")
def my_autoencoder() -> ScenarioSpec:
    return ScenarioSpec(
        name="my-autoencoder",
        data=synthetic_vector_data(feature_shape=(64,), num_samples=500),
        pipeline=PipelineSpec(
            stages={
                "encode": [
                    LayerSpec(
                        component=LinearEncoder(hidden_dims=[32], output_dim=8),
                        keys_in=["features"],
                        keys_out=["latent"],
                    )
                ]
            }
        ),
        training=TrainingSpec(max_epochs=10),
    )
```

`LinearEncoder(...)` owns component-specific configuration. `LayerSpec` owns graph placement such as `keys_in` and `keys_out`. The compiler supplies inferred shapes and other runtime values through `LayerBuildContext`.

## Data

`DataSpec` accepts typed data definitions either as `source` or in `datasets`:

```python
from nexuml.core.types import DataSpec, DatasetSpec
from nexuml_library.data.synthetic import SyntheticDataset

data = DataSpec(
    source=SyntheticDataset(feature_shape=(64,), num_samples=500),
    input_shapes={"features": [64]},
)

multi_data = DataSpec(
    datasets=[DatasetSpec(source=SyntheticDataset(feature_shape=(64,)))],
    input_shapes={"features": [64]},
)
```

## Evaluation And Loading

Evaluation algorithms and loader backends are also typed values:

```python
from nexuml.core.types import EvalAlgorithmSpec, EvaluationSpec, LoaderSpec
from nexuml.data.loaders.definitions import TorchLoader
from nexuml_library.evaluation.anomalous_sound_detection.asd_evaluator import AnomalyEvaluator

loader = LoaderSpec(backend=TorchLoader(), batch_size=32)
evaluation = EvaluationSpec(
    algorithms=[EvalAlgorithmSpec(algorithm=AnomalyEvaluator(), label_key="y_true")]
)
```

## Python And YAML

Python uses concrete classes for navigation, validation, and schemas. `ResolvedConfig.to_yaml()` lowers each definition to its registered identity:

```yaml
component:
  type: LinearEncoder
  version: '1'
  params:
    hidden_dims: [32]
    output_dim: 8
```

`ResolvedConfig.from_yaml()` discovers the registered identity and restores the concrete definition before validation.

## Running

```bash
nexuml resolve my-autoencoder
nexuml build configs/my-autoencoder.yaml
nexuml train my-autoencoder
```

Local trusted files can expose `scenario() -> ScenarioSpec` and run with `nexuml train --scenario-file scenario.py`.

## See Also

- [Define a scenario](../how-to/define-scenario.md)
- [Decorators and discovery](decorators-and-discovery.md)
- [`ScenarioSpec` reference](../reference/scenario-spec.md)
