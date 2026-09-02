# Define a scenario

A scenario is a Python recipe that returns a Pydantic `ScenarioSpec`. It composes the data, TensorDict pipeline, training/evaluation policy, and optional logging/export/execution settings for one experiment.

## Minimal registered scenario

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import DataSpec, LayerSpec, LoaderSpec, PipelineSpec, ScenarioSpec, TrainingSpec
from nexuml.data.loaders.definitions import TorchLoader
from nexuml_library.data.synthetic import SyntheticDataset
from nexuml_library.layers.loss.reconstruction_loss import ReconstructionLoss
from nexuml_library.layers.model.linear_encoder import LinearEncoder


@scenario("my-autoencoder")
def my_autoencoder() -> ScenarioSpec:
    return ScenarioSpec(
        name="my-autoencoder",
        data=DataSpec(
            source=SyntheticDataset(feature_shape=(64,), num_samples=1000),
            input_shapes={"features": [64]},
            loader=LoaderSpec(backend=TorchLoader()),
        ),
        pipeline=PipelineSpec(
            stages={
                "Encoder": [
                    LayerSpec(
                        component=LinearEncoder(hidden_dims=[32], output_dim=8),
                        keys_in=["features"],
                        keys_out=["latent"],
                    )
                ],
                "Decoder": [
                    LayerSpec(
                        component=LinearEncoder(hidden_dims=[32], output_dim=64),
                        keys_in=["latent"],
                        keys_out=["reconstructed"],
                    )
                ],
                "Loss": [
                    LayerSpec(
                        component=ReconstructionLoss(),
                        keys_in=["features", "reconstructed"],
                        keys_out=["reconstruction_loss"],
                    )
                ],
            }
        ),
        training=TrainingSpec(
            max_epochs=5,
            loss_keys={"reconstruction_loss": 1.0},
        ),
    )
```

Python uses concrete typed definitions directly. `LayerSpec` owns graph wiring; component definitions own component-specific semantic parameters.

`TorchLoader()` is selected explicitly here to make the scenario's portable loader contract visible; omitting the backend would select the same default. DALI scenarios must select `DaliLoader()` explicitly.

## Make the scenario discoverable

For an installed library, expose its package with the entry point:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

During local development you can instead register a path:

```bash
nexuml library add /path/to/my_library
```

## Verify the recipe

```bash
nexuml registry list scenarios
nexuml resolve my-autoencoder
nexuml build configs/my-autoencoder.yaml
nexuml train my-autoencoder
```

## What belongs on the scenario?

`ScenarioSpec` is the composition root. Major sections include pipeline, data, training, evaluation, logging, callbacks, tuning, checkpoint/weight-loading policy, exports, and execution placement.

Do not duplicate the full field table here. Use [Scenario and config reference](../reference/scenario-spec.md) and the generated Python API for exact fields/defaults.

## See also

- [Scenarios concept](../learn/scenarios.md)
- [Build a custom library](custom-library.md)
- [Trusted scenario files](scenario-file.md)
