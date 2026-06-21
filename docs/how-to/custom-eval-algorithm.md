# Add a custom eval algorithm

This guide walks through adding a custom post-training evaluation algorithm to a library and using it from an `EvaluationSpec`.

## Prerequisites

- NexuML installed
- A library root registered with `nexuml library add` or an installed entry-point package
- For a new library: see [Register a library](register-library.md)

## 1. Create the eval-algorithm file

Place the algorithm under `evaluation/` inside your library package:

```
my_library/
└── src/
    └── my_library/
        ├── __init__.py
        └── evaluation/
            ├── __init__.py
            └── my_eval.py
```

```python
# my_library/src/my_library/evaluation/my_eval.py
import torch
from tensordict import TensorDict

from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm


@eval_algorithm("l2_error")
class L2ErrorEval(EvalAlgorithm):
    """Compute mean L2 error between a feature key and a prediction key."""

    def __init__(self, feature_key: str = "features", prediction_key: str = "reconstruction"):
        self.feature_key = feature_key
        self.prediction_key = prediction_key
        self._sum = 0.0
        self._count = 0

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        pred = x[self.prediction_key]
        target = x[self.feature_key]
        diff = (pred - target).flatten(start_dim=1)
        self._sum += torch.norm(diff, dim=1).sum().item()
        self._count += diff.shape[0]

    def results(self) -> dict[str, float]:
        return {"l2_error": self._sum / max(1, self._count)}
```

## 2. Register the local root or install the package

For local development:

```bash
nexuml library add my_library
```

For installable packages, declare the entry point:

```toml
[project.entry-points."nexuml.libraries"]
my-library = "my_library"
```

## 3. Verify registration

```bash
nexuml registry list eval
```

You should see `l2_error` in the output.

## 4. Use in a scenario

Reference the algorithm by `type` in an `EvalAlgorithmSpec`:

```python
from nexuml.core.discovery import scenario
from nexuml.core.types import (
    ScenarioSpec,
    DataSpec,
    TrainingSpec,
    PipelineSpec,
    LayerSpec,
    EvaluationSpec,
    EvalAlgorithmSpec,
)

@scenario("eval_tutorial")
def eval_tutorial() -> ScenarioSpec:
    return ScenarioSpec(
        name="eval_tutorial",
        data=DataSpec(source_type="synthetic", params={"feature_shape": [64], "num_samples": 500}),
        training=TrainingSpec(
            lr=1e-3,
            max_epochs=5,
            batch_size=64,
            loss_keys={"reconstruction_loss": 1.0},
        ),
        pipeline=PipelineSpec(stages={
            "encode": [
                LayerSpec(
                    type_key="linear_encoder",
                    keys_in=["features"],
                    keys_out=["z"],
                    params={"input_dim": 64, "output_dim": 8},
                ),
            ],
            "decode": [
                LayerSpec(
                    type_key="linear_decoder",
                    keys_in=["z"],
                    keys_out=["reconstruction"],
                    params={"input_dim": 8, "output_dim": 64},
                ),
            ],
            "loss": [
                LayerSpec(
                    type_key="reconstruction_loss",
                    keys_in=["reconstruction", "features"],
                    keys_out=["reconstruction_loss"],
                    params={"input_dim": 64},
                ),
            ],
        }),
        evaluation=EvaluationSpec(
            algorithms=[
                EvalAlgorithmSpec(
                    type="l2_error",
                    params={"feature_key": "features", "prediction_key": "reconstruction"},
                ),
            ],
            test_result_metrics=["l2_error"],
        ),
    )
```

## Eval-algorithm contract

- Inherit from `EvalAlgorithm`.
- Implement `results() -> dict[str, float]`.
- Optionally override `fit_batch`, `fit_end`, `eval_batch`, and `eval_end` to accumulate statistics across batches.
- Read from the pipeline output `TensorDict` (`x`) and labels (`y`).
- Return result keys that can be surfaced via `evaluation.test_result_metrics` for tuning or logging.

## See also

- [Discovery decorators](../reference/decorators.md)
- [Register a library](register-library.md)
- [Custom library end-to-end tutorial](custom-library.md)
