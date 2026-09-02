# Add A Custom Eval Algorithm

The public algorithm definition stores semantic configuration. A private `EvalAlgorithm` runtime accumulates mutable evaluation state.

```python
import torch
from tensordict import TensorDict

from nexuml.core.components import EvalAlgorithmDefinition, EvalBuildContext
from nexuml.core.discovery import eval_algorithm
from nexuml.evaluation.algorithm import EvalAlgorithm


@eval_algorithm("l2_error")
class L2Error(EvalAlgorithmDefinition):
    prediction_key: str = "reconstructed"

    def build(self, context: EvalBuildContext) -> EvalAlgorithm:
        return _L2ErrorRuntime(
            feature_key=context.feature_key or "features",
            prediction_key=self.prediction_key,
        )


class _L2ErrorRuntime(EvalAlgorithm):
    def __init__(self, feature_key: str, prediction_key: str):
        self.feature_key = feature_key
        self.prediction_key = prediction_key
        self.total = 0.0
        self.count = 0

    def eval_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        difference = (x[self.prediction_key] - x[self.feature_key]).flatten(start_dim=1)
        self.total += torch.norm(difference, dim=1).sum().item()
        self.count += difference.shape[0]

    def results(self) -> dict[str, float]:
        return {"l2_error": self.total / max(1, self.count)}
```

Use the definition directly while routing stays on `EvalAlgorithmSpec`:

```python
from nexuml.core.types import EvalAlgorithmSpec, EvaluationSpec
from my_library.evaluation.l2_error import L2Error

evaluation = EvaluationSpec(
    algorithms=[
        EvalAlgorithmSpec(
            algorithm=L2Error(prediction_key="reconstructed"),
            feature_key="features",
        )
    ],
    test_result_metrics=["l2_error"],
)
```

`enabled`, `name`, axis keys, feature keys, and label keys remain on the surrounding spec because they describe evaluation placement. Algorithm-specific values remain on the definition.

Verify discovery with `nexuml registry list eval`. YAML uses the stable `l2_error@1` identity and validated parameters, while Python imports `L2Error` directly.

## Runtime Contract

- `build(context)` returns an `EvalAlgorithm`.
- The runtime implements `results()`.
- Mutable accumulators stay on the runtime, not the definition.

## See Also

- [Discovery decorators](../reference/decorators.md)
- [Evaluate](evaluate.md)
- [Custom library](custom-library.md)
