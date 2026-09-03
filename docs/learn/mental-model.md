# Mental model

NexuML exists to keep reusable ML implementations separate from the glue code that defines one experiment. The easiest way to reason about it is as four boundaries.

## 1. Define

A Python scenario returns a Pydantic `ScenarioSpec`. It directly contains typed component definitions:

```python
LayerSpec(
    component=LinearEncoder(hidden_dims=[32], output_dim=8),
    keys_in=["features"],
    keys_out=["embedding"],
)
```

The component owns its semantic configuration. `LayerSpec` owns where that component sits in the TensorDict graph. The same pattern is used for data sources, evaluation algorithms, and loader backends.

## 2. Persist

```bash
nexuml resolve my-scenario
```

`resolve` writes a validated YAML representation. Concrete registered definitions are lowered to stable `type`, `version`, and `params` data so the experiment can be reviewed, versioned, and restored later.

Python code uses concrete classes; registry identities primarily matter at discovery and persistence boundaries.

## 3. Materialize

```bash
nexuml build configs/my-scenario.yaml
```

The compiler restores the concrete definitions and calls their explicit build boundaries:

- `LayerDefinition.build(context)` → `PipelineLayer`
- `DataSourceDefinition.build()` → `NexuDataset`
- `EvalAlgorithmDefinition.build(context)` → `EvalAlgorithm`
- `LoaderBackendDefinition.build()` → loader backend

For ordinary one-input/one-output PyTorch modules, `nn_module(...)` provides the generic layer definition instead of requiring a custom NexuML wrapper.

The result is a `CompiledPipeline` whose ordered stages exchange named values through a TensorDict.

## 4. Run

```bash
nexuml train my-scenario
```

`NexuSession` drives the standard lifecycle through PyTorch Lightning:

```text
fit → validate → fit post-train pipeline layers → test
```

Evaluation algorithms consume test outputs and finalize their metrics/artifacts. Scenario logging, callbacks, checkpoint loading, exports, and execution placement are applied around the same lifecycle.

Local and Ray execution change **where** the session runs; they do not define a second model/training implementation.

## The objects to remember

| Object | Responsibility |
| --- | --- |
| `ScenarioSpec` | compose one complete experiment |
| component definitions | immutable, validated semantic configuration |
| `LayerSpec` / `DataSpec` / `EvalAlgorithmSpec` | graph/routing/placement policy |
| `TensorDict` | explicit named data flow between pipeline blocks |
| runtime classes | mutable tensors, modules, datasets, fitted/evaluation state |
| `ResolvedConfig` | stable persistence boundary |
| `CompiledPipeline` | executable model graph |
| `NexuSession` | canonical training/evaluation lifecycle |

## Next

- [Scenarios](scenarios.md)
- [Architecture](../explanation/architecture.md)
- [TensorDict data flow](../explanation/tensordict.md)
- [Tutorials](../tutorials.md)
