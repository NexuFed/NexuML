# Architecture: Define, Persist, Materialize, Run

NexuML separates immutable semantic configuration from mutable execution objects.

## Definitions

Python scenario graphs contain concrete Pydantic definitions:

```python
LayerSpec(
    component=LinearEncoder(hidden_dims=[32], output_dim=8),
    keys_in=["features"],
    keys_out=["latent"],
)
```

The public `LinearEncoder` value declares configurable fields, defaults, validation, and schema. `LayerSpec` declares graph wiring. A data source, evaluation algorithm, or loader backend follows the same typed-definition pattern.

Definitions are frozen portable values with no tensors, modules, loaded data, trainer state, or shared storage.

Ordinary one-input/one-output tensor modules share one core definition and runtime:

```python
LayerSpec(
    component=nn_module(torch.nn.Linear, 128, 64),
    keys_in=["features"],
    keys_out=["embedding"],
)
```

`NnModuleLayer` stores the importable factory target and JSON-safe constructor values. Its `build()` method invokes only those explicit values and places the resulting `torch.nn.Module` inside `TorchModuleAdapter`. Modules needing labels, metadata, build context, custom lifecycle, or richer input/output routing remain registered semantic definitions.

## Identity Registry

Decorators assign explicit `(kind, name, version)` identities. The common `ComponentRegistry` owns only identity lookup, reverse lookup, deterministic listing, and conflict diagnostics. It does not inspect runtime constructors or validate parameter dictionaries.

Scenario functions remain in a separate recipe registry.

## Persistence

At YAML and checkpoint boundaries, generic lowering replaces each definition with stable data:

```yaml
component:
  type: LinearEncoder
  version: '1'
  params:
    hidden_dims: [32]
    output_dim: 8
```

Restoration discovers the component and performs exact kind/name/version lookup followed by `definition_type.model_validate(params)`. Registered semantic definitions persist no Python import path or runtime object.

`NnModule` is the explicit external-code exception: its stable component identity contains a top-level `module:name` factory target. Constructor values are recursively limited to JSON-safe primitives, lists, and string-key mappings. Compiling this trusted config imports and invokes that target; it does not support live instances, lambdas, closures, local definitions, or constructor reflection.

## Materialization

Each role has one explicit build boundary:

- `LayerDefinition.build(LayerBuildContext)` creates a `PipelineLayer`.
- `DataSourceDefinition.build()` creates a `NexuDataset`.
- `EvalAlgorithmDefinition.build(EvalBuildContext)` creates an `EvalAlgorithm`.
- `LoaderBackendDefinition.build()` creates a loader backend.

Runtime-only values are supplied through the surrounding spec or build context. For layers this includes inferred input shapes, TensorDict keys, labels, class count, metadata, shared storage, and scheduling.

Public definitions and private runtimes are usually colocated:

```python
@layer("scaled_relu")
class ScaledReLU(LayerDefinition):
    scale: float = 1.0

    def build(self, context: LayerBuildContext):
        return _ScaledReLURuntime(scale=self.scale, **context.runtime_kwargs())


class _ScaledReLURuntime(PipelineLayer):
    ...
```

## Compile And Run

The compiler propagates shapes and key metadata, constructs `LayerBuildContext`, and calls `spec.component.build(context)` directly. It does not resolve a layer name or inspect `__init__` during normal Python compilation.

The compiled pipeline routes a `TensorDict` through ordered stages. PyTorch Lightning owns training, callbacks, checkpointing, and device execution. Evaluation materializes typed algorithms after training.

## Discovery

Each CLI run scans the built-in library, installed `nexuml.libraries` entry points, and configured local roots. Errors from one module are collected without hiding unrelated components. There is no persistent discovery cache or hard-coded module list.

Direct module factories are not discovered or registered individually. Self-contained export inspects modules nested inside `TorchModuleAdapter` so custom source is packaged while PyTorch and other runtime dependencies remain external. The former `IdentityLayer`, `Dropout`, and `Flatten` component identities have no aliases; their direct PyTorch equivalents use `nn_module(...)`.

## Ownership

| Concern | Owner |
|---|---|
| Component semantics and validation | Concrete definition |
| Graph wiring and placement | Scenario specs |
| Stable persisted identity | Component registry |
| Runtime construction values | Build context |
| Mutable execution state | Private runtime |
| YAML/checkpoint conversion | Generic serialization boundary |

## See Also

- [Scenarios](../learn/scenarios.md)
- [Decorators and discovery](../learn/decorators-and-discovery.md)
- [Custom layer](../how-to/custom-layer.md)
