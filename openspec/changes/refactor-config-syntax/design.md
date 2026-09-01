# refactor-config-syntax Design

## Context

NexuML currently conflates two representations that should be separate:

1. the representation a Python author wants to write and navigate; and
2. the neutral representation that can be persisted to YAML/JSON and restored in a different process.

For layers the current authoring model is:

```python
LayerSpec(
    type_key="LMBE",
    keys_in=["waveform"],
    keys_out=["spectrogram"],
    params={
        "sample_rate": 16_000,
        "n_mels": 128,
        "n_fft": 1024,
        "hop_length": 512,
        "to_db": True,
        "normalize": True,
    },
)
```

`LayerSpec.type_key` is then resolved through `LayerRegistry`; `LayerRegistry.validate_params()` inspects the runtime class `__init__`; and `compiler.py` performs further constructor inspection to determine whether values such as `num_classes` can be injected. Similar string-selector patterns exist for datasets, evaluation algorithms, and loader backends.

This works well as a transport representation but poorly as a Python API:

- `"LMBE"` is not navigable in an IDE;
- `params` is `dict[str, Any]`, so parameter names/types are hidden;
- constructor documentation and defaults are separated from the config object;
- typos are caught late;
- schema generation is weak because the framework does not have a typed model of component parameters;
- the registry/compiler must reconstruct parameter knowledge with reflection;
- runtime-only parameters and semantic component parameters are mixed in one constructor surface.

NexuFL `feature/nex-207-cifar-experiments` demonstrates the desired boundary. Its public configuration graph contains real immutable Pydantic component values such as `FedAvg()`, `IIDDistribution(...)`, and `LocalExecution()`. The registry stores identity/import metadata, and generic lowering converts concrete component types to a neutral identity only when persistence/transport requires it.

NexuML must adopt the same *principle*, not the same concrete implementation. A NexuFL method component can naturally be an immutable domain object. A NexuML `PipelineLayer` is a mutable `torch.nn.Module` with runtime state, instantiated child modules, epoch/lifecycle state, input shape information, shared storage, and graph wiring. Therefore NexuML needs a clean definition/runtime split.

## Design principles

The implementation SHALL optimize for these principles, in order:

1. **Typed Python first.** The normal Python authoring API uses concrete component definition objects.
2. **Serialization is a boundary.** Stable strings are introduced when dumping/loading config, not while authoring Python.
3. **Definition and runtime have different jobs.** Immutable semantic config is not a mutable `nn.Module`.
4. **One owner for each concern.** Definition owns component config; graph spec owns wiring; compiler context owns runtime inputs; registry owns identity; runtime owns execution state.
5. **Explicit beats reflective.** Build contexts replace constructor-signature inspection.
6. **Discovery stays dynamic and simple.** Preserve NexuML entry-point/local-root scanning rather than adding hard-coded imports.
7. **No compatibility architecture.** Migrate the repository atomically and delete old selector syntax instead of supporting two systems.
8. **No abstraction without a current role.** Do not wrap external framework classes or invent future plugin systems merely to make the design look uniform.

## Goals

- Python scenarios use navigable component symbols such as `LMBE(...)` rather than string selectors.
- Component-specific parameters are typed Pydantic fields with defaults, validation, documentation, and generated JSON schema.
- Mutable runtime classes remain ordinary PyTorch/NexuML runtime implementations.
- Runtime-only inputs are supplied explicitly by the compiler/materializer.
- YAML remains portable, readable, and independent of Python import paths.
- YAML round-trips restore the same concrete definition classes.
- One lean registry supports NexuML-owned component roles without duplicating registry logic.
- Existing entry-point and local-library discovery remains the extension mechanism.
- Built-in scenarios and custom-library docs show one canonical authoring style.
- Current compiler/registry reflection that exists solely because parameters were erased into dicts is removed.

## Non-goals

- Do not redesign the training loop, TensorDict pipeline contract, shape propagation semantics, data splitting semantics, evaluation math, or export artifact behavior.
- Do not make `ComponentDefinition` inherit `torch.nn.Module` or vice versa.
- Do not copy NexuFL's `RunSpec`, four-axis model, capabilities, placements, dependency metadata, or `ComponentPlan` hierarchy.
- Do not create a general dependency-injection container.
- Do not create a version migration engine. Version is identity metadata only in NEX-211.
- Do not rename every existing component registration key. Keep existing stable keys unless a concrete collision/invalid key requires a change.
- Do not turn scenario functions into component definitions. A scenario remains a recipe/composition function.
- Do not introduce NexuML wrappers for every `torch.optim`, scheduler, or Lightning callback class.
- Do not preserve the old selector-based Python/YAML syntax through compatibility aliases or translators.

## Target mental model

```text
Python authoring
────────────────────────────────────

LMBE(n_mels=64)
      │
      │ immutable typed definition
      ▼
LayerSpec
  component = LMBE(...)
  keys_in   = [...]
  keys_out  = [...]
      │
      │ compile
      ▼
LMBE.build(LayerBuildContext(...))
      │
      ▼
_LMBERuntime(PipelineLayer)


Persistence
────────────────────────────────────

LMBE(n_mels=64)
      │
      │ lower through registry
      ▼
name="LMBE", version="1", params={"n_mels": 64}
      │
      │ YAML/JSON
      ▼
portable document
      │
      │ registry lookup + Pydantic validation
      ▼
LMBE(n_mels=64)
```

The exact stable registration key may remain `"LMBE"` in NEX-211. Lowercase examples in discussion describe the persistence concept, not a requirement to rename all keys during this refactor.

## Core decisions

### D1 — Public component classes are immutable definitions

Add a minimal base in `src/nexuml/core/components.py`:

```python
from abc import ABC
from typing import ClassVar
from pydantic import BaseModel, ConfigDict


class ComponentDefinition(BaseModel, ABC):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        validate_default=True,
        allow_inf_nan=False,
        arbitrary_types_allowed=False,
    )

    kind: ClassVar[str]
    component_name: ClassVar[str]
    component_version: ClassVar[str] = "1"


class LayerDefinition(ComponentDefinition):
    kind = "layer"

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        raise NotImplementedError
```

Add equivalent small role bases only for roles converted in this change, for example:

- `LayerDefinition`
- `DataSourceDefinition`
- `EvalAlgorithmDefinition`
- `LoaderBackendDefinition`

Do not create role bases for hypothetical future systems.

The concrete public symbol owns semantic config:

```python
@layer("LMBE")
class LMBE(LayerDefinition):
    sample_rate: int = 16_000
    n_mels: int = Field(default=128, gt=0)
    n_fft: int = Field(default=1024, gt=0)
    hop_length: int = Field(default=512, gt=0)
    win_length: int | None = None
    power: int = 2
    fmin: int = 0
    fmax: int = 8_000
    mel_scale: Literal["slaney", "htk"] = "slaney"
    use_librosa: bool = False
    to_db: bool = True
    normalize: bool = False

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _LMBERuntime(definition=self, context=context)
```

The exact validation constraints should reflect actual current component semantics; do not invent restrictive validation that changes valid behavior.

**Why:** this makes `LMBE` itself the thing the user imports, clicks, documents, validates, and serializes. There is no second public `LMBEConfig` or `LMBE.Definition` to learn.

### D2 — Runtime implementations are private mutable objects

The current runtime behavior moves into an implementation class in the same module:

```python
class _LMBERuntime(PipelineLayer):
    def __init__(self, definition: LMBE, context: LayerBuildContext) -> None:
        super().__init__(
            input_sizes=context.input_sizes,
            keys_in=context.keys_in,
            keys_out=context.keys_out,
            label_key=context.label_key,
            label_in_x=context.label_in_x,
            num_classes=context.num_classes,
            shared_memory=context.shared_storage,
            delay_epochs=context.delay_epochs,
            update_every_n_epochs=context.update_every_n_epochs,
        )
        ...
```

The private runtime owns:

- instantiated `torch.nn.Module` children;
- weights/parameters/buffers;
- epoch and Lightning lifecycle state;
- inferred input/output shapes;
- shared-storage references;
- other non-serializable execution state.

The definition SHALL NOT contain tensors, live modules, open resources, dataset bytes, shared storage, trainer objects, registries, or other runtime state.

Do not attempt multiple inheritance between Pydantic definitions and PyTorch runtime classes.

### D3 — Graph wiring stays on graph specs

The public component definition describes *what the component is*. The surrounding spec describes *where/how it is wired*.

Target layer authoring:

```python
LayerSpec(
    component=LMBE(
        sample_rate=16_000,
        n_mels=128,
        n_fft=1024,
        hop_length=512,
        to_db=True,
        normalize=True,
    ),
    keys_in=["waveform"],
    keys_out=["spectrogram"],
)
```

`LayerSpec` continues to own concerns such as:

- `keys_in`
- `keys_out`
- `meta_in`
- `meta_out`
- graph scheduling such as `delay_epochs` / `update_every_n_epochs` if those are currently authored at layer placement level
- any other placement/wiring metadata that is not intrinsic to the component.

Do not put `keys_in`/`keys_out` on every `LMBE`, encoder, loss, etc. definition merely because `PipelineLayer.__init__` currently accepts them.

### D4 — Runtime dependencies are provided through explicit build contexts

Add small frozen dataclasses (or equivalently simple immutable value objects) containing runtime-only construction inputs. Example:

```python
@dataclass(frozen=True, slots=True)
class LayerBuildContext:
    input_sizes: Mapping[str, tuple[int, ...]]
    keys_in: list[str] | dict[str, str]
    keys_out: list[str]
    label_key: str | list[str] | None
    label_in_x: bool
    num_classes: int | None
    metadata: Mapping[str, Any]
    shared_storage: SharedStorage | None
    delay_epochs: int
    update_every_n_epochs: int
```

Only include values the current compiler/runtime actually needs. Do not turn context objects into a general service locator.

A layer that needs `num_classes` explicitly checks `context.num_classes`; the compiler no longer asks whether the runtime constructor happens to contain a parameter named `num_classes`.

Equivalent small contexts may exist for data sources/evaluation/loader materialization if those roles have meaningful framework-owned runtime values. Do not force all roles into one huge context type.

### D5 — The component registry stores identity, not configuration logic

Refactor toward one common registry in `src/nexuml/core/registry.py`:

```python
@dataclass(frozen=True, slots=True)
class ComponentEntry:
    kind: str
    name: str
    version: str
    definition_type: type[ComponentDefinition]
    import_target: str


class ComponentRegistry:
    def register(...): ...
    def get_entry(self, kind: str, name: str, version: str = "1") -> ComponentEntry: ...
    def get_type(self, kind: str, name: str, version: str = "1") -> type[ComponentDefinition]: ...
    def entry_for_type(self, definition_type: type[ComponentDefinition]) -> ComponentEntry: ...
    def entries(self, *, kind: str | None = None) -> tuple[ComponentEntry, ...]: ...
```

Registry responsibilities:

- uniqueness/conflict checking;
- `(kind, name, version)` lookup;
- reverse lookup from definition type;
- deterministic listing;
- discovery integration.

Registry non-responsibilities:

- inspecting runtime constructors;
- validating arbitrary param dicts;
- casting constructor values;
- materializing runtime objects itself;
- storing component defaults or schema separately from the definition class;
- deciding compiler injection behavior.

Delete `LayerRegistry.validate_params()` and analogous behavior after migration. Data/evaluation/loader registries should be removed or reduced to thin role views only if a real call site benefits; do not retain duplicated registry implementations for compatibility.

Scenario discovery may retain a small scenario-function registry because scenarios are recipes rather than `ComponentDefinition` values.

### D6 — Existing dynamic discovery is preserved

Keep NexuML's current discovery strengths:

- `nexuml.libraries` entry points;
- local library roots;
- scanning current files each CLI run;
- resilient import error collection;
- explicit decorators;
- no persistent object cache.

Repurpose component decorators to register definition classes:

```python
@layer("LMBE")
class LMBE(LayerDefinition): ...

@data_source("CIFAR10Dataset")
class CIFAR10Dataset(DataSourceDefinition): ...

@eval_algorithm("mahalanobis")
class Mahalanobis(EvalAlgorithmDefinition): ...
```

If loader backends currently use a dedicated registration API, expose the smallest analogous decorator/registration path needed for `LoaderBackendDefinition`; do not add a broad decorator taxonomy.

`@scenario` continues to decorate scenario functions.

Do not add a NexuFL-style `_BUILTIN_MODULES` tuple. The base library is already discoverable through installed package/local discovery and should remain so.

### D7 — Serialized config uses stable names only at the boundary

Python and YAML intentionally use different representations.

Python:

```python
LayerSpec(
    component=LMBE(n_mels=64),
    keys_in=["waveform"],
    keys_out=["spectrogram"],
)
```

Serialized YAML:

```yaml
pipeline:
  stages:
    Features:
      - type: LMBE
        version: "1"
        keys_in:
          - waveform
        keys_out:
          - spectrogram
        params:
          sample_rate: 16000
          n_mels: 64
          n_fft: 1024
          hop_length: 512
          to_db: true
          normalize: false
```

The exact formatting may use an internal `component` node if doing so materially simplifies the serializer, but the external contract MUST remain compact and obvious. Do not expose Python import paths. Do not require users to write Pydantic discriminator implementation details.

For a data source, prefer an explicit nested source object rather than parallel selector/params fields:

```python
DataSpec(
    source=CIFAR10Dataset(root=".data/cifar10", download=True),
    ...,
)
```

```yaml
data:
  source:
    type: CIFAR10Dataset
    version: "1"
    params:
      root: .data/cifar10
      download: true
```

For evaluation, keep routing/wiring separate:

```python
EvalAlgorithmSpec(
    algorithm=Mahalanobis(...),
    name="machine-distance",
    axis_keys=[...],
    feature_key="embedding",
)
```

For loaders, common loading policy remains on `LoaderSpec`; backend-specific configuration belongs to a typed backend definition:

```python
LoaderSpec(
    backend=DALI(...),
    batch_size=64,
    shuffle_train=True,
)
```

Do not move generic batch-size/shuffle/split policy into every backend definition.

### D8 — Lowering/restoration is generic and small

Add boundary helpers, preferably in `src/nexuml/core/serialization.py` if keeping them in `config.py` would mix concerns.

Conceptually:

```python
def lower_component(component: ComponentDefinition) -> dict[str, Any]:
    entry = registry.entry_for_type(type(component))
    return {
        "type": entry.name,
        "version": entry.version,
        "params": component.model_dump(mode="json"),
    }


def restore_component(*, kind: str, value: Mapping[str, Any]) -> ComponentDefinition:
    entry = registry.get_entry(kind, value["type"], value.get("version", "1"))
    return entry.definition_type.model_validate(value.get("params", {}))
```

The real implementation must ensure values are YAML/JSON safe and preserve existing supported value types such as paths by converting them at the serialization boundary.

Avoid a second parallel schema containing every component field. The concrete definition Pydantic class remains the source of truth.

Do not introduce a recursive `ComponentPlan`/children graph unless current NexuML components actually require nested registered components. Current scenario/spec composition should remain the place where component objects are nested. If a concrete present-day component genuinely contains another registered component, support that case generically; do not build speculative graph machinery.

### D9 — `ResolvedConfig` remains the scenario document boundary

`ResolvedConfig.to_yaml()` / `from_yaml()` SHALL use the lowering/restoration boundary rather than plain `model_dump()` / `ScenarioSpec.model_validate(raw_yaml)` for fields containing component definitions.

Desired behavior:

```python
scenario = ScenarioSpec(... LMBE(...) ...)
config = ResolvedConfig.from_scenario(scenario)
yaml_text = config.to_yaml()
restored = ResolvedConfig.from_yaml(yaml_text).to_scenario()

assert type(restored.pipeline.stages["Features"][0].component) is LMBE
assert restored == scenario
```

The serialized document remains portable plain data and can still be stored in export metadata/sidecars.

No loader for the old `type_key`/`source_type` selector format is added. Repository-owned configs/examples are migrated in the same change.

### D10 — Version is simple identity metadata

Every registered definition has `component_version`, defaulting to `"1"`.

Rules:

- serializer writes version;
- restorer resolves exact `(kind, name, version)`;
- unknown version fails with an actionable error;
- no automatic version upgrades/downgrades;
- no migration table/framework in NEX-211.

This prevents a persisted config from silently changing meaning without creating a compatibility subsystem.

### D11 — Component-specific schema stays on the concrete definition

A user/tool should be able to inspect:

```python
LMBE.model_json_schema()
```

and receive the real component fields/defaults/constraints.

The registry/CLI may expose that schema by looking up the definition type. It SHALL NOT synthesize schema from runtime constructor signatures.

Do not add `ConfigModel` attributes or generated Pydantic subclasses.

### D12 — Scope by component role

NEX-211 converts the NexuML-owned plugin roles where the framework currently loses type information into a selector plus arbitrary params.

#### Layers

Target:

```python
LayerSpec(component=LMBE(...), keys_in=[...], keys_out=[...])
```

Public definition builds private `PipelineLayer` runtime.

#### Data sources / dataset entries

Replace `source_type`/`type_key` + `params` with a typed data source definition. Preserve current split, modality, max-sample, merge-label, preprocessing, and dataset-composition semantics; this change is not permission to redesign `DataSpec`.

A data definition builds/loads the existing mutable `NexuDataset` runtime.

#### Evaluation algorithms

Replace algorithm `type` + `params` with a typed evaluation definition. Keep evaluation routing fields such as `name`, `enabled`, `axis_keys`, `feature_key`, and `label_key` outside the algorithm definition unless they are truly intrinsic to the algorithm.

#### Loader backends

Replace backend string + backend-specific `params` with a typed backend definition. Keep common loader policy such as batch size, worker policy where backend-agnostic, weighted sampling, and shuffle on `LoaderSpec`.

#### Scenarios

Keep `@scenario` functions returning `ScenarioSpec`. Scenarios compose definitions; they are not materialized runtime components.

#### External framework references

Do not create wrappers solely to eliminate every string in the config. `OptimizerSpec.type`, `SchedulerSpec.type`, and callback references may remain explicit import/known-alias references where the configured object belongs to PyTorch/Lightning rather than to the NexuML library component system.

### D13 — Compiler becomes simpler

The layer compilation loop should conceptually become:

```python
for spec in layer_specs:
    context = LayerBuildContext(...)
    layer = spec.component.build(context)
    pipeline_dims.update(_propagate_shapes(layer, pipeline_dims))
    ...
```

Remove:

- lookup by `spec.type_key` during normal Python compilation;
- registry parameter validation/casting;
- constructor-signature inspection;
- special injection based on constructor parameter names.

The registry is involved when a serialized config is restored, not every time an already-typed Python component is compiled.

Shape propagation and TensorDict behavior remain unchanged.

### D14 — Custom libraries get a simpler extension story

A custom layer should look like:

```python
@layer("scaled_relu")
class ScaledReLU(LayerDefinition):
    scale: float = 1.0

    def build(self, context: LayerBuildContext) -> PipelineLayer:
        return _ScaledReLURuntime(self, context)


class _ScaledReLURuntime(PipelineLayer):
    ...
```

Then scenario code uses:

```python
LayerSpec(
    component=ScaledReLU(scale=0.5),
    keys_in=["features"],
    keys_out=["features"],
)
```

No manual registry call, duplicated config model, or string selector is needed in normal Python authoring. Installed entry-point and local-root discovery makes the stable YAML identity restorable.

### D15 — Migration is atomic; legacy syntax is removed

This repository does not need a compatibility layer for unreleased/rapidly evolving configuration APIs.

Implementation sequence may use an intermediate vertical slice while coding, but the final branch SHALL expose one system only.

Remove/replace:

- `LayerSpec.type_key`
- layer `params: dict[str, Any]`
- data source selector fields used solely for registry lookup (`source_type`, dataset `type_key`) and their arbitrary param bags
- eval algorithm selector `type` + arbitrary `params`
- loader backend selector + backend-specific arbitrary `params`
- constructor-reflection validation made obsolete by typed definitions
- old docs/examples/tests that teach selector-based Python authoring

Do not keep `type_key` as an alias to `component`, accept both formats, add `model_validator(mode="before")` translators, or retain old registry APIs solely because old tests call them. Update the tests to the intended architecture.

### D16 — Keep registration keys stable during this refactor

The public persisted identity is the decorator key, not the Python class name or import path.

To keep NEX-211 focused, retain current built-in keys where practical (`LMBE`, `ConvolutionalEncoder`, existing dataset/eval keys, etc.). A later deliberate naming cleanup can normalize casing if desired.

Do not derive persisted names automatically from `__name__`; explicit registration protects configs from ordinary Python refactors.

### D17 — Minimal tests, broad contract coverage

Do not add one bespoke test per migrated component. Use parameterized/generic tests where possible, inspired by NexuFL's direct component contract tests.

Minimum focused coverage:

1. **Definition contract test** — discovered default-constructible definitions are Pydantic values, forbid unknown fields, and expose schema.
2. **Round-trip test** — representative scenario containing layer + data + evaluation + loader definitions survives Python -> YAML -> Python with exact concrete definition types and equal values.
3. **Compiler smoke test** — a representative typed pipeline (LMBE plus a small downstream layer, or an existing lightweight scenario) compiles/forwards without constructor reflection.
4. **Registry conflict test** — duplicate `(kind, name, version)` definitions fail clearly.
5. **Custom-library discovery test** — a tiny external/custom definition is discovered, serialized, restored, and materialized.
6. **Legacy rejection test** — one small test proves removed `type_key`/selector syntax is not silently accepted; do not build a large legacy test suite.
7. **Existing scenario smoke/regression tests** — update existing tests rather than duplicating them.

The implementation should end with fewer registry/reflection-specific tests than today if those tests only cover deleted complexity.

## Expected file/module structure

The target is intentionally small:

```text
src/nexuml/core/
├── components.py
│   ├── ComponentDefinition
│   ├── LayerDefinition
│   ├── DataSourceDefinition
│   ├── EvalAlgorithmDefinition
│   ├── LoaderBackendDefinition
│   └── small build-context value types
├── registry.py
│   ├── ComponentEntry
│   ├── ComponentRegistry
│   └── get_component_registry()
├── discovery.py
│   └── existing scanning/entry-point/local-root mechanism, adapted to definitions
├── serialization.py
│   ├── lower_component()
│   ├── restore_component()
│   └── scenario/config boundary helpers
├── types.py
│   ├── ScenarioSpec
│   ├── PipelineSpec
│   ├── LayerSpec(component=...)
│   ├── DataSpec(source=...)
│   ├── EvalAlgorithmSpec(algorithm=...)
│   └── LoaderSpec(backend=...)
├── config.py
│   └── ResolvedConfig YAML boundary
└── compiler.py
    └── definition.build(context)
```

It is acceptable to keep serialization helpers in `config.py` if a separate module would contain only a few trivial functions; choose the smaller result. Do not create a package hierarchy such as `components/base.py`, `components/registry.py`, `components/serialization.py`, `components/context.py` unless file size/real cyclic dependency pressure proves it necessary.

Component library modules should colocate definition and private runtime:

```text
library/src/nexuml_library/layers/feature/lmbe.py
├── LMBE                # public typed definition
└── _LMBERuntime        # private execution implementation
```

This avoids parallel definition/runtime directory trees and keeps navigation local.

## Reference implementation guidance from NexuFL

Use the following ideas from `NexuFL` branch `feature/nex-207-cifar-experiments`:

- immutable Pydantic component values from `src/nexufl/library/base.py`;
- direct Python authoring visible in `src/nexufl/library/recipes/cifar_resnet.py` and `experiments/journal/cifar/cases.py`;
- lean identity registry from `src/nexufl/library/registry_direct.py`;
- boundary lowering/restoration from `src/nexufl/core/lowering.py`;
- generic concrete-type round-trip testing from `tests/direct/test_components.py`.

Do NOT copy:

- NexuFL's `_BUILTIN_MODULES` discovery list;
- `ComponentPlan`/`ResolvedRunPlan` if NexuML does not need a recursive transport graph;
- RunSpec's federated axes;
- `capabilities`, `placements`, or dependency metadata not required by NexuML;
- runtime methods directly onto definitions when the runtime has mutable PyTorch/dataset state.

## Rejected alternatives

### A — Keep strings but improve registry search

Rejected because it does not solve IDE navigation, typed params, early validation, or reflection complexity.

### B — `LayerSpec(type=LMBE, params={...})`

Rejected because only the selector becomes navigable; component params remain untyped and the registry still needs constructor introspection.

### C — `LayerSpec(component_type=LMBE, config=LMBEConfig(...))`

Rejected because the user has to coordinate two public symbols and configuration can drift from runtime implementation.

### D — `LMBE.Definition(...)`

Rejected because it creates a nested secondary public API and encourages duplicate runtime/config signatures.

### E — Generate Pydantic models from `PipelineLayer.__init__`

Rejected because runtime-only arguments pollute component schema, static analysis remains weak, generated types are harder to navigate, and the implementation doubles down on reflection.

### F — Make `PipelineLayer` itself a Pydantic model

Rejected because PyTorch modules have mutable state, parameters, submodules, hooks, and construction semantics incompatible with an immutable portable domain value.

### G — Persist import paths

Rejected because module refactors would break config even when the public component identity is unchanged. Explicit registration names are the persistence API.

### H — Keep old and new syntax concurrently

Rejected because it creates two mental models, duplicated tests, validators/translators, and permanent maintenance burden. NEX-211 migrates repository-owned callers atomically.

### I — Build a universal plugin framework for every configurable object

Rejected. Only NexuML-owned registered plugin roles are converted. External optimizer/scheduler/callback references stay simple unless a separate concrete need justifies changing them.

## Risks and mitigations

### R1 — Broad migration touches many built-in scenarios

Mitigation: mechanically search for `type_key=`, `source_type=`, eval `type=... params=`, and loader backend strings after core contracts are implemented. Do not redesign scenario semantics while touching those files.

### R2 — Public runtime class names currently collide with desired definition names

Mitigation: keep the public name for the definition and make runtime implementation private in the same module. Internal imports that truly need the runtime should be replaced with definition/build usage rather than exposing a second public runtime name.

### R3 — Pydantic serialization of abstract base-typed fields does not automatically restore concrete subclasses

Mitigation: do not rely on automatic Pydantic polymorphic deserialization. `ResolvedConfig.from_yaml()` explicitly restores components through the registry before final scenario validation, mirroring NexuFL's lowering/restoration boundary.

### R4 — Custom component not installed when loading YAML

Mitigation: discovery runs before restoration. Unknown `(kind, name, version)` fails with a clear list/diagnostic of available components. Do not serialize import paths as a workaround.

### R5 — Circular imports between component bases, runtime classes, and specs

Mitigation: keep base definitions/contexts dependency-light, use forward annotations/`TYPE_CHECKING` where necessary, and colocate concrete definition/runtime pairs. Do not solve cycles by creating many extra modules prematurely.

### R6 — Over-conversion of fields that are actually graph/runtime policy

Mitigation: preserve the ownership rule. If a setting controls placement/wiring/batching/splitting rather than the semantic component implementation, leave it on the surrounding spec/context.

## Validation criteria

The implementation is accepted only if all of the following are true:

- Python examples contain concrete component objects, not registry key strings, for NexuML-owned plugin roles.
- Ctrl/Cmd-clicking a component symbol reaches the class that declares its configuration fields and documentation.
- Component parameter typos are rejected by Pydantic without compiling the pipeline.
- `LMBE.model_json_schema()` (and equivalents) exposes real component parameters.
- `ResolvedConfig.to_yaml()` emits plain portable YAML with explicit component identity/version/params.
- `ResolvedConfig.from_yaml()` restores exact concrete definition classes through discovery/registry lookup.
- compiler materialization uses `definition.build(context)` rather than runtime constructor reflection.
- graph wiring is not duplicated into every definition.
- one common component registry is the source of component identity.
- installed/local custom library discovery still works without a persistent cache or hard-coded built-in list.
- old selector-based authoring syntax is removed from framework types, base library scenarios, docs, and tests.
- no compatibility parser/aliases for old config syntax are added.
- no unrelated training/data/evaluation behavior changes are bundled into NEX-211.
- tests remain focused and generic rather than growing one test file per migrated component.
