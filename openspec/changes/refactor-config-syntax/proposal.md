# refactor-config-syntax

## Why

NexuML currently exposes persistence-oriented selector syntax as its primary Python authoring API. A typical layer is configured as:

```python
LayerSpec(
    type_key="LMBE",
    keys_in=["waveform"],
    keys_out=["spectrogram"],
    params={
        "sample_rate": 16_000,
        "n_mels": 128,
        "n_fft": 1024,
    },
)
```

This erases useful Python type information at authoring time. The user cannot Ctrl/Cmd-click `"LMBE"`, constructor parameters are hidden inside `dict[str, Any]`, autocomplete and static checking are weak, typos are discovered late, and the framework later has to recover information that Python already knew by consulting registries and inspecting runtime constructors.

The same selector-plus-untyped-params pattern appears in other configurable plugin roles such as data sources, evaluation algorithms, and loader backends.

NexuFL's direct component design on `feature/nex-207-cifar-experiments` demonstrates the better authoring model: config graphs contain real typed component values such as `FedAvg()`, `IIDDistribution(...)`, and `LocalExecution()`, while stable string identities are introduced only when lowering the graph for persistence or transport. NexuML should adopt that principle, but not copy the NexuFL implementation literally: NexuML pipeline layers and datasets are mutable runtime objects (`torch.nn.Module`, loaded dataset state, etc.), so the serializable public component definition must remain separate from the runtime object it materializes.

The typed-definition model should not force ordinary one-input/one-output PyTorch modules through a second NexuML definition and registration ceremony. A user who already has an importable `torch.nn.Module` class or factory should be able to reference that real Python symbol directly, retain IDE navigation and static constructor checking where supported, and rely on one universal NexuML adapter for TensorDict routing and portable reconstruction.

The desired rule is:

> Python authoring uses typed component definitions for NexuML semantics and direct importable factories for ordinary PyTorch modules. String identities exist only at serialization/discovery boundaries.

## What Changes

- Introduce a small immutable Pydantic `ComponentDefinition` base and role-specific definition bases for NexuML-owned configurable plugin roles.
- Make the public library symbol itself the typed definition. For example, Python authoring becomes:

```python
LayerSpec(
    component=LMBE(
        sample_rate=16_000,
        n_mels=128,
        n_fft=1024,
    ),
    keys_in=["waveform"],
    keys_out=["spectrogram"],
)
```

- Keep mutable execution state separate. `LMBE` is the immutable definition; a private runtime implementation such as `_LMBERuntime(PipelineLayer)` is created during compilation.
- Add one built-in `NnModuleLayer` Python definition, registered under the stable persisted identity `NnModule`, and a typed `nn_module(factory, *args, **kwargs)` helper for importable factories that return ordinary `torch.nn.Module` objects. The helper stores only a portable import target and JSON-safe constructor values; it does not register each wrapped module type.
- Reuse the existing `TorchModuleAdapter` as the single mutable runtime for direct modules. The initial contract is deliberately narrow: exactly one tensor input, one tensor output, no label consumption, and no component-specific runtime context injection.
- Reject live module instances, lambdas, closures, local definitions, non-importable factories, and non-JSON-safe constructor values when authoring the helper. Reject non-module factory results during materialization and non-tensor module outputs during execution, with clear errors rather than pretending those values satisfy the portable direct-module contract.
- Add explicit build/materialization contexts instead of inspecting runtime constructor signatures. Graph/runtime inputs such as `input_sizes`, `keys_in`, `keys_out`, `num_classes`, shared storage, and compiler metadata are provided through context objects, not serialized as component parameters.
- Keep graph wiring outside component definitions. `keys_in`, `keys_out`, `meta_in`, `meta_out`, scheduling, and similar placement concerns remain on `LayerSpec`/other graph specs.
- Replace the current behavior-heavy layer/data/evaluation registries with one lean component registry whose responsibility is identity and discovery metadata only: `(kind, name, version) -> definition type` plus reverse lookup by definition type.
- Reuse NexuML's existing installed-package entry-point discovery, local library root discovery, resilient scanning, and no-persistent-cache behavior. Do not copy NexuFL's hard-coded built-in module list.
- Lower typed definitions to stable YAML/JSON-safe component references only at persistence boundaries. Registered semantic components use stable registration identity plus version and validated parameter values rather than Python import paths. Portable direct-framework specs store an explicit external factory import target plus JSON-safe constructor values.
- Restore YAML into the same concrete definition type through the registry before compiling.
- Preserve existing serialized registration keys in this change unless a key is objectively broken. NEX-211 is a syntax/type-system refactor, not a registry-key renaming campaign.
- Convert the NexuML-owned plugin roles that currently expose selector + arbitrary params syntax: pipeline layers, data sources/dataset entries, evaluation algorithms, and loader backends.
- Remove the redundant built-in `IdentityLayer`, `Dropout`, and `Flatten` definitions after replacing their canonical forms with `nn_module(torch.nn.Identity)`, `nn_module(torch.nn.Dropout, ...)`, and `nn_module(torch.nn.Flatten, start_dim=1, end_dim=-1)`. Do not add compatibility aliases for their old Python imports, serialized identities, or checkpoint state paths.
- Keep scenario functions as scenario recipes (`@scenario`) rather than turning them into component definitions.
- Do not require arbitrary third-party framework classes to become NexuML library definitions. Use `nn_module(...)` for modules satisfying its narrow tensor contract and typed `optimizer(...)`, `scheduler(...)`, `callback(...)`, `strategy(...)`, or `writer(...)` helpers for importable factories whose runtime dependencies are supplied later.
- Update built-in library scenarios, tests, examples, and documentation to use typed Python authoring.
- Remove the old Python authoring fields and old YAML syntax after migration. There is no dual syntax, compatibility shim, deprecation adapter, or legacy parser in this change.

## Architectural Guardrails

The implementation MUST remain small, direct, and understandable. In particular, it SHALL NOT introduce any of the following:

- dynamically generated Pydantic config models from runtime `__init__` signatures;
- `ConfigModel`, `Definition` inner classes, or parallel config/runtime class hierarchies that duplicate fields;
- `LayerSpec(type=LMBE, params={...})` or any other class-reference-plus-untyped-dict compromise;
- metaclasses, descriptors, code generation, runtime schema synthesis, or reflection-heavy factories;
- constructor-signature inspection to validate component parameters or decide what runtime values to inject;
- a central `Union[...]` containing every built-in component type;
- component-specific `if/elif` serialization dispatch tables;
- Python import paths as the persisted component identity;
- automatic constructor-signature inspection or generated schemas for direct factory helpers;
- support for live module instances, lambdas, closures, local definitions, or arbitrary non-JSON constructor objects in portable resolved config;
- a version migration framework or compatibility layer for old config documents;
- a hard-coded list of built-in component modules for discovery;
- separate duplicated registries that each reimplement the same lookup/discovery logic;
- broad changes to unrelated training/runtime behavior.

External factory targets are not replacements for stable registry identities. They are reconstruction values for explicitly importable framework implementations and are accepted only under trusted-config assumptions.

NexuFL is a conceptual reference for typed authoring, immutable values, registry identity, lowering/restoration, and generic round-trip tests. It is not a template to copy wholesale. NexuML does not need NexuFL's RunSpec axes, `ComponentPlan` graph model, placement/capability metadata, dependency metadata, or hard-coded discovery list for this change.

## Capabilities

### New

- `typed-component-authoring`
- `component-config-serialization`
- `component-runtime-materialization`

### Modified

- `library-discovery`
- `decorator-discovery-docs`

## Impact

Primary framework files expected to change:

- `src/nexuml/core/types.py`
- `src/nexuml/core/compiler.py`
- `src/nexuml/core/registry.py`
- `src/nexuml/core/discovery.py`
- `src/nexuml/core/config.py`
- new `src/nexuml/core/components.py`
- new `src/nexuml/core/serialization.py` if serialization does not fit cleanly in `config.py`
- new `src/nexuml/core/factory.py` for portable factory targets and typed role helpers
- `src/nexuml/core/torch_adapter.py` for the universal definition/helper and its existing generic runtime
- `src/nexuml/__init__.py` for the direct public helper
- `src/nexuml/core/export.py` so wrapped custom child-module source is included in self-contained package exports
- `src/nexuml/data/registry.py` and data construction paths, which should be deleted or reduced as the common component registry takes over
- `src/nexuml/evaluation/registry.py`, likewise
- `src/nexuml/data/loaders/registry.py`, likewise

Base library files expected to change:

- all decorated built-in layer modules under `library/src/nexuml_library/layers/`
- removal of the redundant `IdentityLayer`, `Dropout`, and `Flatten` definitions/modules
- configured data source modules under `library/src/nexuml_library/data/`
- configured evaluation algorithm modules
- scenario builders that currently author `type_key` / `source_type` / `type` plus `params`

Documentation expected to change:

- `docs/learn/scenarios.md`
- `docs/learn/decorators-and-discovery.md`
- `docs/how-to/custom-layer.md`
- `docs/how-to/custom-data-source.md`
- `docs/how-to/custom-eval-algorithm.md`
- `docs/how-to/custom-library.md`
- `docs/reference/decorators.md`
- `docs/reference/scenario-spec.md`
- `docs/explanation/architecture.md`
- other examples found by searching for removed selector syntax

## Acceptance Summary

The change is complete when a normal user can author a scenario with real Python symbols such as `LMBE(...)`, `nn_module(torch.nn.Dropout, p=0.5)`, and `callback(ModelCheckpoint, ...)`, navigate directly to those definitions/factories in the IDE, receive typed validation/autocomplete appropriate to each path, serialize the scenario to portable YAML, restore registered definitions or importable factories as applicable, and compile/run it without registry constructor reflection.

The final implementation should contain less indirection than the current system: a semantic definition class owns its configuration and validation, ordinary external modules share one universal adapter, runtime objects own mutable execution state, the graph spec owns wiring, the registry owns stable NexuML identity, and serialization is the only place where Python factory references become import-target strings.
