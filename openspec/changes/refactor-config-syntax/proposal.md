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

The desired rule is:

> Python authoring uses typed component definitions. String identities exist only at serialization/discovery boundaries.

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
- Add explicit build/materialization contexts instead of inspecting runtime constructor signatures. Graph/runtime inputs such as `input_sizes`, `keys_in`, `keys_out`, `num_classes`, shared storage, and compiler metadata are provided through context objects, not serialized as component parameters.
- Keep graph wiring outside component definitions. `keys_in`, `keys_out`, `meta_in`, `meta_out`, scheduling, and similar placement concerns remain on `LayerSpec`/other graph specs.
- Replace the current behavior-heavy layer/data/evaluation registries with one lean component registry whose responsibility is identity and discovery metadata only: `(kind, name, version) -> definition type` plus reverse lookup by definition type.
- Reuse NexuML's existing installed-package entry-point discovery, local library root discovery, resilient scanning, and no-persistent-cache behavior. Do not copy NexuFL's hard-coded built-in module list.
- Lower typed definitions to stable YAML/JSON-safe component references only at persistence boundaries. Serialized documents use stable registration identity plus version and validated parameter values; they do not store Python import paths.
- Restore YAML into the same concrete definition type through the registry before compiling.
- Preserve existing serialized registration keys in this change unless a key is objectively broken. NEX-211 is a syntax/type-system refactor, not a registry-key renaming campaign.
- Convert the NexuML-owned plugin roles that currently expose selector + arbitrary params syntax: pipeline layers, data sources/dataset entries, evaluation algorithms, and loader backends.
- Keep scenario functions as scenario recipes (`@scenario`) rather than turning them into component definitions.
- Do not wrap arbitrary third-party framework objects merely for purity. Optimizer/scheduler class references and Lightning callback references may remain import/alias based where NexuML does not own a meaningful component abstraction.
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
- a version migration framework or compatibility layer for old config documents;
- a hard-coded list of built-in component modules for discovery;
- separate duplicated registries that each reimplement the same lookup/discovery logic;
- broad changes to unrelated training/runtime behavior.

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
- `src/nexuml/data/registry.py` and data construction paths, which should be deleted or reduced as the common component registry takes over
- `src/nexuml/evaluation/registry.py`, likewise
- `src/nexuml/data/loaders/registry.py`, likewise

Base library files expected to change:

- all decorated built-in layer modules under `library/src/nexuml_library/layers/`
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

The change is complete when a normal user can author a scenario with real Python symbols such as `LMBE(...)`, navigate directly to those definitions in the IDE, receive typed validation/autocomplete for component parameters, serialize the scenario to a stable YAML document, reload that YAML into the same concrete definition classes, and compile/run it without registry constructor reflection.

The final implementation should contain less indirection than the current system: the definition class owns its configuration and validation, the runtime class owns mutable execution state, the graph spec owns wiring, the registry owns identity, and serialization is the only place where strings replace Python component objects.
