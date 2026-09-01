# refactor-config-syntax Tasks

## 1. Lock the architecture before migrating call sites

- [ ] 1.1 Read this change's `proposal.md`, `design.md`, and all delta specs before editing code.
- [ ] 1.2 Inspect the conceptual reference in NexuFL `feature/nex-207-cifar-experiments`, especially `library/base.py`, `library/registry_direct.py`, `core/lowering.py`, the CIFAR recipes/cases, and `tests/direct/test_components.py`.
- [ ] 1.3 Record the current selector/reflection call sites with repository searches for at least `type_key`, `source_type`, `params`, `validate_params`, `inspect.signature`, evaluation algorithm selectors, and loader backend selectors.
- [ ] 1.4 Do not begin by adding compatibility aliases or a second config representation. The target architecture is the only final architecture.

## 2. Add the minimal typed component foundation

- [ ] 2.1 Add `ComponentDefinition` in `src/nexuml/core/components.py` as an immutable Pydantic base with `extra="forbid"`, validated defaults, finite numeric values, and no arbitrary live runtime types.
- [ ] 2.2 Add only the role bases required by current selector-based NexuML plugin roles: `LayerDefinition`, `DataSourceDefinition`, `EvalAlgorithmDefinition`, and `LoaderBackendDefinition`.
- [ ] 2.3 Add small explicit materialization context value types, beginning with `LayerBuildContext`; include only runtime values currently needed by the compiler/runtime.
- [ ] 2.4 Keep graph wiring/runtime placement fields out of concrete definitions.
- [ ] 2.5 Ensure concrete definitions can expose their real parameter schema through ordinary Pydantic `model_json_schema()` without generated config classes.

## 3. Replace duplicated behavior-heavy registries with one identity registry

- [ ] 3.1 Refactor `src/nexuml/core/registry.py` around a small `ComponentEntry` and `ComponentRegistry` keyed by explicit component kind/name/version.
- [ ] 3.2 Support reverse lookup from concrete definition type to registration entry for serialization.
- [ ] 3.3 Keep deterministic listing and clear conflict/unknown-component diagnostics.
- [ ] 3.4 Remove registry constructor-parameter validation/casting and any config-schema reconstruction based on `inspect.signature`.
- [ ] 3.5 Fold data-source, evaluation, and loader component lookup into the common registry; delete or reduce their separate registry implementations rather than maintaining duplicate logic for compatibility.
- [ ] 3.6 Keep scenario-function registration separate if needed because scenarios are recipes, not component definitions.

## 4. Adapt discovery without replacing it

- [ ] 4.1 Keep current installed `nexuml.libraries` entry-point discovery.
- [ ] 4.2 Keep current local library root discovery and fresh scanning on each CLI run.
- [ ] 4.3 Keep resilient import/registration error collection so one broken module does not hide unrelated valid components.
- [ ] 4.4 Adapt `@layer`, `@data_source`, and `@eval_algorithm` to decorate the corresponding definition classes and register their explicit stable identities.
- [ ] 4.5 Add the smallest loader-backend registration hook only if the existing loader registry needs one for typed restoration; do not create a broad decorator framework.
- [ ] 4.6 Keep `@scenario` semantics unchanged.
- [ ] 4.7 Do not add a hard-coded built-in-module import list or persistent discovery cache.

## 5. Implement the serialization boundary

- [ ] 5.1 Add small generic lowering/restoration helpers in `src/nexuml/core/serialization.py`, or keep them in `config.py` if that produces a materially smaller design.
- [ ] 5.2 Lower a definition through reverse registry lookup to stable `type`/`version`/validated `params` plain data.
- [ ] 5.3 Restore serialized component data by exact kind/name/version lookup followed by `definition_type.model_validate(params)`.
- [ ] 5.4 Ensure serialized values are YAML/JSON safe, including current supported path-like values.
- [ ] 5.5 Do not serialize concrete Python module/import paths.
- [ ] 5.6 Do not create component-specific serializer branches; the concrete Pydantic definition remains the field/schema source of truth.
- [ ] 5.7 Do not create a recursive NexuFL-style component-plan graph unless a current NexuML component demonstrably requires nested registered definitions.
- [ ] 5.8 Write exact version identity but do not add version migration/upgrade machinery.

## 6. Refactor `ResolvedConfig` and scenario spec boundaries

- [ ] 6.1 Change `LayerSpec` from `type_key + params` authoring to `component: LayerDefinition` plus its existing graph wiring fields.
- [ ] 6.2 Replace data-source selector/param fields with typed data source definition fields while preserving existing split/dataset/preprocessing semantics.
- [ ] 6.3 Replace evaluation algorithm `type + params` with an `algorithm: EvalAlgorithmDefinition` field while keeping routing fields (`name`, `enabled`, axis/feature/label keys) on `EvalAlgorithmSpec`.
- [ ] 6.4 Replace loader backend selector + backend-specific arbitrary params with `backend: LoaderBackendDefinition`; keep backend-independent loader policy on `LoaderSpec`.
- [ ] 6.5 Update `ResolvedConfig.to_yaml()` to lower component definitions into portable plain data.
- [ ] 6.6 Update `ResolvedConfig.from_yaml()` to run discovery/registry restoration before final Pydantic scenario validation.
- [ ] 6.7 Preserve the resolved config/export sidecar use case with the new serialized syntax.
- [ ] 6.8 Do not accept the removed selector syntax through aliases, `mode="before"` translators, fallback parsing, or compatibility models.

## 7. Prove the design with one vertical layer slice before mass migration

- [ ] 7.1 Convert `library/src/nexuml_library/layers/feature/lmbe.py` first.
- [ ] 7.2 Make public `LMBE` the typed `LayerDefinition` declaring semantic parameters/defaults/validation.
- [ ] 7.3 Move current mutable torchaudio/PyTorch behavior into a private runtime class in the same module, e.g. `_LMBERuntime(PipelineLayer)`.
- [ ] 7.4 Implement `LMBE.build(context)` as the direct construction boundary.
- [ ] 7.5 Update one small existing scenario/pipeline fragment to author `LayerSpec(component=LMBE(...), ...)`.
- [ ] 7.6 Verify Python -> YAML -> Python preserves concrete `LMBE` type and values.
- [ ] 7.7 Verify the pipeline compiles/forwards using `definition.build(context)` with no constructor-signature inspection.
- [ ] 7.8 If this vertical slice requires a more complicated abstraction than described in `design.md`, simplify the implementation rather than generalizing prematurely.

## 8. Migrate all built-in pipeline layers mechanically

- [ ] 8.1 Convert each registered built-in layer so its existing public class name is the typed definition where practical and its mutable implementation is private/local to the module.
- [ ] 8.2 Move only semantic constructor parameters onto the definition; keep compiler/wiring/runtime arguments in the build context/spec.
- [ ] 8.3 Preserve current runtime behavior and defaults unless an existing bug is directly exposed by the migration.
- [ ] 8.4 Update built-in scenario fragments to import and instantiate the concrete definitions instead of writing registration strings and untyped param dicts.
- [ ] 8.5 Keep existing registration keys stable during NEX-211 unless a concrete collision/invalid identity requires a targeted change.
- [ ] 8.6 Remove dead runtime constructor parameters/registry plumbing made obsolete by the definition/context split.

## 9. Migrate data sources without redesigning data semantics

- [ ] 9.1 Convert registered data sources to typed `DataSourceDefinition` values with private/current mutable `NexuDataset` runtime implementations.
- [ ] 9.2 Keep dataset root/download/layout defaults on the appropriate data definition/builders rather than duplicating them in scenarios.
- [ ] 9.3 Update `DataSpec`/`DatasetSpec` call sites to pass concrete data definitions.
- [ ] 9.4 Preserve existing train/val/test split, multi-dataset, modality, preprocessing, merge-label, sharding, and exported-dataset behavior.
- [ ] 9.5 Remove obsolete `DatasetRegistry.instantiate(type_key, **params)`-style plumbing after all call sites use typed definitions/restoration.

## 10. Migrate evaluation algorithms and loader backends narrowly

- [ ] 10.1 Convert registered evaluation algorithms to typed `EvalAlgorithmDefinition` values.
- [ ] 10.2 Keep evaluation routing/axis/feature/label fields outside the algorithm definition when they describe graph/evaluation placement rather than algorithm semantics.
- [ ] 10.3 Convert loader backend identity/backend-specific params to a typed `LoaderBackendDefinition`.
- [ ] 10.4 Keep common loader settings (batch size, generic shuffle/weighted-sampling policy, etc.) on `LoaderSpec` rather than duplicating them in each backend definition.
- [ ] 10.5 Do not change DALI/Torch loader execution semantics as part of this syntax refactor.

## 11. Simplify compiler/materialization paths

- [ ] 11.1 Refactor the pipeline compiler so it constructs `LayerBuildContext` and calls `spec.component.build(context)` directly.
- [ ] 11.2 Remove normal Python compile-time lookup by `spec.type_key`.
- [ ] 11.3 Remove constructor parameter validation/casting from the layer registry.
- [ ] 11.4 Remove compiler checks such as inspecting `__init__` to determine whether `num_classes` can be injected; components consume explicit context instead.
- [ ] 11.5 Preserve shape propagation, metadata handoff (`meta_in`/`meta_out`), stage skipping, TensorDict key semantics, and compiled pipeline output.
- [ ] 11.6 Apply the same explicit materialization principle to data/eval/loader roles only where they currently perform string lookup + arbitrary param instantiation.

## 12. Custom-library and public API cleanup

- [ ] 12.1 Update custom extension examples so a custom component is one public typed definition plus a private runtime implementation when runtime state is required.
- [ ] 12.2 Verify installed entry-point libraries and configured local library roots can restore serialized custom component identities.
- [ ] 12.3 Ensure registry CLI listing still reports component kind/name/type/module clearly.
- [ ] 12.4 If schema inspection is exposed through CLI/reference tooling, source it directly from the concrete definition type.
- [ ] 12.5 Remove old public helper APIs that exist only to author/instantiate selector+param components; do not keep deprecated forwarding wrappers.

## 13. Migrate documentation and examples to one canonical syntax

- [ ] 13.1 Update `docs/learn/scenarios.md` to use concrete definitions such as `LMBE(...)`.
- [ ] 13.2 Update `docs/learn/decorators-and-discovery.md` to explain that decorators register typed definitions for persisted identity/discovery, while Python uses the definition class directly.
- [ ] 13.3 Update `docs/how-to/custom-layer.md`, `custom-data-source.md`, `custom-eval-algorithm.md`, and `custom-library.md`.
- [ ] 13.4 Update `docs/reference/decorators.md`, `docs/reference/scenario-spec.md`, and `docs/explanation/architecture.md`.
- [ ] 13.5 Search all docs/examples for removed authoring fields and migrate them; no docs should teach `LayerSpec(type_key=..., params=...)` or equivalent old data/eval syntax.
- [ ] 13.6 Explain the distinction explicitly: Python uses concrete definitions; YAML uses stable registered identities.

## 14. Focused contract tests only

- [ ] 14.1 Add/adjust one parameterized definition contract test covering discovered definition classes without writing one bespoke test per component.
- [ ] 14.2 Add one representative full scenario round-trip test that includes layer, data source, evaluation algorithm, and loader definitions and checks exact restored concrete types.
- [ ] 14.3 Add/update one lightweight compiler smoke test proving a typed layer definition materializes and forwards successfully.
- [ ] 14.4 Keep/update one registry duplicate-identity test for `(kind, name, version)` conflicts.
- [ ] 14.5 Add/update one tiny custom-library discovery/serialization/materialization test.
- [ ] 14.6 Add one small negative test proving removed legacy selector syntax is not silently accepted; do not create a compatibility test matrix.
- [ ] 14.7 Prefer updating/deleting existing registry/reflection tests over adding parallel coverage for code that no longer exists.

## 15. Final cleanup and validation

- [ ] 15.1 Repository-search for `LayerSpec(type_key`, layer `params`, `source_type`, dataset `type_key`, old eval algorithm selector syntax, loader backend param bags, and obsolete registry `validate_params`; all remaining matches must be intentional serialization/reference/docs-history cases.
- [ ] 15.2 Repository-search for `inspect.signature` in component construction paths and remove uses that only support deleted selector/param reflection.
- [ ] 15.3 Confirm there is no generated `ConfigModel`, dynamic Pydantic model factory, metaclass config system, import-path persistence, hard-coded built-in discovery list, or compatibility parser.
- [ ] 15.4 Run focused tests for component definitions/serialization/compiler/discovery first.
- [ ] 15.5 Run the normal non-slow test suite with `uv run pytest` (use repository markers/optional dependencies appropriately; do not require external data/GPU tests for this syntax refactor unless an existing CI job does).
- [ ] 15.6 Run `uv run ruff check src library/src tests`.
- [ ] 15.7 Run `uv run ty check` using the repository's configured paths.
- [ ] 15.8 Run documentation checks/build used by the repository if touched docs are covered by CI.
- [ ] 15.9 Run `openspec validate refactor-config-syntax --strict` (or the installed OpenSpec CLI's equivalent strict validation command).
- [ ] 15.10 Review the final diff specifically for unnecessary new abstractions and delete any layer that does not have a clear current responsibility described in `design.md`.
