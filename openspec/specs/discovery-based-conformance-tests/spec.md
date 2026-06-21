## Purpose

Define discovery-based conformance tests that automatically test all registered pipeline layers, evaluation algorithms, and scenario builders using parametrized pytest suites.

## Requirements

### Requirement: Layer registry drives parametrized forward-contract tests
The system SHALL provide a pytest suite that, at collection time, discovers all `PipelineLayer` subclasses registered in the NexuML layer registry and parametrizes a forward-contract test over them. Each registered layer SHALL be tested by instantiating it with a synthetic minimal TensorDict and verifying that `forward()` returns `(TensorDict, Optional[TensorDict])` without raising.

#### Scenario: New layer is automatically tested
- **WHEN** a developer registers a new layer with `@layer("MyLayer")` and adds it to the library
- **THEN** the conformance suite SHALL pick it up at the next test collection without requiring a new test file

#### Scenario: Layer forward returns correct types
- **WHEN** the forward-contract test runs for any registered layer
- **THEN** `forward(x, y)` SHALL return a tuple of `(TensorDict, Optional[TensorDict])` and the output TensorDict SHALL contain at least one key

#### Scenario: PostTrainFitLayer is tested in both modes
- **WHEN** the forward-contract test runs for a `PostTrainFitLayer` subclass
- **THEN** the test SHALL also call `collect_batch(x, y)`, `finalize_fit()`, and then `forward(x, y)` in fitted mode, verifying outputs in each phase

### Requirement: EvalAlgorithm registry drives parametrized eval-batch-contract tests
The system SHALL provide a pytest suite that discovers all `EvalAlgorithm` subclasses in the registry and parametrizes an eval-batch-contract test over them. Each algorithm SHALL be tested by calling `eval_batch(x, y)` with a synthetic TensorDict and verifying `results()` returns a dict of scalar-valued metrics.

#### Scenario: New eval algorithm is automatically tested
- **WHEN** a developer registers a new eval algorithm with `@eval_algorithm("my_algo")` and adds it to the library
- **THEN** the conformance suite SHALL pick it up without requiring a new test file

#### Scenario: Eval algorithm results contract holds
- **WHEN** `eval_batch(x, y)` and `eval_end()` are called on any registered eval algorithm
- **THEN** `results()` SHALL return a `dict[str, float]` (or empty dict) without raising

### Requirement: Scenario builders drive parametrized build-contract tests
The system SHALL provide a pytest suite that discovers all scenario builder functions in the library and parametrizes a build-contract test over them. Each scenario builder SHALL be tested by calling it with minimal required arguments and verifying that a `ScenarioSpec` with a non-empty pipeline is returned.

#### Scenario: New scenario builder is automatically tested
- **WHEN** a developer adds a new scenario builder function to the library and registers it in the discovery index
- **THEN** the conformance suite SHALL pick it up without a new test file

#### Scenario: Scenario builder returns valid ScenarioSpec
- **WHEN** the build-contract test calls a scenario builder with minimal arguments
- **THEN** the returned `ScenarioSpec` SHALL validate against the pydantic model, the `pipeline` field SHALL be non-empty, and the `evaluation.algorithms` list SHALL only contain types registered as `EvalAlgorithm`

### Requirement: Conformance failures are isolated per component
The system SHALL mark each parametrized conformance test case independently so that a failure in one component does not suppress results for other components. Components that require non-synthetic data or special initialization SHALL be skippable via a conformance fixture declaration on the component.

#### Scenario: One broken layer does not hide other results
- **WHEN** one registered layer's `forward()` raises an unexpected exception in the conformance suite
- **THEN** only that layer's test case SHALL fail; all other layer cases SHALL continue to execute and report independently

#### Scenario: Component can declare conformance skip
- **WHEN** a registered component decorates with `@conformance_skip(reason="requires real audio data")`
- **THEN** the conformance test for that component SHALL be marked as `pytest.skip` with the declared reason

### Requirement: Conformance collection covers configured local libraries
The conformance suite SHALL include registry elements discovered from local library roots configured through the library-management path.

#### Scenario: Local library layer is collected for conformance
- **WHEN** a test fixture adds a temporary local library root containing a decorated layer
- **THEN** pytest collection for the layer conformance suite SHALL include a parameter case for that layer key

#### Scenario: Local library scenario is collected for conformance
- **WHEN** a test fixture adds a temporary local library root containing a decorated scenario builder
- **THEN** pytest collection for the scenario conformance suite SHALL include a parameter case for that scenario key

#### Scenario: Local library data source and eval algorithm are collected for conformance
- **WHEN** a test fixture adds a temporary local library root containing a decorated data source and evaluation algorithm
- **THEN** pytest collection for the corresponding conformance suites SHALL include parameter cases for those keys

### Requirement: Conformance collection covers installed library entry points
The conformance suite SHALL include registry elements discovered through installed package entry-point discovery.

#### Scenario: Entry-point package contributes registry elements
- **WHEN** an isolated test simulates an installed package entry point that exposes registry elements
- **THEN** discovery SHALL load that package and conformance collection SHALL include its registry keys

### Requirement: Unexpected conformance skips fail visibility checks
The conformance suite SHALL distinguish explicit optional skips from unexpected failures or broad exception masking, using a test-side allowlist rather than production decorator metadata.

#### Scenario: Component on the test-side allowlist remains skipped
- **WHEN** a discovered component's key appears in a contract test's `{key: reason}` allowlist of components known to be untestable on synthetic data
- **THEN** its conformance parameter case SHALL be skipped with that reason

#### Scenario: Component not on the allowlist fails on contract error
- **WHEN** a discovered component whose key is not on the allowlist raises during contract execution
- **THEN** only that component's parameter case SHALL fail and the failure SHALL be visible in pytest output

### Requirement: Backend registries are discovery-driven in conformance tests
Conformance coverage for extensible backend registries SHALL be parametrized over the registry's listing function rather than a hardcoded name list.

#### Scenario: Export backend conformance reflects the live registry
- **WHEN** `tests/_registry/test_backend_contract.py` runs the `data-export` category
- **THEN** the tested backend names SHALL come from `list_export_backends()`, not a hardcoded list, so a newly registered export backend is automatically included

#### Scenario: Loader backend conformance reflects the live registry
- **WHEN** `tests/_registry/test_backend_contract.py` runs the `data-loader` category
- **THEN** the tested backend names SHALL come from `list_loader_backends()`, not a hardcoded list, so a newly registered loader backend is automatically included

### Requirement: Registry coverage meta-test detects uncollected discovered keys
The test suite SHALL include a meta-test that compares discovered registry keys with conformance parameter coverage.

#### Scenario: Discovered key missing from conformance collection
- **WHEN** discovery returns a registry key for a category covered by conformance tests
- **THEN** the meta-test SHALL fail if no corresponding conformance parameter case exists for that key
