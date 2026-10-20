# Spec: decorator-discovery-docs

## Purpose

Documentation covering NexuML decorators and component discovery, providing a learning page that users encounter before extension how-to guides.

## Requirements

### Requirement: Decorator Learning Page

The documentation SHALL include a learning page that explains NexuML decorators and discovery before users reach extension how-to guides.

#### Scenario: User asks how NexuML finds components

- **WHEN** a user opens the decorators and discovery learning page
- **THEN** the page explains that NexuML discovers decorated components from installed packages, entry-point libraries, and local library roots.

### Requirement: Scenario Decorator Explained

The documentation SHALL explain `@scenario`.

#### Scenario: User registers a scenario

- **WHEN** the docs introduce `@scenario("key")`
- **THEN** they explain that the decorated function becomes available by key to CLI commands such as resolve and train.

### Requirement: Layer Decorator Explained
The documentation SHALL explain that `@layer` registers a typed `LayerDefinition` under a stable persisted identity while normal Python authoring uses the definition class directly.

#### Scenario: User adds a custom layer
- **WHEN** the docs introduce `@layer("key")`
- **THEN** they SHALL show a public typed `LayerDefinition` with explicit Pydantic fields and a direct runtime build/materialization method
- **AND** they SHALL show scenario authoring with `LayerSpec(component=MyLayer(...), ...)`
- **AND** they SHALL explain that the decorator key is used for discovery and serialized YAML identity rather than as the primary Python selector.

### Requirement: Data Source Decorator Explained
The documentation SHALL explain that `@data_source` registers a typed `DataSourceDefinition` while normal Python authoring uses the concrete definition value.

#### Scenario: User adds a custom dataset source
- **WHEN** the docs introduce `@data_source("key")`
- **THEN** they SHALL show a typed data source definition with explicit fields
- **AND** they SHALL show the definition passed directly to the relevant `DataSpec`/`DatasetSpec` field
- **AND** they SHALL NOT teach `source_type="key"` plus an arbitrary source `params` dict as the Python authoring API.

### Requirement: Evaluation Algorithm Decorator Explained
The documentation SHALL explain that `@eval_algorithm` registers a typed `EvalAlgorithmDefinition` while the surrounding evaluation spec retains evaluation routing/wiring fields.

#### Scenario: User adds evaluation logic
- **WHEN** the docs introduce `@eval_algorithm("key")`
- **THEN** they SHALL show a concrete typed evaluation definition passed to the evaluation algorithm spec
- **AND** they SHALL distinguish intrinsic algorithm parameters from surrounding fields such as name/enabled/axis/feature/label routing
- **AND** they SHALL NOT teach algorithm `type="key"` plus an arbitrary component `params` dict as the Python authoring API.

### Requirement: Registry Verification Commands

The decorator learning page SHALL show how to verify discovery through registry commands.

#### Scenario: User verifies registered components

- **WHEN** a user has installed or added a library
- **THEN** the docs show verification commands for scenarios, layers, data sources, and evaluation algorithms using `nexuml registry list`.

### Requirement: Decorator Docs Link To Reference And How-To Pages

The decorator learning page SHALL link to detailed reference and task pages.

#### Scenario: User needs exact decorator API

- **WHEN** a user needs exact syntax and compact reference
- **THEN** the learning page links to `docs/reference/decorators.md`.

#### Scenario: User wants to build an extension

- **WHEN** a user wants to add a layer, data source, scenario, or library
- **THEN** the learning page links to the corresponding how-to guide.

### Requirement: Python and serialized component syntax are distinguished
The documentation SHALL explicitly explain that typed Python authoring and portable serialized configuration intentionally use different representations.

#### Scenario: User compares Python and YAML examples
- **WHEN** documentation shows the same component in Python and YAML
- **THEN** the Python example SHALL use a concrete definition value such as `LMBE(...)`
- **AND** the YAML example SHALL use the stable registered identity/version/parameter representation
- **AND** the docs SHALL explain that registry strings are a persistence/discovery contract rather than the preferred Python API.

### Requirement: Definition/runtime separation is documented
The extension documentation SHALL explain the separation between immutable component definitions and mutable runtime implementations where runtime state is required.

#### Scenario: User implements a stateful PyTorch layer
- **WHEN** a custom layer needs parameters, submodules, buffers, or lifecycle state
- **THEN** the docs SHALL show one public typed definition and one private/runtime `PipelineLayer` implementation as needed
- **AND** SHALL NOT instruct users to combine Pydantic config state and mutable `torch.nn.Module` state in one object.

### Requirement: Loader backend typed configuration is documented
If loader backends are public extension points after NEX-211, the documentation SHALL show backend-specific options through typed `LoaderBackendDefinition` values while keeping common loader policy on `LoaderSpec`.

#### Scenario: User selects a loader backend
- **WHEN** a Python example configures a registered loader backend
- **THEN** the example SHALL pass a concrete backend definition to `LoaderSpec`
- **AND** the serialized example SHALL show its stable registered identity.

### Requirement: Direct PyTorch module authoring is documented
The layer extension documentation SHALL distinguish ordinary direct PyTorch modules from registered NexuML semantic layers.

#### Scenario: User wraps an ordinary PyTorch module
- **WHEN** documentation shows a one-input/one-output module such as `torch.nn.Dropout`
- **THEN** it SHALL prefer `LayerSpec(component=nn_module(torch.nn.Dropout, p=0.5), ...)`
- **AND** explain that the module factory remains directly navigable and needs no module-specific decorator or library registration
- **AND** show the corresponding stable `NnModule` YAML representation.

#### Scenario: User chooses between direct and registered authoring
- **WHEN** documentation introduces custom layers
- **THEN** it SHALL direct ordinary importable tensor-to-tensor modules to `nn_module(...)`
- **AND** direct context-, label-, metadata-, lifecycle-, multi-input-, or multi-output-dependent behavior to a registered typed `LayerDefinition`.

#### Scenario: User reviews direct-module portability limits
- **WHEN** documentation explains resolved config/export behavior
- **THEN** it SHALL state that factories must be top-level/importable, constructor values must be YAML/JSON-safe, and external targets are trusted executable references
- **AND** SHALL state that live module instances, lambdas, closures, and local definitions are unsupported.
