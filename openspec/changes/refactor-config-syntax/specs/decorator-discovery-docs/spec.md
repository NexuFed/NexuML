## MODIFIED Requirements

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

## ADDED Requirements

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
