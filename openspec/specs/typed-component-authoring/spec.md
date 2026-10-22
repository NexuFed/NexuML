# typed-component-authoring Specification

## Purpose

Define typed Python authoring for NexuML components and portable external framework factories while separating configuration from runtime state.

## Requirements

### Requirement: Typed component definitions are the primary Python authoring API
The system SHALL represent NexuML-owned configurable library components as concrete typed definition values in Python rather than registry selector strings plus arbitrary parameter dictionaries. Ordinary external PyTorch modules that satisfy the direct-module contract MAY instead be authored through the typed `nn_module(...)` helper without a module-specific NexuML definition.

#### Scenario: User authors a pipeline layer
- **WHEN** a user adds an LMBE layer to a Python `PipelineSpec`
- **THEN** the user SHALL pass a concrete `LMBE(...)` definition to the layer specification
- **AND** the component-specific fields SHALL be declared on `LMBE` itself
- **AND** the Python authoring path SHALL NOT require `type_key="LMBE"` plus a component `params` dict.

#### Scenario: User navigates component configuration
- **WHEN** a user navigates from a component symbol such as `LMBE` in an IDE
- **THEN** the symbol SHALL resolve to the public definition class that declares its configurable fields, defaults, validation, and documentation.

#### Scenario: Invalid component parameter is provided
- **WHEN** a user passes an unknown or invalid component-specific field to a definition
- **THEN** validation SHALL fail when the definition is constructed
- **AND** validation SHALL NOT be deferred until registry lookup or runtime materialization.

### Requirement: Component definitions are immutable portable values
NexuML component definitions SHALL contain semantic configuration only and SHALL be safe to validate and serialize as Pydantic values.

#### Scenario: Definition schema is inspected
- **WHEN** tooling calls `model_json_schema()` on a concrete definition class
- **THEN** the returned schema SHALL describe that component's actual configurable fields and constraints
- **AND** the schema SHALL NOT be synthesized from a runtime constructor signature.

#### Scenario: Runtime state is created
- **WHEN** a typed component is materialized for execution
- **THEN** tensors, modules, loaded data, shared storage, trainer state, and other mutable/live state SHALL exist on the runtime object rather than the serializable definition.

### Requirement: Public definition and mutable runtime are separate
The system SHALL separate public immutable component definitions from mutable execution implementations for roles that own runtime state.

#### Scenario: Layer definition is materialized
- **WHEN** `LMBE(...)` is compiled
- **THEN** `LMBE` SHALL remain the public typed definition value
- **AND** it SHALL materialize a `PipelineLayer` runtime implementation
- **AND** the runtime implementation MAY be private to the same module.

#### Scenario: Component module is organized
- **WHEN** a library component requires a separate runtime implementation
- **THEN** the definition and its private runtime implementation SHOULD be colocated in the same module
- **AND** the framework SHALL NOT require a parallel public `LMBEConfig`, `LMBE.Definition`, or generated `ConfigModel` hierarchy.

### Requirement: Graph wiring remains separate from component semantics
The system SHALL keep graph placement and wiring on surrounding specification objects rather than duplicating those fields into every component definition.

#### Scenario: Layer is connected to TensorDict keys
- **WHEN** a layer is authored with `keys_in`, `keys_out`, `meta_in`, or `meta_out`
- **THEN** those values SHALL belong to `LayerSpec`/graph placement
- **AND** `LMBE` or another concrete layer definition SHALL contain only parameters intrinsic to the layer implementation.

#### Scenario: Runtime-only compiler value is required
- **WHEN** a component requires inferred input shapes, `num_classes`, shared storage, or another compiler/runtime value
- **THEN** that value SHALL be provided through explicit materialization context
- **AND** SHALL NOT be persisted as ordinary component configuration solely to satisfy a runtime constructor.

### Requirement: Typed authoring applies to current NexuML-owned plugin roles
The system SHALL apply the typed-definition authoring model to current NexuML-owned plugin roles that otherwise use a selector plus arbitrary component parameters.

#### Scenario: Data source is authored
- **WHEN** a scenario configures a registered data source or dataset entry in Python
- **THEN** it SHALL use a concrete `DataSourceDefinition` value rather than `source_type`/`type_key` plus an untyped component parameter bag.

#### Scenario: Evaluation algorithm is authored
- **WHEN** a scenario configures a registered evaluation algorithm in Python
- **THEN** it SHALL use a concrete `EvalAlgorithmDefinition` value
- **AND** routing fields such as enabled/name/axis/feature/label wiring SHALL remain on the surrounding evaluation spec when they are not intrinsic algorithm parameters.

#### Scenario: Loader backend is authored
- **WHEN** a scenario selects a NexuML loader backend in Python
- **THEN** it SHALL use a concrete `LoaderBackendDefinition` value for backend identity and backend-specific parameters
- **AND** common loader policy SHALL remain on `LoaderSpec`.

### Requirement: External framework factories use typed portable helpers
The system SHALL NOT require every configurable external framework class to become a module-specific NexuML component definition, but Python authoring SHALL reference the real factory symbol instead of a selector-plus-params bag.

#### Scenario: User selects a PyTorch optimizer or Lightning callback
- **WHEN** configuration references an external optimizer, scheduler, or callback class
- **THEN** it SHALL use the corresponding `optimizer(...)`, `scheduler(...)`, or `callback(...)` helper with the real importable factory symbol and direct constructor arguments
- **AND** SHALL NOT introduce one NexuML wrapper class per external implementation.

#### Scenario: User configures a strategy or preprocessing writer
- **WHEN** strategy- or writer-specific constructor settings are required
- **THEN** Python authoring SHALL use `strategy(RealStrategy, ...)` or `writer(RealBackend, ...)`
- **AND** SHALL NOT expose `strategy_params` or `writer_params` dictionaries.

### Requirement: Ordinary PyTorch modules can be authored directly
The system SHALL provide one universal `nn_module(factory, *args, **kwargs)` authoring helper for importable factories that construct ordinary `torch.nn.Module` values without requiring each factory/module type to be added to `nexuml_library` or registered separately.

#### Scenario: User adds a standard PyTorch module
- **WHEN** a user authors `LayerSpec(component=nn_module(torch.nn.Dropout, p=0.5), ...)`
- **THEN** the Python configuration SHALL contain a real navigable `torch.nn.Dropout` symbol
- **AND** the helper SHALL return the single registered universal layer definition
- **AND** no `Dropout`-specific NexuML definition or registry entry SHALL be required.

#### Scenario: User navigates a direct module
- **WHEN** a user navigates the factory symbol passed to `nn_module(...)`
- **THEN** the IDE SHALL resolve that real Python class/function rather than a registry selector string
- **AND** the helper signature SHOULD preserve the factory constructor signature for static checking where the configured type checker supports `ParamSpec` callable inference.

#### Scenario: User supplies a live module instance
- **WHEN** a user passes an already-created `torch.nn.Module` instance instead of an importable factory and portable constructor values
- **THEN** construction SHALL fail with an error explaining that live instances cannot be reconstructed from resolved YAML
- **AND** the system SHALL NOT silently pickle or introspect that instance to synthesize configuration.

### Requirement: Direct modules remain a narrow external-module path
The direct-module helper SHALL be limited to ordinary single-tensor transformations and SHALL NOT replace registered definitions that own NexuML-specific semantics.

#### Scenario: Layer requires NexuML-specific behavior
- **WHEN** a layer consumes labels or metadata, requires compiler-derived context in its constructor, publishes auxiliary outputs, accepts multiple tensors, returns non-tensor structures, or owns custom lifecycle behavior
- **THEN** it SHALL use a registered typed `LayerDefinition` and an appropriate runtime implementation
- **AND** the universal direct-module path SHALL NOT add reflection or generic hooks to emulate those behaviors.

#### Scenario: Redundant built-in wrappers are removed
- **WHEN** the direct-module path is available
- **THEN** `IdentityLayer`, `Dropout`, and `Flatten` SHALL be replaced by `nn_module(torch.nn.Identity)`, `nn_module(torch.nn.Dropout, ...)`, and `nn_module(torch.nn.Flatten, start_dim=1, end_dim=-1)` respectively
- **AND** compatibility aliases for their old Python symbols SHALL NOT remain.

### Requirement: Native component contracts without schema extras

Component authors SHALL express semantic configuration and supported I/O contracts through typed Python declarations, without authored `json_schema_extra` or `x-nexuml-outputs`, `x-nexuml-routing` and `x-nexuml-category` schema keywords. Generated parameter schemas SHALL continue to describe actual fields and constraints. Graph placement and wiring SHALL remain on the surrounding scenario specs.

#### Scenario: Author declares an intrinsic input-key parameter
- **WHEN** an evaluator declares a scalar, optional or collection parameter that consumes feature or label keys
- **THEN** its Python type identifies that semantic role without a separate schema dictionary naming the parameter
- **AND** ordinary parameter values still serialize as strings, nulls and lists in the existing configuration shape

#### Scenario: Author declares surrounding graph wiring
- **WHEN** a scenario selects layer or evaluator wiring
- **THEN** existing placement, feature, label, alias, axis and metadata fields remain on their current surrounding specs
- **AND** no additional component configuration model or duplicated routing fields are required

#### Scenario: Definition schema is generated
- **WHEN** a built-in definition exposes its parameter schema
- **THEN** the schema contains its native fields, defaults and validation constraints without custom NexuML schema keywords
- **AND** immutable definitions and rejection of unknown fields remain intact

### Requirement: I/O facts are shared with runtime construction

Known source labels and evaluator context-input defaults SHALL have one native declaration used by runtime construction and tooling. Explicit scenario overrides SHALL retain precedence. Tooling SHALL distinguish unknown contracts from known-empty contracts and SHALL NOT infer arbitrary Python behavior from parameter names or by constructing runtimes.

#### Scenario: Dataset declares fixed labels
- **WHEN** a built-in image dataset declares its fixed label keys
- **THEN** its runtime label names and tooling-visible outputs derive from the same declaration
- **AND** no parallel runtime constant and UI-only label list need to be maintained

#### Scenario: Evaluator uses a context default
- **WHEN** a supported evaluator receives no feature or label override
- **THEN** runtime construction and tooling resolve the same existing fallback key
- **AND** omission, null and empty-value behavior are preserved for that evaluator

#### Scenario: Scenario overrides a context default
- **WHEN** an evaluator has a declared fallback and the scenario supplies an explicit key
- **THEN** both runtime construction and tooling use the scenario key without rewriting the saved configuration or the component's default

#### Scenario: Source labels require runtime information
- **WHEN** labels cannot be determined from a native fixed declaration or already explicit configuration
- **THEN** tooling reports them as unknown rather than as an empty set or fabricated keys
- **AND** ordinary configuration editing and the existing explicit build/run path remain available

#### Scenario: External component supplies no static I/O contract
- **WHEN** an otherwise valid contributed definition provides parameters but no native I/O declaration
- **THEN** it remains discoverable, serializable and usable by the existing Python runtime
- **AND** tooling exposes its parameter controls while retaining the unresolved I/O limitation

### Requirement: Fixed parameter choices are actual validation constraints

Supported fixed-choice parameters SHALL declare their accepted values through native Python types. Their generated schemas and Python/configuration validation SHALL agree. Valid choices SHALL retain their existing serialized values; invalid choices SHALL fail at definition validation instead of being accepted solely because schema hints are not enforced by the runtime model.

#### Scenario: User selects a supported visualization method or model variant
- **WHEN** a definition receives a supported fixed-choice value
- **THEN** it validates and round-trips as the same ordinary configuration value
- **AND** the generated schema advertises that value to generic controls

#### Scenario: User supplies an unsupported fixed-choice value
- **WHEN** a definition or configuration is validated with a choice outside the supported set
- **THEN** validation rejects it before model or evaluator construction
- **AND** no separately authored schema enumeration is needed
