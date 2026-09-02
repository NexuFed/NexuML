## ADDED Requirements

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

### Requirement: External framework references are not wrapped without need
The system SHALL NOT require every configurable external framework class to become a module-specific NexuML component definition.

#### Scenario: User selects a PyTorch optimizer or Lightning callback
- **WHEN** configuration references an external optimizer, scheduler, or callback class through an existing explicit import/alias mechanism
- **THEN** NEX-211 MAY leave that mechanism unchanged
- **AND** SHALL NOT introduce NexuML wrapper classes solely to remove every string from configuration.

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
