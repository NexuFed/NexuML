## ADDED Requirements

### Requirement: Runtime materialization is explicit
The system SHALL materialize mutable NexuML runtime objects from typed component definitions through explicit role-specific build/materialization APIs.

#### Scenario: Pipeline compiler creates a layer
- **WHEN** the compiler processes a `LayerSpec` containing a typed layer definition
- **THEN** it SHALL construct the layer materialization context and call the definition's build/materialization method
- **AND** it SHALL NOT resolve the layer by a `type_key` string during normal already-typed Python compilation.

#### Scenario: Layer needs compiler-derived values
- **WHEN** a layer requires inferred input shapes, TensorDict key wiring, `num_classes`, metadata, shared storage, or layer scheduling state
- **THEN** those values SHALL be supplied through explicit build context/spec fields
- **AND** the compiler SHALL NOT inspect the runtime `__init__` signature to decide which values can be injected.

### Requirement: Component registry owns identity only
The component registry SHALL store registration/discovery identity and definition types without duplicating definition configuration or runtime construction logic.

#### Scenario: Component is restored from YAML
- **WHEN** the serializer asks for a registered `(kind, name, version)`
- **THEN** the registry SHALL return the corresponding definition type
- **AND** the definition type SHALL validate its own fields.

#### Scenario: Component is compiled from Python
- **WHEN** a component definition object already exists in a Python `ScenarioSpec`
- **THEN** runtime materialization SHALL use that object directly
- **AND** SHALL NOT convert it back to a string and look it up again merely to construct the runtime.

#### Scenario: Registry sees arbitrary parameter values
- **WHEN** component-specific parameter values are validated
- **THEN** validation SHALL be performed by the concrete definition model
- **AND** the registry SHALL NOT implement constructor signature validation/casting.

### Requirement: Mutable runtime behavior remains behaviorally equivalent
NEX-211 SHALL preserve current runtime semantics while changing the configuration/materialization boundary.

#### Scenario: Typed layer pipeline is compiled
- **WHEN** an existing built-in scenario is migrated to typed definitions
- **THEN** shape propagation, TensorDict `keys_in`/`keys_out`, metadata handoff, stage skipping, loss/metric wiring, and layer lifecycle behavior SHALL continue to operate with the same semantics unless a separately identified bug is fixed explicitly.

#### Scenario: Data/evaluation/loader component is materialized
- **WHEN** a migrated data source, evaluation algorithm, or loader backend is used
- **THEN** NEX-211 SHALL preserve its existing data/evaluation/loading behavior
- **AND** SHALL limit changes to configuration typing, identity lookup, and explicit materialization plumbing.

### Requirement: Definition/runtime architecture remains small
The materialization design SHALL avoid reflective or generated abstractions that recreate the complexity removed by typed definitions.

#### Scenario: Developer adds a new layer component
- **WHEN** a developer implements a new registered layer
- **THEN** the normal implementation SHALL require one public typed definition and, only when needed, one runtime implementation plus a direct build method
- **AND** SHALL NOT require a generated config class, metaclass, duplicated registry schema, or registration-time constructor inspection.

#### Scenario: Runtime context grows
- **WHEN** a component role needs an additional framework-owned runtime input
- **THEN** that input MAY be added deliberately to the role-specific materialization context
- **AND** the context SHALL NOT become a generic dependency-injection/service-locator container.
