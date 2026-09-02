## MODIFIED Requirements

### Requirement: Decorated object discovery
The system SHALL discover typed component definitions and scenario recipes from objects explicitly marked with NexuML registration decorators.

#### Scenario: Decorated layer definition is discovered
- **WHEN** a scanned library module contains a `LayerDefinition` class decorated as a layer
- **THEN** the system SHALL register that definition type in the common component registry using the explicit decorator key and version
- **AND** normal Python scenario authoring SHALL use the definition class directly rather than the key string.

#### Scenario: Decorated layer is discovered
- **WHEN** a scanned library module contains a `LayerDefinition` class decorated as a layer
- **THEN** the system SHALL register that definition type in the common component registry using the decorator key and version.

#### Scenario: Decorated data source definition is discovered
- **WHEN** a scanned library module contains a `DataSourceDefinition` class decorated as a data source
- **THEN** the system SHALL register that definition type in the common component registry using the explicit decorator key and version.

#### Scenario: Decorated data source is discovered
- **WHEN** a scanned library module contains a `DataSourceDefinition` class decorated as a data source
- **THEN** the system SHALL register that definition type in the common component registry using the decorator key and version.

#### Scenario: Decorated scenario is discovered
- **WHEN** a scanned library module contains a function decorated as a scenario
- **THEN** the system SHALL register that function as a scenario recipe using the decorator key
- **AND** scenario recipe discovery MAY remain separate from component-definition lookup.

#### Scenario: Decorated evaluation algorithm definition is discovered
- **WHEN** a scanned library module contains an `EvalAlgorithmDefinition` class decorated as an evaluation algorithm
- **THEN** the system SHALL register that definition type in the common component registry using the explicit decorator key and version.

#### Scenario: Decorated evaluation algorithm is discovered
- **WHEN** a scanned library module contains an `EvalAlgorithmDefinition` class decorated as an evaluation algorithm
- **THEN** the system SHALL register that definition type in the common component registry using the decorator key and version.

#### Scenario: Registered loader backend definition is discovered
- **WHEN** a scanned library exposes a registered `LoaderBackendDefinition`
- **THEN** the system SHALL make that definition type available through the same component identity/discovery mechanism used for other typed component roles.

### Requirement: Registry conflict reporting
The system SHALL reject conflicting registration identities for the same component kind/name/version when they refer to different definition types.

#### Scenario: Two libraries define the same layer identity
- **WHEN** two discovered layer definitions use the same registration key and version and are not the same definition class
- **THEN** registry loading SHALL fail with an error identifying both definition modules/types.

#### Scenario: Two libraries define same layer key
- **WHEN** two discovered layer definitions use the same registration key and version and are not the same definition class
- **THEN** registry loading SHALL fail with an error identifying both definition modules/types.

#### Scenario: Same definition is encountered more than once
- **WHEN** discovery encounters the same definition class with the same kind/name/version through overlapping scan paths
- **THEN** registration MAY be idempotent
- **AND** SHALL NOT create duplicate registry entries.

### Requirement: Core and base library separation
The system SHALL keep framework component-definition/runtime infrastructure in the `nexuml` core package and reusable concrete component definitions in the separate same-repository base library package.

#### Scenario: Core package is installed alone
- **WHEN** only the core package is installed
- **THEN** NexuML framework commands and component base/registry/serialization infrastructure SHALL remain available without requiring the base library package.

#### Scenario: Base library package is installed
- **WHEN** the same-repository base library package is installed
- **THEN** its decorated layer definitions, data source definitions, scenario recipes, evaluation definitions, and registered loader definitions SHALL be discoverable through the standard registry loading path.

#### Scenario: Base component is used in Python
- **WHEN** a user imports a concrete component such as `LMBE` from the base library
- **THEN** that public symbol SHALL be the typed authoring definition
- **AND** any mutable runtime implementation required by the definition SHOULD remain an implementation detail of the component module.
