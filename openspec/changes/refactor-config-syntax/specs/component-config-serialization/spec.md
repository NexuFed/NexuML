## ADDED Requirements

### Requirement: Component strings exist only in serialized configuration
The system SHALL lower concrete component definitions to stable registered identities when producing portable YAML/JSON configuration and SHALL restore those identities to concrete definition types when loading configuration.

#### Scenario: Typed layer is serialized
- **WHEN** Python config contains `LayerSpec(component=LMBE(n_mels=64), ...)`
- **THEN** serialized config SHALL contain the registered LMBE identity, component version, and validated parameter values as plain YAML/JSON-safe data
- **AND** SHALL NOT require a Python import path for `LMBE`.

#### Scenario: Serialized layer is restored
- **WHEN** NexuML loads a serialized layer containing a known layer identity/version and parameters
- **THEN** discovery/registry lookup SHALL resolve the concrete definition class
- **AND** Pydantic validation SHALL restore an instance of that exact definition class before compilation.

### Requirement: Component identity is explicit and versioned
Every serialized registered component SHALL carry an explicit stable registration identity and version independent of the Python class import path.

#### Scenario: Python component module is refactored
- **WHEN** a component definition moves to another Python module while retaining its registered kind/name/version
- **THEN** existing config written with that identity SHALL not depend on the old module import path.

#### Scenario: Unknown component version is loaded
- **WHEN** serialized config references a component version not registered in the active environment
- **THEN** loading SHALL fail with an actionable unknown-component/version error
- **AND** the system SHALL NOT silently substitute another version.

### Requirement: ResolvedConfig round-trips concrete component types
`ResolvedConfig` SHALL serialize component definitions as portable data and restore them to concrete definition objects.

#### Scenario: Scenario round-trip succeeds
- **WHEN** a `ScenarioSpec` containing registered layer, data source, evaluation algorithm, and loader backend definitions is converted to YAML and loaded again
- **THEN** all non-component scenario values SHALL retain their meaning
- **AND** each registered component SHALL restore to the same concrete definition type and equivalent field values.

#### Scenario: Resolved config is exported as a sidecar
- **WHEN** NexuML writes `resolved_config.yaml` or stores resolved config in export metadata
- **THEN** the config SHALL remain plain portable data
- **AND** it SHALL use the same registered component identity/version contract as ordinary configuration serialization.

### Requirement: Serialization is generic rather than component-specific
The system SHALL use one generic component lowering/restoration mechanism based on the component registry and Pydantic definition fields.

#### Scenario: New custom component is installed
- **WHEN** a custom library registers a new definition class and uses only supported serializable definition field types
- **THEN** that component SHALL serialize and restore without adding a component-specific serializer branch to NexuML core.

#### Scenario: Component schema changes internally
- **WHEN** a concrete definition adds or changes a validated field within its registered version during development
- **THEN** serialization SHALL obtain fields from the concrete definition model
- **AND** SHALL NOT require updating a duplicate central config schema.

### Requirement: Old selector config syntax is not a second supported format
The system SHALL expose one canonical component config syntax after NEX-211 and SHALL NOT retain the previous selector-plus-params syntax as a compatibility path.

#### Scenario: Old Python layer syntax is used
- **WHEN** code attempts to construct the new layer specification with removed `type_key` plus component `params`
- **THEN** the construction SHALL fail rather than translating silently to a typed definition.

#### Scenario: Old serialized selector document is loaded
- **WHEN** a config document uses fields removed by NEX-211 instead of the new serialized component structure
- **THEN** normal config validation SHALL reject it
- **AND** NexuML SHALL NOT run a legacy parser, alias translator, or automatic migration step.

### Requirement: Persisted identity is not automatically derived from class names
The system SHALL treat the explicit decorator/registry key as the persisted identity.

#### Scenario: Definition class is renamed
- **WHEN** a Python definition class is renamed but its explicit registered identity is intentionally preserved
- **THEN** the serialized identity SHALL remain the explicit registered identity
- **AND** the serializer SHALL NOT derive the persisted key from `__name__`.
