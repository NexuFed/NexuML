## ADDED Requirements

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
