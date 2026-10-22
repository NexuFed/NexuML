## ADDED Requirements

### Requirement: Parameter schemas and native I/O descriptions are separate

The existing component catalog SHALL expose native I/O descriptions separately from parameter JSON Schema. Descriptions SHALL identify declared input domains, parameter or context bindings, collection shape, requiredness and applicable defaults, plus known source labels. The API SHALL derive them from selected-installation contracts rather than custom schema keywords or component-name tables.

#### Scenario: Client discovers a typed evaluator
- **WHEN** the catalog contains an evaluator with native input declarations
- **THEN** its ordinary parameter schema describes configuration fields and its separate I/O description identifies the declared parameter and context inputs
- **AND** scalar, optional and collection bindings retain their distinct shapes
- **AND** identity, version, provenance and discovery diagnostics remain available

#### Scenario: Client discovers a data source
- **WHEN** a source declares fixed label outputs
- **THEN** the catalog exposes those known label keys separately from its parameter schema
- **AND** a source with unknown outputs is distinguished from a source declaring no labels

#### Scenario: A contributed component lacks native I/O information
- **WHEN** discovery finds an otherwise valid contributed definition without a native static contract
- **THEN** its ordinary schema and identity remain available with I/O explicitly unresolved
- **AND** neither the API nor its clients guess ports from field-name suffixes or a hard-coded component map

### Requirement: Contract discovery does not construct runtimes

Catalog inspection SHALL derive schemas and I/O bindings from public definitions and already declared configuration facts without calling component builds, inspecting runtime constructor signatures, downloading data, allocating models or starting work. Existing trusted-code library import semantics and transport/file protections SHALL remain unchanged.

#### Scenario: Definition has required parameters
- **WHEN** discovery inspects a definition that cannot be instantiated without user parameters
- **THEN** it describes the definition without inventing an instance or executing its build method

#### Scenario: Dataset construction would download or scan data
- **WHEN** the catalog exposes that dataset's available contract
- **THEN** discovery does not construct the dataset, download files or scan its configured data paths
- **AND** unavailable static outputs remain unknown

#### Scenario: Library imports trusted Python
- **WHEN** an installed or configured library is imported for existing discovery
- **THEN** the API continues to identify that operation as trusted local code, not sandboxed inspection
- **AND** no authentication, origin, filesystem or process-ownership protection is weakened for the new contract fields

### Requirement: Native contracts do not change configuration persistence

I/O descriptions SHALL be derived information, not additional executable YAML fields. Existing valid component identities, parameter values, surrounding routing and ordered configuration SHALL round-trip unchanged. The catalog SHALL describe the selected installation without introducing a second registry or copying UI-derived state into runtime configuration.

#### Scenario: User saves and reopens a scenario
- **WHEN** a valid configuration is loaded, inspected through native contracts and saved
- **THEN** its semantic identity, parameter values, routing and stage order are unchanged
- **AND** neither native descriptors nor UI grouping values are persisted into the executable configuration

#### Scenario: User edits a parameter-backed route
- **WHEN** Studio changes a route described as an intrinsic parameter binding
- **THEN** the request updates that existing parameter field with its original scalar or collection representation
- **AND** context-backed routes continue to update the surrounding evaluation fields instead
