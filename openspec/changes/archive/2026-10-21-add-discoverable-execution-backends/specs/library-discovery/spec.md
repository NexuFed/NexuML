# Library Discovery Delta

## MODIFIED Requirements

### Requirement: Decorated object discovery
The system SHALL discover layers, data sources, scenarios, evaluation algorithms and execution backend definitions from objects explicitly marked with NexuML registration decorators. Execution definitions SHALL be discovered from the core package and the same installed-library/local-root sources without requiring a separate API registration path.

#### Scenario: Decorated layer is discovered
- **WHEN** a scanned library module contains a class decorated as a layer
- **THEN** the system registers that class in the layer registry using the decorator key

#### Scenario: Decorated data source is discovered
- **WHEN** a scanned library module contains a class decorated as a data source
- **THEN** the system registers that class in the data registry using the decorator key

#### Scenario: Decorated scenario is discovered
- **WHEN** a scanned library module contains a function decorated as a scenario
- **THEN** the system registers that function in the scenario registry using the decorator key

#### Scenario: Decorated evaluation algorithm is discovered
- **WHEN** a scanned library module contains a class decorated as an evaluation algorithm
- **THEN** the system registers that class in the evaluation registry using the decorator key

#### Scenario: Decorated execution backend is discovered
- **WHEN** a scanned core or library module contains a registered execution backend definition
- **THEN** NexuML exposes its registered identity and configuration schema through the execution catalog used by Python, CLI and API

#### Scenario: Core execution definitions need no base library
- **WHEN** NexuML is installed without the optional base library or local API
- **THEN** core execution definitions remain discoverable and Local remains usable
