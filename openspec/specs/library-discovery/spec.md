# library-discovery Specification

## Purpose
TBD - created by archiving change normalize-dataset-roots-downloads. Update Purpose after archive.
## Requirements
### Requirement: Library data builders provide canonical dataset policies
The base library SHALL provide canonical data builder functions for reusable datasets, and those builders SHALL be the primary place for dataset root defaults, layout defaults, and download defaults.

#### Scenario: Scenario uses library data builder policy
- **WHEN** a composed scenario needs DCASE, AudioSet, CIFAR, or synthetic data
- **THEN** it calls the corresponding library data builder instead of duplicating root or layout defaults inline

#### Scenario: User overrides dataset policy intentionally
- **WHEN** a user needs a custom dataset root or custom dataset subset
- **THEN** the data builder accepts explicit override arguments without requiring edits to composed scenario internals

### Requirement: Decorated object discovery
The system SHALL discover layers, data sources, scenarios, and evaluation algorithms from objects explicitly marked with NexuML registration decorators.

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

### Requirement: Installed library package discovery
The system SHALL discover installed NexuML library packages advertised through the `nexuml.libraries` Python entry point group.

#### Scenario: Installed base library is available
- **WHEN** a package exposes a `nexuml.libraries` entry point and is installed in the active environment
- **THEN** NexuML scans the entry point target for decorated objects during registry loading

#### Scenario: No installed libraries exist
- **WHEN** no packages expose `nexuml.libraries` entry points
- **THEN** registry loading completes without failing because of missing optional libraries

### Requirement: Local library root discovery
The system SHALL discover decorated objects from user-configured local library root folders.

#### Scenario: Registered local library contains decorated modules
- **WHEN** a local library root has been added and contains decorated Python modules
- **THEN** NexuML scans the current files under that root and registers discovered objects

#### Scenario: Registered local library changes after being added
- **WHEN** files inside a registered local library root are edited after the root was added
- **THEN** the next NexuML CLI run discovers the current file contents without requiring a refresh command

### Requirement: No persistent discovery cache
The system SHALL NOT persist discovered registry objects as a cache for reuse across CLI runs.

#### Scenario: Registry is loaded on a new CLI run
- **WHEN** a NexuML CLI command starts and loads registries
- **THEN** it scans configured libraries and installed library packages instead of reading previously cached registry contents

### Requirement: Flexible data source layout
The system SHALL support decorated data sources in any Python module under a scanned library root, including direct `data` modules and nested `data/sources` packages.

#### Scenario: Data source is directly under data folder
- **WHEN** `library/data/dcase.py` contains a decorated data source
- **THEN** the data source is discoverable and registered

#### Scenario: Data source is in nested source folder
- **WHEN** `library/data/sources/dcase.py` contains a decorated data source
- **THEN** the data source is discoverable and registered

#### Scenario: Data source spans a package folder
- **WHEN** `library/data/dcase/` is a Python package containing a decorated data source
- **THEN** the data source is discoverable and registered

### Requirement: Registry conflict reporting
The system SHALL reject conflicting registration keys for the same object kind when they refer to different objects.

#### Scenario: Two libraries define same layer key
- **WHEN** two discovered layer objects use the same registration key and are not the same class
- **THEN** registry loading fails with an error identifying both object modules

### Requirement: Core and base library separation
The system SHALL keep framework code in the `nexuml` core package and base reusable content in a separate same-repository library package.

#### Scenario: Core package is installed alone
- **WHEN** only the core package is installed
- **THEN** NexuML framework commands remain available without requiring the base library package

#### Scenario: Base library package is installed
- **WHEN** the same-repository base library package is installed
- **THEN** its decorated layers, data sources, scenarios, and evaluation algorithms are discoverable through the standard registry loading path

