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
The system SHALL discover typed component definitions, scenario recipes and execution backend definitions from objects explicitly marked with NexuML registration decorators. Execution definitions SHALL be discovered from the core package and the same installed-library/local-root sources without requiring a separate API registration path.

#### Scenario: Decorated layer definition is discovered
- **WHEN** a scanned library module contains a `LayerDefinition` class decorated as a layer
- **THEN** the system SHALL register that definition type in the common component registry using the explicit decorator key and version
- **AND** normal Python scenario authoring SHALL use the definition class directly rather than the key string.

#### Scenario: Decorated layer is discovered
- **WHEN** a scanned library module contains a class decorated as a layer
- **THEN** the system registers that class in the layer registry using the decorator key

#### Scenario: Decorated data source definition is discovered
- **WHEN** a scanned library module contains a `DataSourceDefinition` class decorated as a data source
- **THEN** the system SHALL register that definition type in the common component registry using the explicit decorator key and version.

#### Scenario: Decorated data source is discovered
- **WHEN** a scanned library module contains a class decorated as a data source
- **THEN** the system registers that class in the data registry using the decorator key

#### Scenario: Decorated scenario is discovered
- **WHEN** a scanned library module contains a function decorated as a scenario
- **THEN** the system registers that function in the scenario registry using the decorator key

#### Scenario: Decorated evaluation algorithm definition is discovered
- **WHEN** a scanned library module contains an `EvalAlgorithmDefinition` class decorated as an evaluation algorithm
- **THEN** the system SHALL register that definition type in the common component registry using the explicit decorator key and version.

#### Scenario: Decorated evaluation algorithm is discovered
- **WHEN** a scanned library module contains a class decorated as an evaluation algorithm
- **THEN** the system registers that class in the evaluation registry using the decorator key

#### Scenario: Registered loader backend definition is discovered
- **WHEN** a scanned library exposes a registered `LoaderBackendDefinition`
- **THEN** the system SHALL make that definition type available through the same component identity/discovery mechanism used for other typed component roles.

#### Scenario: Decorated execution backend is discovered
- **WHEN** a scanned core or library module contains a registered execution backend definition
- **THEN** NexuML exposes its registered identity and configuration schema through the execution catalog used by Python, CLI and API

#### Scenario: Core execution definitions need no base library
- **WHEN** NexuML is installed without the optional base library or local API
- **THEN** core execution definitions remain discoverable and Local remains usable

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
The system SHALL keep framework component-definition/runtime infrastructure, including the universal direct-module adapter, in the `nexuml` core package and reusable concrete semantic component definitions in the separate same-repository base library package.

#### Scenario: Core package is installed alone
- **WHEN** only the core package is installed
- **THEN** NexuML framework commands, component base/registry/serialization infrastructure, and `nn_module(...)` SHALL remain available without requiring the base library package.

#### Scenario: Base library package is installed
- **WHEN** the same-repository base library package is installed
- **THEN** its decorated layer definitions, data source definitions, scenario recipes, evaluation definitions, and registered loader definitions SHALL be discoverable through the standard registry loading path.

#### Scenario: Base component is used in Python
- **WHEN** a user imports a concrete component such as `LMBE` from the base library
- **THEN** that public symbol SHALL be the typed authoring definition
- **AND** any mutable runtime implementation required by the definition SHOULD remain an implementation detail of the component module.

### Requirement: Direct PyTorch modules do not require library discovery
An importable PyTorch module factory used through `nn_module(...)` SHALL NOT need a module-specific registration decorator, component registry entry, entry point, or configured local library root.

#### Scenario: Installed framework module is used directly
- **WHEN** Python authoring passes an installed importable factory such as `torch.nn.Dropout` to `nn_module(...)`
- **THEN** only the universal core `NnModule` definition SHALL be registered
- **AND** normal library scanning SHALL NOT register or enumerate `torch.nn.Dropout` as a NexuML component.

#### Scenario: Direct-module YAML is restored in a fresh process
- **WHEN** resolved YAML contains the built-in `NnModule` identity
- **THEN** the core package SHALL make that universal definition available before component restoration
- **AND** restoration SHALL NOT depend on `nexuml_library` package scanning.

#### Scenario: Wrapped custom module is exported
- **WHEN** an importable custom `torch.nn.Module` is materialized inside `TorchModuleAdapter` and a self-contained package is exported
- **THEN** package source discovery SHALL inspect wrapped child modules as well as outer pipeline-layer classes
- **AND** custom source SHALL be included under the existing package policy while runtime dependencies such as PyTorch remain external.
