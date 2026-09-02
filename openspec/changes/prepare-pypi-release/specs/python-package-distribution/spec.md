## Purpose

Defines the public packaging, licensing, dependency, versioning, and installation contract for the independently installable NexuML distributions.

## ADDED Requirements

### Requirement: Distributions use Apache License 2.0
The `nexuml` and `nexuml-library` distributions SHALL declare the SPDX license expression `Apache-2.0` and SHALL include the complete Apache License 2.0 text in their source and binary distribution artifacts.

#### Scenario: User inspects package licensing
- **WHEN** a user inspects either distribution's PyPI metadata or downloaded wheel or source distribution
- **THEN** the distribution identifies its license as Apache-2.0
- **AND** the distribution artifact contains the Apache License 2.0 text.

### Requirement: Initial public distributions use version 0.2.0
The initial PyPI-ready `nexuml` and `nexuml-library` distributions produced by this change SHALL both use version `0.2.0`, and the version reported by the installed core package SHALL match its distribution metadata.

#### Scenario: Installed version is inspected
- **WHEN** version `0.2.0` of the core wheel is installed
- **THEN** package metadata reports `0.2.0`
- **AND** `nexuml.__version__` reports `0.2.0`.

#### Scenario: Base library metadata is inspected
- **WHEN** the base-library wheel is built for this release
- **THEN** its distribution metadata reports version `0.2.0`.

### Requirement: Core installation does not require the base library
The mandatory dependencies of `nexuml` SHALL NOT include `nexuml-library`, and the installed core framework and CLI SHALL remain usable when no base-library distribution is installed.

#### Scenario: User installs only the core distribution
- **WHEN** a user installs `nexuml` without requesting optional extras in a clean environment
- **THEN** the installer does not install `nexuml-library`
- **AND** importing `nexuml` succeeds
- **AND** the NexuML CLI can display its help and perform framework operations that do not require library-provided components.

### Requirement: Base library is an optional one-way extension
The `nexuml` distribution SHALL expose a `library` extra that installs `nexuml-library` without an exact distribution-version pin. The `nexuml-library` distribution SHALL depend on a compatible minimum `nexuml` version without creating a dependency from core back to the library.

#### Scenario: User requests the base library through core
- **WHEN** a user installs `nexuml[library]`
- **THEN** the environment contains both distributions
- **AND** library-provided components and scenarios are discoverable through the standard installed-library entry point.

#### Scenario: Dependency metadata is resolved
- **WHEN** an installer resolves either distribution
- **THEN** the dependency graph contains no `nexuml -> nexuml-library -> nexuml` mandatory cycle
- **AND** the core package does not require one exact `nexuml-library` release.

### Requirement: Published dependencies express runtime compatibility
Each distribution SHALL declare only dependencies needed by its base runtime or an advertised optional feature. Published dependency metadata SHALL use compatibility constraints rather than development lock versions, except where a documented third-party compatibility requirement necessitates a bounded range.

#### Scenario: Core metadata is inspected
- **WHEN** a user inspects the core wheel's `Requires-Dist` entries
- **THEN** unused packages and dependencies used only by the base library are absent from mandatory core requirements
- **AND** optional feature dependencies are associated with their corresponding extras.

#### Scenario: Development environment is resolved
- **WHEN** a contributor creates the repository development environment
- **THEN** exact transitive versions may be resolved by the repository lockfile
- **AND** those lock versions do not become exact pins in published runtime metadata unless explicitly required for compatibility.

### Requirement: Both distributions provide complete standard artifacts
Each release SHALL produce an installable wheel and source distribution for both `nexuml` and `nexuml-library`, including package source, declared package data, README metadata, and license files without relying on the repository checkout at installation time.

#### Scenario: Distribution contents are validated
- **WHEN** the wheel and source distribution for either package are inspected
- **THEN** all files needed for its documented runtime behavior are present
- **AND** metadata validation succeeds
- **AND** a wheel can be built from the source distribution.

### Requirement: PyPI metadata identifies the project
Both distributions SHALL publish valid descriptions, supported Python metadata, project URLs, maintainership information, development status, and relevant Python classifiers.

#### Scenario: User views either PyPI project page
- **WHEN** PyPI renders the uploaded distribution metadata
- **THEN** the project description and README render successfully
- **AND** users can navigate to the source repository, documentation, issue tracker, and release history.
