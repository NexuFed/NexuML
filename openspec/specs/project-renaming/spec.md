## Purpose

Define the canonical project renaming: core package `nexuml`, base library `nexuml_library`, extension entry point namespace `nexuml.libraries`, and user-facing `NexuML` references.

## Requirements

### Requirement: Canonical core project name
The system SHALL use `nexuml` as the canonical core package import name, CLI command name, console script name, and distribution metadata name.

#### Scenario: User imports core package
- **WHEN** user code imports the core framework package after installation
- **THEN** the import path is `nexuml`

#### Scenario: User runs CLI command
- **WHEN** the user invokes the framework CLI after installation
- **THEN** the executable command is `nexuml`

#### Scenario: Package metadata is inspected
- **WHEN** Python package metadata or project configuration is inspected
- **THEN** core project identifiers use `nexuml` instead of `nexuml`

### Requirement: Canonical base library name
The system SHALL use `nexuml_library` as the canonical base library package import name and distribution metadata name.

#### Scenario: User imports base library package
- **WHEN** user code imports the same-repository base library after installation
- **THEN** the import path is `nexuml_library`

#### Scenario: Base library metadata is inspected
- **WHEN** Python package metadata or base library project configuration is inspected
- **THEN** base library identifiers use `nexuml_library` instead of `nexuml_library`

### Requirement: Canonical extension entry point namespace
The system SHALL use `nexuml.libraries` as the Python entry point group for installed library packages.

#### Scenario: Installed library advertises reusable content
- **WHEN** a package exposes a `nexuml.libraries` entry point and is installed in the active environment
- **THEN** NexuML scans the entry point target for decorated objects during registry loading

### Requirement: User-facing project references use NexuML
The repository SHALL use `NexuML` for user-facing project name references and `nexuml` for command/package references in active documentation, examples, configs, tests, and specs.

#### Scenario: User reads active project documentation
- **WHEN** a user reads active project documentation or examples
- **THEN** commands, imports, package names, and project references use `nexuml`, `nexuml_library`, and `NexuML`

#### Scenario: Tests assert user-facing output
- **WHEN** tests assert CLI output, errors, config paths, or discovery names
- **THEN** expected values use the renamed project and library identifiers