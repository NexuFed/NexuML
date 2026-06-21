# library-management-cli Specification

## Purpose
TBD - created by archiving change split-base-library-discovery. Update Purpose after archive.
## Requirements
### Requirement: Add local library root
The CLI SHALL provide a command to add a local library root folder for future discovery.

#### Scenario: User adds a library root
- **WHEN** the user runs `nexuml library add ./library`
- **THEN** the CLI stores the normalized library root path for future NexuML runs

#### Scenario: User adds a missing library root
- **WHEN** the user runs `nexuml library add ./missing` and the path does not exist
- **THEN** the CLI fails with a clear error and does not store the path

### Requirement: Remove local library root

The CLI SHALL provide a command to remove a previously configured local library root folder from future discovery.

#### Scenario: User removes a configured library root

- **WHEN** the user runs `nexuml library delete ./library` for a root that was previously added
- **THEN** the CLI removes the normalized library root path from the stored configuration

#### Scenario: User removes an unconfigured library root

- **WHEN** the user runs `nexuml library delete ./other-library` and the normalized path is not configured
- **THEN** the CLI fails with a clear error and leaves the stored configuration unchanged

### Requirement: List configured library roots

The CLI SHALL provide a command to list available library sources, including configured local library root folders and the base `nexuml_library` package when it is importable.

#### Scenario: User lists libraries

- **WHEN** the user runs `nexuml library list`
- **THEN** the CLI prints the configured local library roots

#### Scenario: Base library package is available

- **WHEN** the user runs `nexuml library list` and `nexuml_library` is importable
- **THEN** the CLI prints `nexuml_library` as an available library source

#### Scenario: No libraries are available

- **WHEN** the user runs `nexuml library list`, no local roots are configured, and `nexuml_library` is not importable
- **THEN** the CLI reports that no libraries are available or configured without failing

### Requirement: List registered layers
The CLI SHALL provide a command to list layers discovered from installed library packages and configured local library roots.

#### Scenario: User lists layers
- **WHEN** the user runs `nexuml registry list layers`
- **THEN** the CLI prints discovered layer keys and their implementation modules

### Requirement: List registered data sources
The CLI SHALL provide a command to list data sources discovered from installed library packages and configured local library roots.

#### Scenario: User lists data sources
- **WHEN** the user runs `nexuml registry list data`
- **THEN** the CLI prints discovered data source keys and their implementation modules

### Requirement: List registered scenarios
The CLI SHALL provide a command to list scenarios discovered from installed library packages and configured local library roots.

#### Scenario: User lists scenarios
- **WHEN** the user runs `nexuml registry list scenarios`
- **THEN** the CLI prints discovered scenario keys and their implementation modules

### Requirement: List registered evaluation algorithms
The CLI SHALL provide a command to list evaluation algorithms discovered from installed library packages and configured local library roots.

#### Scenario: User lists evaluation algorithms
- **WHEN** the user runs `nexuml registry list eval`
- **THEN** the CLI prints discovered evaluation algorithm keys and their implementation modules

### Requirement: Registry list commands reflect current library files
The CLI SHALL show registry contents based on the current files in installed and configured libraries at command execution time.

#### Scenario: User edits a registered local library before listing
- **WHEN** the user edits a decorated object in a configured local library root and then runs a registry list command
- **THEN** the CLI output reflects the edited library state

### Requirement: Library add is tested end-to-end against discovery
The test suite SHALL verify that adding a local library root through the CLI affects subsequent discovery and registry output.

#### Scenario: Added library root contributes registry keys
- **WHEN** a test runs `nexuml library add <temporary-library-root>` for a root containing decorated registry elements
- **THEN** a subsequent registry list or discovery call SHALL include those elements without requiring element-specific tests

#### Scenario: Added library root feeds conformance tests
- **WHEN** a temporary library root is added through the same configuration path used by the CLI
- **THEN** registry conformance collection SHALL include parameter cases for elements from that root

#### Scenario: Missing library root is not persisted
- **WHEN** a test runs `nexuml library add <missing-path>`
- **THEN** the command SHALL fail and the persisted library configuration SHALL remain unchanged

### Requirement: Library delete is tested end-to-end against discovery
The test suite SHALL verify that deleting a local library root through the CLI removes it from subsequent discovery and registry output.

#### Scenario: Deleted library root no longer contributes registry keys
- **WHEN** a test adds a temporary library root, verifies its registry keys are visible, and then runs `nexuml library delete <temporary-library-root>`
- **THEN** a subsequent registry list or discovery call SHALL no longer include keys from that root

#### Scenario: Deleting unconfigured root preserves configuration
- **WHEN** a test runs `nexuml library delete <unconfigured-root>`
- **THEN** the command SHALL fail clearly and the persisted library configuration SHALL remain unchanged

### Requirement: Library CLI tests are isolated from user configuration
Library-management tests SHALL use isolated temporary configuration locations and SHALL NOT read or mutate a developer's real NexuML configuration.

#### Scenario: Test configuration isolation
- **WHEN** library CLI tests run
- **THEN** configured library roots SHALL be stored in a temporary test location that is removed after the test

