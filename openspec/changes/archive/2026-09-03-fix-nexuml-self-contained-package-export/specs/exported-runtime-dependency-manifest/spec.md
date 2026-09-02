## ADDED Requirements

### Requirement: Runtime dependency manifest is generated
The system SHALL emit a generated `requirements.txt` for external runtime dependencies referenced by the package.

#### Scenario: requirements file contains only used external dependencies
- **WHEN** `pipeline.package` is exported
- **THEN** the export directory SHALL contain `requirements.txt`
- **AND** it SHALL list only externalized distributions actually referenced by the packaged payload
- **AND** it SHALL NOT list `nexuml` or `nexuml_library` as requirements for the clean load path.

#### Scenario: Versions are pinned when resolvable
- **WHEN** an external dependency distribution version can be resolved at export time
- **THEN** `requirements.txt` SHALL include an exact pinned specifier such as `torch==<version>`.

#### Scenario: Unresolved versions are reported
- **WHEN** an external dependency module cannot be mapped to a distribution version
- **THEN** the dependency SHALL still be represented in structured metadata
- **AND** the unresolved reason SHALL be recorded.

### Requirement: Structured external dependency metadata is written
The system SHALL write structured runtime dependency metadata in `metadata.json` and in `artifact.pkl`.

#### Scenario: Metadata lists external dependencies
- **WHEN** a package is exported
- **THEN** `metadata["external_dependencies"]` SHALL list external dependencies as objects containing module, distribution, version, specifier, and reason where available.

#### Scenario: Standard-library modules are excluded from install manifest
- **WHEN** the exporter observes standard-library modules such as `sys`, `json`, `pathlib`, `typing`, or `datetime`
- **THEN** those modules SHALL NOT appear in `requirements.txt`
- **AND** they MAY be omitted from structured external dependency metadata.
