# dataset-root-resolution Specification

## Purpose
TBD - created by archiving change normalize-dataset-roots-downloads. Update Purpose after archive.
## Requirements
### Requirement: Consistent dataset root resolution
The system SHALL resolve dataset roots consistently for library data builders: absolute paths are used unchanged, relative paths are resolved under `NEXUML_DATA_ROOT` when that environment variable is set, and otherwise remain relative to the current working directory.

#### Scenario: Absolute path is preserved
- **WHEN** a library data builder receives `/mnt/local/DCASET2` as its dataset root
- **THEN** the resulting dataset specifications use `/mnt/local/DCASET2` without prepending `NEXUML_DATA_ROOT`

#### Scenario: Relative path uses global data root
- **WHEN** `NEXUML_DATA_ROOT=/mnt/local` and a library data builder uses logical root `DCASET2`
- **THEN** the resulting dataset specifications use `/mnt/local/DCASET2`

#### Scenario: Relative path without global data root
- **WHEN** `NEXUML_DATA_ROOT` is unset and a library data builder uses logical root `cifar10`
- **THEN** the resulting dataset specifications use `cifar10` relative to the current working directory

### Requirement: Dataset builders own reusable dataset defaults
The system SHALL keep reusable dataset layout defaults, catalog defaults, and root defaults in library data builders rather than duplicating them in composed scenario functions.

#### Scenario: Composed DCASE scenario uses data-builder defaults
- **WHEN** a user resolves `dcase-ast-adacos` without explicit dataset root or machine catalog overrides
- **THEN** the scenario obtains its DCASE root, machine defaults, section defaults, and split defaults from the DCASE T2 data builder

#### Scenario: Composed AudioSet scenario uses data-builder defaults
- **WHEN** a user resolves an AudioSet composed scenario without an explicit dataset root
- **THEN** the scenario obtains the AudioSet root and split layout from the AudioSet data builder

### Requirement: DCASE Task 2 dataset naming and root
The system SHALL use `DCASET2Dataset` as the DCASE Task 2 dataset key and SHALL default DCASE Task 2 roots to logical path `DCASET2`.

#### Scenario: DCASE T2 resolves under global root
- **WHEN** `NEXUML_DATA_ROOT=/mnt/local` and the DCASE T2 data builder uses its default root
- **THEN** it produces dataset specs with `type_key: DCASET2Dataset` and root `/mnt/local/DCASET2`

### Requirement: Clear missing dataset diagnostics
The system SHALL report missing datasets with the resolved path, expected layout, and available recovery action such as setting `NEXUML_DATA_ROOT` or enabling a supported download.

#### Scenario: Missing dataset without download
- **WHEN** a dataset root does not exist and download is disabled
- **THEN** the user receives an error that includes the resolved path and guidance for providing or downloading the data

