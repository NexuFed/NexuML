## Purpose

Define the logs root resolution system: prefixing relative experiment log and artifact paths with `NEXUML_LOGS_ROOT`, preserving absolute paths, applying to default outputs, and handling MLflow file URIs.

## Requirements

### Requirement: Relative log paths use NEXUML_LOGS_ROOT
The system SHALL prefix relative experiment log and artifact paths with `NEXUML_LOGS_ROOT` when the environment variable is set.

#### Scenario: Experiments path is prefixed exactly
- **WHEN** `NEXUML_LOGS_ROOT` is `/mnt/logs` and a configured log path is `.experiments/tensorboard`
- **THEN** the resolved path is `/mnt/logs/.experiments/tensorboard`

#### Scenario: Arbitrary relative path is prefixed
- **WHEN** `NEXUML_LOGS_ROOT` is `/mnt/logs` and a configured log path is `runs/tensorboard`
- **THEN** the resolved path is `/mnt/logs/runs/tensorboard`

#### Scenario: Environment variable is unset
- **WHEN** `NEXUML_LOGS_ROOT` is unset and a configured log path is `.experiments/tensorboard`
- **THEN** the resolved path remains `.experiments/tensorboard`

### Requirement: Absolute log paths are preserved
The system SHALL preserve absolute experiment log and artifact paths when `NEXUML_LOGS_ROOT` is set.

#### Scenario: Absolute path with logs root
- **WHEN** `NEXUML_LOGS_ROOT` is `/mnt/logs` and a configured log path is `/var/tmp/nexuml/tensorboard`
- **THEN** the resolved path remains `/var/tmp/nexuml/tensorboard`

### Requirement: Default experiment outputs use logs root resolution
The system SHALL apply logs root resolution to default experiment outputs including trainer root, TensorBoard, DVCLive, file-based MLflow storage, diagrams, Optuna storage, preprocessing output, and relative temporary artifact directories.

#### Scenario: Default outputs are relocated together
- **WHEN** `NEXUML_LOGS_ROOT` is `/mnt/logs` and default experiment output settings are used
- **THEN** trainer, logger, diagram, tuning, preprocessing, and relative artifact output paths resolve under `/mnt/logs/.experiments` or the corresponding prefixed relative path

### Requirement: MLflow file URIs use logs root resolution
The system SHALL apply logs root resolution to relative file-based MLflow tracking URIs and SHALL leave non-file tracking URIs unchanged.

#### Scenario: Relative file URI is prefixed
- **WHEN** `NEXUML_LOGS_ROOT` is `/mnt/logs` and the MLflow tracking URI is `file:./.experiments/mlflow`
- **THEN** the resolved MLflow tracking URI points to `/mnt/logs/.experiments/mlflow`

#### Scenario: Remote tracking URI is unchanged
- **WHEN** `NEXUML_LOGS_ROOT` is `/mnt/logs` and the MLflow tracking URI is `https://mlflow.example.com`
- **THEN** the resolved MLflow tracking URI remains `https://mlflow.example.com`