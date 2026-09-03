## ADDED Requirements

### Requirement: Lightning checkpoint sidecar is preserved
The system SHALL preserve or generate a `lightning.ckpt` sidecar when Lightning checkpoint state is available.

#### Scenario: Export from checkpoint preserves checkpoint
- **WHEN** `nexuml export <scenario> --checkpoint <path> -o <dir>` is run
- **THEN** `<dir>/lightning.ckpt` SHALL exist
- **AND** it SHALL preserve the original Lightning checkpoint data or a normalized equivalent sufficient for NexuML-side resume/fine-tuning.

#### Scenario: Existing sidecars remain
- **WHEN** a package is exported
- **THEN** `state_dict.pt`, `resolved_config.yaml`, and `metadata.json` SHALL still be written.

### Requirement: Checkpoint metadata is normalized
The system SHALL expose checkpoint/training provenance as plain metadata.

#### Scenario: Checkpoint epoch and step are available
- **WHEN** the source checkpoint contains `epoch` or `global_step`
- **THEN** `metadata["checkpoint"]["epoch"]` and `metadata["checkpoint"]["global_step"]` SHALL be populated where present
- **AND** the same data SHALL be available through `artifact.pkl`.

#### Scenario: Validation metrics are available
- **WHEN** validation metrics or callback metrics are present in the checkpoint/trainer state
- **THEN** `metadata["checkpoint"]["validation_metrics"]` SHALL include those metrics where they can be safely serialized.

#### Scenario: Best checkpoint data is available
- **WHEN** ModelCheckpoint state contains best model score/path/monitor/mode data
- **THEN** `metadata["checkpoint"]` SHALL include the best model score, best model path, monitor, and mode where present.

#### Scenario: Hyperparameters are available
- **WHEN** Lightning `hyper_parameters` are present in the checkpoint
- **THEN** JSON-safe hyperparameters SHALL be included in `metadata["checkpoint"]["hyper_parameters"]`.

### Requirement: Training state is available for downstream adapters
The system SHALL preserve machine training state where available.

#### Scenario: Optimizer and scheduler state is available
- **WHEN** optimizer or scheduler states are present in the trainer or checkpoint
- **THEN** `payload["training_state"]` SHALL include those states
- **AND** `training_state.pt` SHALL be written as a sidecar.

#### Scenario: NexuML-specific checkpoint state is preserved
- **WHEN** NexuML eval results or post-train layer state are present in a Lightning checkpoint
- **THEN** the export SHALL preserve them in the Lightning sidecar
- **AND** SHALL expose JSON-safe summaries in metadata where practical.
