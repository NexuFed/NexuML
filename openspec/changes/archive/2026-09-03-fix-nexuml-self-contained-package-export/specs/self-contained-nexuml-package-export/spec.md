## ADDED Requirements

### Requirement: Package exports include pipeline source code
The system SHALL export `pipeline.package` so that Python source modules needed to unpickle and train the exported pipeline are included in the package.

#### Scenario: Built-in CIFAR ResNet package loads without NexuML workspace source
- **WHEN** `nexuml export cifar-resnet --checkpoint <ckpt> -o <dir>` creates `<dir>/pipeline.package`
- **AND** a clean Python process has runtime dependencies installed but no `PYTHONPATH` pointing at the NexuML source workspace
- **THEN** `torch.package.PackageImporter("<dir>/pipeline.package").load_pickle("nexuml_export", "artifact.pkl")` SHALL succeed
- **AND** the payload SHALL contain `pipeline`, `resolved_config`, `metadata`, and `training_state`
- **AND** `payload["pipeline"]` SHALL be a trainable `torch.nn.Module`.

#### Scenario: NexuML library code is packaged
- **WHEN** the exported pipeline contains layers from `nexuml_library`
- **THEN** the package SHALL include the required `nexuml_library` Python source modules
- **AND** loading SHALL NOT require `nexuml_library` to be installed from the source workspace.

#### Scenario: Custom layer code is packaged
- **WHEN** the exported pipeline contains a custom layer defined outside `nexuml` and `nexuml_library`
- **THEN** the package SHALL include the source module defining that layer
- **AND** loading SHALL succeed without the custom source directory on `PYTHONPATH`.

### Requirement: Runtime dependencies remain external
The system SHALL keep heavyweight runtime dependencies external to the package.

#### Scenario: Torch and TensorDict are external
- **WHEN** a package is exported
- **THEN** dependencies such as `torch`, `torchvision`, `tensordict`, `pydantic`, `lightning`, `torchmetrics`, `numpy`, and `pandas` SHALL NOT be vendored into the package
- **AND** the clean load environment SHALL provide those dependencies as external runtime modules.

### Requirement: Stable artifact entry exists
The system SHALL expose `nexuml_export/artifact.pkl` as the stable package entry.

#### Scenario: Primary package entry loads payload
- **WHEN** a consumer loads `nexuml_export/artifact.pkl`
- **THEN** the returned object SHALL be a dict
- **AND** the dict SHALL contain at least `pipeline`, `resolved_config`, `metadata`, and `training_state`.

#### Scenario: Legacy package entry remains available
- **WHEN** a consumer loads `model/pipeline.pkl`
- **THEN** the package SHOULD return the packaged pipeline object for backward compatibility.

### Requirement: Exported pipeline supports continued training
The system SHALL export a pipeline object that supports downstream continued training.

#### Scenario: One optimizer step succeeds
- **WHEN** NexuFL loads the exported pipeline with `PackageImporter`
- **AND** builds a dummy CIFAR TensorDict batch
- **THEN** the pipeline SHALL run forward
- **AND** the downstream adapter SHALL be able to compute a scalar loss
- **AND** an optimizer step SHALL complete without requiring the NexuML source workspace.
