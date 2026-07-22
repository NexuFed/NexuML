## MODIFIED Requirements

### Requirement: NexuFL loads NexuML package artifacts
NexuFL SHALL load NexuML `pipeline.package` artifacts through the stable `nexuml_export/artifact.pkl` entry before falling back to legacy entries.

#### Scenario: NexuFL loads primary artifact entry
- **WHEN** NexuFL receives an export directory containing `pipeline.package`
- **THEN** it SHALL attempt to load `nexuml_export/artifact.pkl`
- **AND** if the payload is a dict containing `pipeline`, it SHALL use that pipeline as the model.

#### Scenario: NexuFL trains loaded pipeline
- **WHEN** NexuFL loads a NexuML pipeline package
- **AND** the package runtime dependencies are installed
- **THEN** the TrainerV1 adapter SHALL initialize the model
- **AND** run one train batch
- **AND** compute a scalar loss
- **AND** complete an optimizer step.

#### Scenario: Missing runtime dependency reports manifest
- **WHEN** package loading fails because an external runtime dependency is missing
- **THEN** the error message SHALL point to the generated `requirements.txt` or `metadata.external_dependencies`
- **AND** it SHALL NOT instruct the user to install the full NexuML workspace source tree.
