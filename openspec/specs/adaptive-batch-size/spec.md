## Purpose

Define the adaptive batch size system: structured automatic batch-size configuration, loader precedence, CUDA probe resolution, safety policies, and observable probe results.

## Requirements

### Requirement: Structured automatic training batch size
The system SHALL allow `training.batch_size` to be configured either as an explicit positive integer or as a structured automatic batch-size configuration.

#### Scenario: Explicit integer remains valid
- **WHEN** a scenario config sets `training.batch_size` to `64`
- **THEN** the system uses `64` as the training batch-size default without running automatic probing

#### Scenario: Structured automatic config is accepted
- **WHEN** a scenario config sets `training.batch_size` to an automatic batch-size object with mode `auto`, minimum, maximum, candidate strategy, and safety policy
- **THEN** the system validates the configuration and defers selecting the effective integer batch size until runtime

### Requirement: Loader batch size precedence remains explicit
The system SHALL treat `training.batch_size` as the scenario-level default and `data.loader.batch_size` as an optional dataloader override.

#### Scenario: Loader batch size is unset
- **WHEN** `data.loader.batch_size` is unset and `training.batch_size` resolves to an integer
- **THEN** the dataloader uses the resolved training batch size

#### Scenario: Loader batch size is explicitly set
- **WHEN** `data.loader.batch_size` is set to an explicit integer
- **THEN** the dataloader uses `data.loader.batch_size` rather than `training.batch_size`

### Requirement: CUDA probe resolves automatic batch size
The system SHALL resolve automatic batch-size configuration by trying real CUDA forward/backward probe passes for candidate batch sizes before training starts.

#### Scenario: Candidate succeeds
- **WHEN** a candidate batch size completes the probe forward/backward pass without CUDA out-of-memory
- **THEN** the candidate is recorded as successful and remains eligible for selection

#### Scenario: Candidate runs out of memory
- **WHEN** a candidate batch size raises a CUDA out-of-memory error during probing
- **THEN** the system records the candidate as failed, clears recoverable CUDA memory state, and continues resolution without treating the whole run as failed if smaller candidates remain

#### Scenario: No candidate succeeds
- **WHEN** every candidate batch size fails during probing
- **THEN** the system fails before training with a diagnostic that includes the automatic batch-size bounds and failed candidates

### Requirement: Safety policy selects effective batch size
The system SHALL select the effective integer batch size from successful probe candidates according to the configured safety policy.

#### Scenario: Largest policy
- **WHEN** the safety policy is `largest` and candidates `1`, `2`, `4`, and `8` succeed
- **THEN** the selected batch size is `8`

#### Scenario: Previous power-of-two policy
- **WHEN** the safety policy is `previous_power_of_two` and candidates `1`, `2`, `4`, and `8` succeed
- **THEN** the selected batch size is `4`

#### Scenario: Margin policy
- **WHEN** the safety policy is `margin`, the margin is `0.9`, candidates `1`, `2`, and `4` complete probing, and candidate `4` exceeds 90% peak CUDA memory usage
- **THEN** the selected batch size is `2`, the largest successful candidate that stays within the configured memory margin

### Requirement: Probe result is observable
The system SHALL include automatic batch-size resolution metadata in runtime logs or run artifacts.

#### Scenario: Automatic batch size is selected
- **WHEN** automatic probing selects an effective batch size
- **THEN** the run metadata includes the selected batch size, candidate attempts, safety policy, and relevant CUDA device information