## Purpose

Define the library warmup cosine scheduler: importable by dotted path, linear warmup then cosine decay, literal minimum learning rate, explicit parameters, and test coverage.

## Requirements

### Requirement: Library warmup cosine scheduler is importable directly
The library SHALL provide a navigable `WarmupCosineLR` scheduler at `nexuml_library.training.schedulers.WarmupCosineLR` that can be configured through NexuML's typed `scheduler(...)` helper.

#### Scenario: Scenario uses library scheduler symbol
- **WHEN** a scenario defines `scheduler(WarmupCosineLR, warmup_epochs=2, max_epochs=100, min_lr=1e-6)`
- **THEN** NexuML SHALL preserve the real factory symbol and typed constructor arguments while resolving and instantiating the scheduler without any scheduler decorator or registry.

### Requirement: Scheduler performs linear warmup then cosine decay
The scheduler SHALL increase learning rates linearly during the configured warmup period and then apply cosine decay for the remaining configured epoch horizon.

#### Scenario: Learning rate warms up before cosine decay
- **WHEN** `warmup_epochs` is greater than zero and training steps through epochs before the warmup boundary
- **THEN** the learning rate SHALL increase linearly from the initial warmup value toward the optimizer base learning rate.

#### Scenario: Learning rate decays after warmup
- **WHEN** the scheduler steps through epochs after the warmup boundary and before `max_epochs`
- **THEN** the learning rate SHALL follow a cosine decay curve from the optimizer base learning rate toward `min_lr`.

### Requirement: Minimum learning rate is literal
The scheduler SHALL treat `min_lr` as an absolute learning-rate floor for each optimizer parameter group, not as an additive multiplier or multiplicative factor.

#### Scenario: Final decay approaches min_lr
- **WHEN** training reaches the end of the configured scheduler horizon
- **THEN** each parameter group's learning rate SHALL approach `min_lr` rather than `base_lr * min_lr`.

### Requirement: Scheduler parameters are explicit
The scheduler SHALL require its timing and floor parameters as direct `scheduler(...)` constructor arguments, including `max_epochs`, without requiring NexuML core to infer them from `TrainingSpec`.

#### Scenario: Core package remains unchanged
- **WHEN** the warmup cosine scheduler is added to the library
- **THEN** NexuML core scheduler resolution, `TrainingSpec`, and compiler behavior SHALL remain unchanged.

### Requirement: Scheduler behavior is covered by tests
The library SHALL include tests that verify scheduler importability, constructor behavior, and representative learning-rate values across warmup and cosine phases.

#### Scenario: Tests validate key epochs
- **WHEN** the scheduler is tested with known `base_lr`, `warmup_epochs`, `max_epochs`, and `min_lr`
- **THEN** tests SHALL assert expected learning-rate behavior at initial, warmup-boundary, decay, and final epochs.
