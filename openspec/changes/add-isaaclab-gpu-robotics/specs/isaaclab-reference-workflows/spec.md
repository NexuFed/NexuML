# Spec Delta

## Purpose

Provide reproducible Cartpole and Franka Isaac reference workflows that demonstrate existing NexuML training, checkpoints, export, evaluation, and experiment conventions before the later SO-101 reference.

## ADDED Requirements

### Requirement: Native baselines precede NexuML integration claims

The integration SHALL demonstrate native Cartpole and Franka reaching in the pinned environment before claiming their NexuML adapters work. Evidence SHALL record exact task/backend/configuration, versions, seed, environment count, controls, timing, resource limits, commands, and results. Native baseline runners SHALL NOT become NexuML's training engine.

#### Scenario: Native Cartpole and Franka are validated
- **WHEN** their approved native baseline commands execute
- **THEN** each produces bounded rollout and training evidence, with resources shut down and matching input/control semantics recorded

#### Scenario: Native execution is unavailable
- **WHEN** a required prerequisite prevents a native baseline
- **THEN** the gate remains explicitly incomplete rather than being satisfied by static compilation or fake tests

### Requirement: Reference scenarios use ordinary NexuML entry points

Discoverable Cartpole and Franka state-policy scenarios SHALL support ordinary Python factories, registry selection, YAML resolution/restoration, repeatable CLI overrides, training, checkpoints, and complete-policy export. They SHALL NOT require user source edits, a second parser, or a simulator-specific trainer/session.

#### Scenario: Users run either reference from resolved YAML
- **WHEN** environment width, finite frame budget, seed, and ordinary algorithm settings are overridden
- **THEN** the resolved configuration is shown and restored through the normal path and a supported GPU run executes

#### Scenario: Invalid execution settings are supplied
- **WHEN** an override violates the supported device, topology, or action contract
- **THEN** the normal diagnostic path exits nonzero without launching simulation

### Requirement: Both reference round trips require actual learning and replay

Each reference SHALL demonstrate collection, finite PPO metrics, changed actor parameters, interrupted checkpoint resume, complete-policy export/reload, and seeded fresh-process evaluation. Recorded metrics SHALL identify actual frames and evaluated episodes. No stochastic reward/convergence threshold SHALL gate routine tests.

#### Scenario: Cartpole completes its acceptance run
- **WHEN** the bounded native and NexuML tests are executed
- **THEN** real updates, remaining-budget resume, portable inference equality, fresh simulator evaluation, and shutdown are verified

#### Scenario: Franka completes its acceptance run
- **WHEN** the bounded reaching tests are executed
- **THEN** actual articulated-joint control and the same training/resume/export/replay round trip are verified without claiming manipulation mastery

### Requirement: Results preserve task and experiment provenance

Native NexuML logging/artifacts SHALL trace task/source revision, backend, robot/asset identity, action mapping, control rate, environment width, seeds, resolved configuration, and dependency profile. Comparisons SHALL distinguish native execution parity from different learners' training performance.

#### Scenario: Native and NexuML runs are compared
- **WHEN** a completion report describes the two paths
- **THEN** control/transition parity is checked under matching settings and any reward/configuration differences are explicit
- **AND** native PPO scores are not presented as NexuML training results

### Requirement: Tests and instructions distinguish capabilities from prerequisites

Fast CPU tests SHALL verify metadata, configuration, failure behavior, and optional-import isolation without downloading or launching Isaac. Real simulator/GPU acceptance SHALL require explicit opt-in. Documentation SHALL provide exact installation, native baseline, train, checkpoint, resume, export, replay, and artifact-inspection commands with supported limitations.

#### Scenario: Default CI executes
- **WHEN** ordinary CPU checks run without Isaac installed
- **THEN** no simulator/system install or asset download occurs and heavy tests skip with explicit prerequisites

#### Scenario: Phase 2 completion is reported
- **WHEN** the change is marked complete
- **THEN** Cartpole, Franka, and the required later SO-101 capability have actual execution evidence
- **AND** missing GPU/system proofs, existing baseline failures, and remote CI execution status are stated separately
