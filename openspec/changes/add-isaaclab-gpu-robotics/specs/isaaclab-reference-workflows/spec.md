# Spec Delta

## Purpose

Provide reproducible native and NexuML Isaac reference workflows, including development-container execution, with profile-specific training, checkpoint, export, evaluation and completion evidence.

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
- **THEN** Cartpole, Franka, SO-101 joint reaching and the upstream SO-101 vial-placement reference have actual execution evidence in their declared profiles
- **AND** missing GPU/system proofs, existing baseline failures, and remote CI execution status are stated separately

### Requirement: Development-container workflows are executable without a worker image

Documentation SHALL provide exact isolated setup, editable-install, interpreter/debug selection, native probe, train, checkpoint/resume, export, scalar/recorded replay and artifact-inspection commands inside the existing NVIDIA DevContainer. Commands SHALL identify the selected profile and work without user source edits, fixed worker mount paths or an additional running container.

#### Scenario: A developer follows the documented workflow
- **WHEN** the approved container prerequisites and isolated profile are prepared
- **THEN** the commands execute from the documented checkout with the selected interpreter and produce native NexuML checkpoints, exports, metrics and media
- **AND** repeat setup verifies or reuses matching sources without overwriting unrelated environments, staged work or reset data

### Requirement: Tutorial-native validation precedes NexuML manipulation claims

The pinned upstream vial task SHALL pass bounded native task-discovery, asset/reset-data, contact/control, outcome and cleanup checks before its NexuML reference is claimed supported. Native settings and any NexuML observation/learner differences SHALL be recorded. Upstream scripts or policies SHALL remain comparison evidence, not NexuML training or learned-success evidence.

#### Scenario: Upstream vial placement runs but NexuML has not trained it
- **WHEN** an upstream learner or scripted reference completes a placement
- **THEN** the report identifies that evidence separately and keeps NexuML updates, resume, export and replay gates open

### Requirement: Expanded completion preserves historical evidence without borrowing it

The original 42 completed tasks SHALL retain their evidence and completion state. New DevContainer installation, both-profile audit and vial-placement gates SHALL remain unchecked until verified. Final evidence SHALL identify source/environment manifests and commands per profile, distinguish controlled task success from learned policy success, and preserve unrelated roadmap gates.

#### Scenario: The original integration is complete but the extension is not
- **WHEN** only the previously validated worker-image tests have passed
- **THEN** the baseline remains recorded as complete while the expanded change is in progress
- **AND** no DevContainer-native, Newton or manipulation correctness claim is inferred from those original passes
