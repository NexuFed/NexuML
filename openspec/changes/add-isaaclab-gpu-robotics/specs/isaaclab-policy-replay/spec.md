# Spec Delta

## Purpose

Evaluate and record complete Isaac-trained policies through NexuML's existing inference-only workflow while respecting process-global simulator ownership and preserving portable inference artifacts.

## ADDED Requirements

### Requirement: Replay runs in a fresh simulator process without a learner

The existing policy-evaluation command SHALL support the three Isaac references in a fresh process using one evaluation environment, the embedded contract/configuration, and ordinary overrides. Replay SHALL construct no training session, algorithm, optimizer, or collector and SHALL NOT require an external native trainer checkpoint.

#### Scenario: A supported Isaac policy is replayed
- **WHEN** the user invokes the evaluation command with approved simulator prerequisites and a positive episode count
- **THEN** a fresh owned simulator evaluates the loaded complete policy and returns episode metrics without learner construction

#### Scenario: The process already owns an incompatible simulator context
- **WHEN** replay would allocate a second conflicting context
- **THEN** it fails clearly before disturbing the active simulator and directs the user to fresh-process replay

### Requirement: Training rejects unsupported in-fit Isaac evaluation

Isaac training references SHALL explicitly disable automatic episode evaluation initially. Requests for automatic in-fit/start/end evaluation or recording that require a competing simulator context SHALL fail before simulator allocation, with fresh-process replay guidance. Existing non-Isaac scheduled evaluation SHALL remain unchanged.

#### Scenario: Periodic recording is requested during Isaac training
- **WHEN** an override enables automatic evaluation/media for a supported Isaac training reference
- **THEN** training exits with a focused unsupported-lifecycle diagnostic before starting the simulator
- **AND** it neither silently ignores the setting nor resets the training runtime

#### Scenario: The reference uses its documented defaults
- **WHEN** training finishes with automatic evaluation disabled
- **THEN** training artifacts remain available and the documented separate replay command performs evaluation

### Requirement: Policy inference remains independent of simulation and learning

Each complete policy package SHALL load and execute on matching synthetic observations with Isaac, TorchRL, simulator assets, and recording imports blocked. Packages SHALL retain declared action semantics and provenance while excluding live simulator/collector/learner state. Training checkpoints SHALL remain distinct from normal inference artifacts.

#### Scenario: A policy is loaded outside the Isaac environment
- **WHEN** a clean process blocks simulation, learning, and encoder imports
- **THEN** Cartpole, Franka, and SO-101 policy inference matches pre-export deterministic outputs on the same inputs

#### Scenario: A user requests simulation replay without Isaac
- **WHEN** the portable policy loads but simulator prerequisites are unavailable
- **THEN** inference remains usable and replay reports the missing execution prerequisite without starting training

### Requirement: Replay preserves evaluation and task semantics

Replay SHALL validate observation/action compatibility, preserve policy modes and relevant RNG streams, and distinguish termination from truncation and auto-reset observations. Optional task outcomes SHALL remain separate from episode returns. SO-101 replay SHALL report joint-reaching error and success under the declared task rule.

#### Scenario: A seeded SO-101 policy is evaluated
- **WHEN** complete episodes finish in a fresh simulator
- **THEN** return/length/end statistics and separate reaching-error/success results identify the evaluated seeds and task configuration

#### Scenario: An environment override changes control meaning
- **WHEN** the requested task/asset/action mapping differs incompatibly from the packaged policy
- **THEN** replay rejects it rather than treating matching tensor dimensions alone as compatibility

### Requirement: Recording reuses native media and logging conventions

Replay SHALL support bounded GIF, MP4, or both from the same first evaluated episode through existing native artifact/logging behavior. TensorBoard previews SHALL use only its configured writer; MLflow-only SHALL not access one. Media SHALL not include reset observations/frames falsely labeled as the prior episode's terminal state.

#### Scenario: Both media formats are selected
- **WHEN** recording prerequisites and the selected rendering path are available
- **THEN** one bounded captured rollout supplies finalized GIF/MP4 and provenance JSON through configured native loggers without rerolling the policy

#### Scenario: A rendered episode ends with native auto-reset
- **WHEN** the captured step is terminal or truncated
- **THEN** any terminal frame belongs to the ending episode rather than the reset scene
- **AND** unavailable rendering is diagnosed clearly rather than replacing the renderer silently

### Requirement: Provenance and cleanup retain native behavior

Replay SHALL finalize existing native loggers and owned simulator/media resources on success, error, or cancellation without hiding primary failures. Results SHALL record source package, task/asset/backend/profile identity, seeds, timing, capture bounds, and explicit source-frame-or-zero semantics under existing logs-root conventions.

#### Scenario: An MP4-only MLflow replay completes
- **WHEN** only MLflow is configured and MP4 is selected
- **THEN** scalar/task results, MP4, and JSON belong to its native evaluation run without an extra media run or downloadable GIF

#### Scenario: Delivery or simulator shutdown fails
- **WHEN** encoding/upload/cleanup raises during replay
- **THEN** failure remains visible, owned resources are closed, and already delivered artifacts are not claimed to have transactional rollback
