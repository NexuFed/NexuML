# Spec Delta

## Purpose

Evaluate and record complete Isaac-trained policies through NexuML's existing inference-only workflow while respecting process-global simulator ownership and preserving portable inference artifacts.

## ADDED Requirements

### Requirement: Replay runs in a fresh simulator process without a learner

The existing policy-evaluation command SHALL support all four Isaac references in fresh processes using one evaluation environment, the declared profile, embedded contract/configuration and ordinary overrides. Replay SHALL construct no training session, algorithm, optimizer or collector and SHALL NOT require an external native trainer checkpoint.

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

Replay SHALL validate observation/action/profile compatibility, preserve policy modes and relevant RNG streams, and distinguish termination from truncation and auto-reset observations. Task outcomes SHALL remain separate from episode returns. SO-101 reaching SHALL report joint error/success; vial placement SHALL report physical placement success and available manipulation outcomes under its own task rule.

#### Scenario: A seeded SO-101 policy is evaluated
- **WHEN** complete episodes finish in a fresh simulator
- **THEN** return/length/end statistics and separate reaching-error/success results identify the evaluated seeds and task configuration

#### Scenario: An environment override changes control meaning
- **WHEN** the requested task/asset/action mapping differs incompatibly from the packaged policy
- **THEN** replay rejects it rather than treating matching tensor dimensions alone as compatibility

### Requirement: Recording reuses native media and logging conventions

Replay SHALL support bounded GIF, MP4, or both from the same first evaluated episode through existing native artifact/logging behavior. TensorBoard previews SHALL be animated using only its configured writer; MLflow-only SHALL not access one. Media SHALL NOT label reset frames as the prior episode's terminal state. MP4 dependencies SHALL remain optional for scalar/GIF workflows.

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

### Requirement: Tutorial replay uses its declared state policy and renderer

Vial-placement replay SHALL run in its isolated Newton profile from canonical home starts, using the exported state policy and nominal tutorial evaluation settings. Requested recording SHALL use a verified headless camera through the pinned Newton renderer without adding actor camera inputs or starting Isaac Sim. Unsupported rendering SHALL remain an explicit incomplete gate, not trigger a backend switch.

#### Scenario: A vial policy is evaluated with both media formats
- **WHEN** the user selects the documented Newton profile, renderer and bounded recording
- **THEN** one evaluated rollout supplies GIF, MP4, animated TensorBoard preview when configured and native artifacts with correct terminal-scene identity
- **AND** logs identify home-start rows/seeds, task/profile and renderer without claiming learned success from return alone

#### Scenario: A scalar or GIF-only tutorial replay omits MP4 extras
- **WHEN** only the selected output's prerequisites are installed
- **THEN** evaluation and requested output succeed without importing an MP4 encoder

#### Scenario: Recorded Newton replay is cancelled
- **WHEN** cancellation occurs before encoding or delivery finishes
- **THEN** the run preserves cancellation status, finalizes native loggers, releases owned runtime/media resources and publishes no incomplete recording as a successful artifact
