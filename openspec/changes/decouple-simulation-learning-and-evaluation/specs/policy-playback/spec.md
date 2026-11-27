## ADDED Requirements

### Requirement: Playback is an ordinary inference-only workflow

The simulator-independent public play CLI and Python entrypoint SHALL reuse complete policy loading, inference rollouts and worker-local lifecycle in an explicitly selected compatible environment. Play SHALL support continuous or bounded runs and seeded deterministic or stochastic inference with fixed weights and no optimizer. Play SHALL remain separate from formal evaluation and native scoring; its results SHALL NOT silently become benchmark or training data.

#### Scenario: A trusted policy is played in a compatible native task
- **WHEN** the user starts play from an exported policy or supported trusted checkpoint
- **THEN** the existing inference path runs the native task without changing weights or creating a learner, and exits cleanly on requested stop, cancellation or configured bound.

#### Scenario: The selected definition supports only benchmarking
- **WHEN** the user requests play with that definition
- **THEN** the request fails and asks for a playable environment rather than substituting benchmark execution.

### Requirement: Playback preserves native parallel environment semantics

Playback SHALL support single and qualified native vectorized environments, preserving native batches and devices without double-vectorizing. The simulated count SHALL be explicit and independent of the visible subset. Termination and truncation SHALL use native autoreset or masked reset per instance. Optional pacing SHALL throttle wall-clock playback relative to the native control period without changing physics dt or skipping required control steps.

#### Scenario: One of sixteen instances terminates
- **WHEN** one parallel environment terminates or is truncated
- **THEN** only its recurrent, chunk, processor and history state and episode counters reset. Its terminal observation stays distinct from its reset observation; the other rows continue.

### Requirement: Interactive playback controls are explicitly scoped to simulation

Qualified playback SHALL support viewer navigation and provider/viewer-supported simulation controls. Only explicit simulation play SHALL permit mutations authorized at command dispatch. Unsupported or forged commands SHALL fail closed; training and evaluation spectators SHALL remain nonmutating. Simulation play SHALL NOT implicitly actuate physical ROS 2 hardware or enable arbitrary executable commands, unsafe exploration or unapproved deployment.

#### Scenario: A user selects a native playback control
- **WHEN** the provider and viewer support pause/resume, single-step, selected or all resets, speed, diagnostics or an explicitly declared bounded task command
- **THEN** the user can apply that control during simulation play.

#### Scenario: A playback user changes a walking target
- **WHEN** an approved native task exposes that bounded command during simulation play
- **THEN** the qualified command is applied at a valid step boundary and recorded as an intervention; the same request is denied during training, formal evaluation or an unapproved physical runtime.

### Requirement: Playback recording and placement reuse existing services

Local playback and independent remote Ray Jobs SHALL use existing dispatch, explicit worker resources and cleanup of owned resources. The remote client SHALL receive connection and job details without starting a local simulator or new service. Playback SHALL reuse qualified bounded media recording, identifying the source policy, environment and actual frames and times captured. Studio and other qualified providers SHALL be able to reuse the API.

#### Scenario: A continuous play session is recorded
- **WHEN** the user requests recording for a continuous rollout
- **THEN** recording stops at its finite capture budget even if playback continues.

#### Scenario: Remote playback is stopped
- **WHEN** an independent Ray playback job is cancelled or reaches its bound
- **THEN** owned simulator, policy, viewer and recording resources are released, configured outputs are finalized and the borrowed cluster remains running.
