## ADDED Requirements

### Requirement: Playback is an ordinary inference-only workflow

NexuML SHALL provide a simulator-independent public CLI play command and callable Python entrypoint using existing complete policy loading, explicit compatible environment selection, inference rollout primitives and worker-local lifecycle. It SHALL support continuous playback until stopped and explicit bounded playback, seeded deterministic or stochastic inference, and fixed actor weights without constructing an optimizer. Formal evaluation and native benchmark scoring SHALL remain separate; play results SHALL NOT be silently treated as benchmark or training data. Unsupported benchmark-only definitions SHALL require a playable environment.

#### Scenario: A trusted policy is played in a compatible native task
- **WHEN** the user starts play from an exported policy or supported trusted checkpoint
- **THEN** the existing inference path runs the native task without changing weights or creating a learner, and exits cleanly on requested stop, cancellation or configured bound.

### Requirement: Playback preserves native parallel environment semantics

Playback SHALL support single and qualified native vectorized environments, explicit simulated count independent of visible subset, per-instance termination/truncation and correct native autoreset or masked reset. Recurrent/chunk/processor/history state and episode counters SHALL reset only for affected instances; terminal observations SHALL not be replaced by reset observations. Native batch/device semantics SHALL be retained without double-vectorizing. Optional real-time pacing/speed SHALL throttle wall-clock execution relative to the native control period without changing physics dt or skipping required control steps.

#### Scenario: One of sixteen instances terminates
- **WHEN** one parallel environment reaches its native termination
- **THEN** its own episode and policy state reset, other rows continue, the displayed subset does not change the simulation count, and pacing does not alter physical timing.

### Requirement: Interactive playback controls are explicitly scoped to simulation

Qualified native playback SHALL support viewer navigation and, where provider/viewer permits, pause/resume, single-step, selected/all resets, speed, diagnostics and explicitly declared bounded task commands. Simulation mutations SHALL be authorized at the actual dispatch boundary and applied only during an explicit simulation play purpose; unsupported/forged commands SHALL fail closed. Train and evaluate spectators SHALL retain nonmutation guarantees. Simulation play SHALL NOT implicitly actuate physical ROS 2 hardware or enable arbitrary executable commands, unsafe exploration or unapproved deployment.

#### Scenario: A playback user changes a walking target
- **WHEN** an approved native task exposes that bounded command during simulation play
- **THEN** the qualified command is applied at a valid step boundary and recorded as an intervention; the same request is denied during training, formal evaluation or an unapproved physical runtime.

### Requirement: Playback recording and placement reuse existing services

Playback SHALL support bounded existing media recording where qualified, with source policy, environment and actual frame/time provenance. Local and independent remote Ray Jobs execution SHALL use existing execution dispatch, explicit worker resources and owned cleanup; the remote client SHALL receive connection and job details without starting a local simulator or a new service. A continuous playback recording SHALL have a finite capture budget. The API SHALL be reusable by future Studio integration and other qualified simulator providers.

#### Scenario: Remote playback is stopped
- **WHEN** an independent Ray playback job is cancelled or reaches its bound
- **THEN** owned simulator, policy, viewer and recording resources are released, configured outputs are finalized and the borrowed cluster remains running.
