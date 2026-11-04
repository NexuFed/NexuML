# Spec Delta

## Purpose

Provide reproducible native MuJoCo and Meta-World policy workflows through NexuML's existing training, evaluation and portable inference interfaces, without confusing integration evidence with learned task quality or hardware readiness.

## ADDED Requirements

### Requirement: All three references use the ordinary workflow

The existing `mujoco-humanoid-ppo` and `mujoco-reacher-ppo` and new `metaworld-reach-ppo` SHALL support discovery, configuration round trips, ordinary overrides, training, evaluation, interrupted resume and complete-policy export/load without user source edits or an alternate trainer.

#### Scenario: A reference is discovered without simulator dependencies
- **WHEN** any of the three scenarios is discovered, resolved or compiled with simulation imports blocked
- **THEN** its explicit contract and configuration remain usable without allocating simulation or rendering resources

#### Scenario: A native training smoke completes
- **WHEN** each reference runs with approved prerequisites and a small finite frame budget
- **THEN** real rollouts, changed actor parameters, finite losses, evaluation and normal result construction are verified
- **AND** no stochastic convergence threshold is required by fast CI

### Requirement: Declared controls agree with the native task

Each reference SHALL validate its declared observation/action structure, dtypes, bounds and control timing against its pinned native environment. Action ordering, physical meaning, upstream task/settings and model provenance SHALL be recorded. Equal tensor dimensions SHALL NOT imply compatible controls or a named hardware embodiment.

#### Scenario: Runtime settings disagree with the compiled contract
- **WHEN** native fields, action bounds, task identity or control timing disagree with the selected policy contract
- **THEN** execution fails before the first policy command rather than rewriting the contract or silently clipping/reinterpreting actions

#### Scenario: Meta-World reaching is selected
- **WHEN** `metaworld-reach-ppo` describes its maintained `reach-v3` task
- **THEN** its four continuous actions identify normalized Cartesian displacement and gripper semantics, including the upstream scale/frame and policy period
- **AND** generic Humanoid/Reacher are not described as G1, SO-101 or other hardware models

### Requirement: Native transitions and lifecycle remain correct

The references SHALL preserve seeded resets, separate termination/truncation, finite actual-frame budgets, owned-resource cleanup and supported CPU direct/process collection. Native final observations SHALL be retained for completed transitions; reset observations SHALL NOT be substituted for them.

#### Scenario: An episode reaches its native time limit
- **WHEN** collection encounters a timeout rather than a true task termination
- **THEN** the transition remains truncated with its correct final observation and value-bootstrap semantics
- **AND** the next episode starts through the supported reset path

#### Scenario: Training fails or is cancelled
- **WHEN** allocation, rollout, optimization or cancellation interrupts a reference
- **THEN** owned environments, renderers and collector workers are closed without hiding the primary failure

### Requirement: Resume and inference are distinct proven paths

All three references SHALL restore learner state and remaining frame budget from an interrupted training checkpoint while rebuilding live environments. Their complete inference artifacts SHALL exclude critics, optimizers, rollout buffers and live handles, and execute deterministically without simulator, learner or encoder imports.

#### Scenario: Interrupted training resumes
- **WHEN** a checkpoint taken before the finite budget completes is restored
- **THEN** actor, critic, optimizer and progress counters are restored and only remaining collection advances
- **AND** exact continuation of interrupted physics trajectories is not claimed

#### Scenario: Exported inference runs in a clean process
- **WHEN** matching observations are passed to a reloaded policy with simulator, learning and recording imports blocked
- **THEN** deterministic actions match the pre-export reference
- **AND** simulation replay without simulator prerequisites reports a focused prerequisite error instead of constructing a learner

### Requirement: Headless recording reuses native artifacts

All three references SHALL support bounded GIF, MP4 and both-format replay through existing logging/media behavior. NVIDIA EGL and CPU OSMesa SHALL be validated explicitly; state-only training SHALL NOT require a renderer. Recording failures SHALL NOT silently switch backends or publish incomplete media as successful.

#### Scenario: Both formats are requested
- **WHEN** a single-environment replay requests GIF and MP4 with valid rendering/encoding prerequisites
- **THEN** one first-episode rollout supplies both finalized formats, native logger delivery and provenance
- **AND** recording limits do not shorten the remaining scalar evaluation episodes

#### Scenario: Graphics are absent but CUDA is visible
- **WHEN** the selected NVIDIA renderer lacks working EGL/vendor support despite compute visibility
- **THEN** recording fails with actionable graphics diagnostics while renderer-free training remains a separate capability

### Requirement: Meta-World results identify their evaluation protocol

Meta-World evidence SHALL report success separately from return using the pinned task's native success rule and explicit goal/reset/horizon settings. Multi-seed comparisons SHALL use matched frozen evaluation conditions and identify sample counts and uncertainty. Custom held-out variations SHALL be labeled separately from official benchmark protocols.

#### Scenario: Success occurs before the last step
- **WHEN** the native success signal becomes true at any step in an evaluation episode
- **THEN** that episode counts once as successful even if its final-step success is false
- **AND** the next episode does not inherit that success

#### Scenario: A small benchmark comparison is published
- **WHEN** three PPO training seeds and a seeded random reference are evaluated on the same frozen goal/reset set
- **THEN** results retain per-seed/per-condition success and returns, trial counts, variability and runtime
- **AND** a short optimizer smoke or small comparison is not claimed as benchmark mastery, a complete Phase 4 gate or hardware readiness

### Requirement: Completion is backed by scoped evidence

The change SHALL publish exact native, round-trip, headless-media, dependency-composition and regression evidence with versions, source/configuration identities and failures. Simulator/GPU tests SHALL remain opt-in; skipped tests and prepared CI configuration SHALL NOT count as executed acceptance.

#### Scenario: The change is declared implemented
- **WHEN** completion is reported
- **THEN** all three reference round trips, both headless backend checks, benchmark comparison and shared-environment acceptance have actual evidence
- **AND** unrelated baseline failures, unrun remote CI and deferred roadmap phases remain explicit
