# Spec Delta

## Purpose

Make GPU-native Isaac rollouts usable by the existing NexuML reinforcement training path without double-vectorization, incorrect episode boundaries, or a separate learning engine.

## ADDED Requirements

### Requirement: Collection reuses the existing learning lifecycle

Isaac training SHALL use the existing session, Lightning ownership, PPO update path, direct collector, synchronization, and checkpoint conventions. Provider-specific transition handling SHALL remain optional and SHALL NOT replace the framework-neutral runtime contract or change existing supervised/Gymnasium behavior.

#### Scenario: A supported Isaac PPO run executes
- **WHEN** the user selects a reference scenario through the ordinary CLI/session path
- **THEN** collection and real parameter updates occur under the existing trainer rather than an external native training runner

#### Scenario: An existing non-Isaac scenario executes
- **WHEN** the new optional bridge is installed but a supervised or Gymnasium scenario is selected
- **THEN** its configuration, collector path, lifecycle, and results retain their established behavior

### Requirement: Device and batch semantics are preserved

Training SHALL retain observations, actions, and rollout storage on the selected single GPU, preserve environment/time axes through advantage estimation, and count actual environment transitions. Reference frame batches SHALL contain an integer number of native batch steps. Unsupported frame geometry SHALL be diagnosed before allocation.

#### Scenario: A multi-environment batch is collected
- **WHEN** N environments execute T policy steps in a rollout
- **THEN** the frame counter advances by N times T, per-field shapes exclude N, and temporal advantage estimation does not mix episodes between environments
- **AND** no action is converted to CPU/NumPy in the simulation hot path

#### Scenario: The requested batch is not divisible by the native width
- **WHEN** the frame batch size is not divisible by N
- **THEN** the run identifies the incompatible width before simulator allocation rather than reporting fabricated frame progress

### Requirement: Same-step reset preserves both sides of the episode boundary

Auto-reset handling SHALL distinguish each completed transition's final observation from the next episode's initial observation. Native reset SHALL happen once per completed environment, leave other environments undisturbed, and preserve transition buffers against later in-place mutation.

#### Scenario: One environment terminates while another continues
- **WHEN** only one row in a native batch ends
- **THEN** its transition retains the prior episode's final observation and its next action uses the reset observation
- **AND** the continuing row is neither reset nor overwritten

#### Scenario: The provider reuses mutable buffers
- **WHEN** subsequent native steps mutate reward/done/observation buffers
- **THEN** earlier collected transitions retain their original values and episode identity

### Requirement: Time-limit bootstrapping uses the correct final state

PPO SHALL bootstrap time-limit truncations from the real final observation and suppress terminal bootstrapping according to existing algorithm semantics. Missing final observations SHALL produce a clear unsupported-configuration error, not reset-state values, NaN replacement, or treating all truncations as terminations.

#### Scenario: Terminal and reset observations have distinct values
- **WHEN** a controlled truncated transition has known final and reset states
- **THEN** the value target matches the final-state calculation and differs from the reset-state calculation

#### Scenario: The selected bridge cannot supply final observations
- **WHEN** a supported task reaches a timeout without the required final-state data
- **THEN** execution fails visibly instead of returning apparently successful but invalid PPO metrics

### Requirement: Actions retain declared task and embodiment meaning

Continuous actions SHALL use an explicitly declared finite envelope and matching sampling/log-probability semantics. Task scaling, physical units, actuator/joint order, offsets, limits, gripper behavior, and control period SHALL be documented and traceable. Undeclared post-sampling action clipping or categorical substitution SHALL NOT change controls silently.

#### Scenario: Isaac Cartpole is selected
- **WHEN** its policy emits an action
- **THEN** the action is a continuous force-control input with the declared scale, not a categorical Gymnasium CartPole command

#### Scenario: Joint action metadata is incompatible
- **WHEN** an override changes ordering, scale, limits, or timing incompatibly with the packaged contract
- **THEN** validation rejects the mismatch before applying actions

### Requirement: Resume restores learning state without claiming simulator continuation

Checkpoint resume SHALL restore policy, critic, optimizer state, update counters, and remaining frame budget while building a fresh simulator. It SHALL NOT serialize live physics resources or claim bitwise continuation of the interrupted physical trajectory. Policy synchronization SHALL precede the next rollout when due.

#### Scenario: Training resumes from an intermediate checkpoint
- **WHEN** a fresh process restores a partially completed supported run
- **THEN** learning state is restored, the new runtime is reset, only the remaining frame budget is collected, and synchronized policy weights are used

#### Scenario: The frame budget is already complete
- **WHEN** a completed-budget checkpoint is restored
- **THEN** no new rollout or training simulator is allocated and replay remains a separate operation

### Requirement: TorchRL integration is audited against pinned runtime behavior

Both profiles SHALL have an evidence-linked audit against the user-linked TorchRL IsaacLab guide and their actual installed sources. The audit SHALL cover launch order, process ownership, batch/device/spec mapping, mutable buffers, reset/final states, numerical GAE, action/log-probability consistency, on-policy synchronization and cleanup. Each item SHALL identify pass, justified non-applicability or an unresolved blocker.

#### Scenario: A guide example differs from the implemented bridge
- **WHEN** wrapper choice, startup order or another recommendation differs
- **THEN** the report records the pinned source behavior and focused tests supporting the choice, or leaves the gate unresolved
- **AND** the guide's recurrent, multi-GPU and asynchronous examples do not expand this feed-forward single-GPU scope

#### Scenario: The tutorial profile is evaluated for integration correctness
- **WHEN** native tests exercise its real batch and the selected bridge
- **THEN** no-done, partial/all-done, termination, truncation, simultaneous flags, masked resets and buffer-reuse cases preserve final/reset identity and expected GAE values
- **AND** the original EA passes are not substituted for new-profile evidence

### Requirement: Bridge selection preserves numerical correctness over wrapper preference

The maintained wrapper SHALL be selected only when supported public composition preserves all required transition semantics in the selected profile. Otherwise the existing generic adapter over canonical provider normalization SHALL remain the supported path with recorded probe evidence. NaN replacement, double resets, private patches and a separate collector or learner SHALL NOT satisfy acceptance.

#### Scenario: A wrapper avoids double reset but omits terminal states
- **WHEN** its native-autoreset mode returns NaN or unavailable final observations
- **THEN** the wrapper probe fails the timeout-value gate despite apparently finite training metrics
- **AND** the proven canonical fallback remains valid rather than being replaced solely to match documentation

### Requirement: The tutorial reuses the existing state-policy learner contract

Vial-placement training SHALL use the tutorial's declared policy-state observation for the existing actor and critic and retain its six continuous control meanings. Any difference from the upstream privileged-critic training configuration SHALL be explicit. Additional privileged or camera inputs SHALL NOT enter exported actors implicitly, and the tutorial's learner SHALL NOT replace NexuML PPO.

#### Scenario: The tutorial exposes extra critic-only contact observations
- **WHEN** its state task is configured for NexuML
- **THEN** the actor and critic use the documented policy-state fields, unused critic-only groups are disabled and the symmetric-critic deviation is recorded
- **AND** native control/transition comparisons use matching task settings without promising identical learner curves
