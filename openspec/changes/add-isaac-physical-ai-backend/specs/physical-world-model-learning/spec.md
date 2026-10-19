# Spec Delta

## Purpose

Train learned dynamics and DreamerV3 policies from Isaac and authorized recorded experience, measuring predictive and control utility rather than treating world-model support as a documentation feature.

## ADDED Requirements

### Requirement: DreamerV3 is an executable required workflow

The backend SHALL train a DreamerV3 reference through NexuML with actual world-model, actor and value updates, checkpoint resume, complete inference export and fresh Isaac evaluation. The selected algorithm protocol and deviations SHALL be explicit; earlier Dreamer versions SHALL NOT be relabeled DreamerV3.

#### Scenario: The world-model workstream is assessed for completion
- **WHEN** only a generic predictor or older Dreamer loss has run
- **THEN** DreamerV3 acceptance remains incomplete even if that model produces finite predictions

### Requirement: Replay preserves causal sequence boundaries

World-model training SHALL sample valid contiguous trajectories with explicit first-step, termination, truncation, padding and stream identity. Warm-up or burn-in, posterior-state reuse and partial resets SHALL NOT leak latent context across episodes or workers.

#### Scenario: A replay sequence contains an episode boundary
- **WHEN** latent filtering and prediction losses execute
- **THEN** state resets at the boundary and padded targets do not contribute to the loss

### Requirement: Model and imagination objectives are distinguishable

The selected DreamerV3 recipe SHALL record its latent dynamics, reconstruction, reward, continuation, representation/dynamics regularization, actor/value objectives and target-update semantics. Physical collection, replay samples and imagined transitions SHALL have separate counters and budgets.

#### Scenario: Policy updates use imagined trajectories
- **WHEN** training metrics are published
- **THEN** imagined steps are not reported as newly collected simulator frames and each loss is attributed to the recorded algorithm recipe

### Requirement: Learning state is completely resumable

World-model resume SHALL preserve all required trainable/target modules, optimizers, normalizers, update ratios, randomness and declared replay state. Restarted environments SHALL begin fresh streams; omitted replay SHALL require a documented refill policy without false continuation claims.

#### Scenario: A partial-budget Dreamer run is resumed
- **WHEN** a fresh process restores the checkpoint
- **THEN** restored learning state and remaining physical-frame budget are verified while incomplete old simulator streams are closed explicitly

### Requirement: Predictive quality is evaluated beyond training loss

World-model evidence SHALL include held-out one-step and multi-step errors, reward/continuation accuracy and rollout-horizon degradation under declared distribution shifts. Synthetic observations and generated trajectories SHALL retain origin labels and SHALL NOT be mixed into real-data evaluation silently.

#### Scenario: Predictions drift over longer imagined horizons
- **WHEN** horizon-wise validation is reported
- **THEN** the report exposes compounding error and its control implications rather than presenting reconstruction quality as physical fidelity

### Requirement: World models demonstrate comparative control utility

A DreamerV3 policy SHALL be evaluated against the established PPO or other conventional reference on matched task, observations and physical-data budget. Reports SHALL separate sample efficiency, compute cost, task quality and uncertainty; improvement SHALL NOT be presumed or enforced as a flaky smoke-test threshold.

#### Scenario: Dreamer trains but underperforms the reference
- **WHEN** the benchmark finishes
- **THEN** the actual result and limitations are retained without hiding failures or claiming imagination automatically improves robotics control

### Requirement: Latent inference exports all required state estimation

World-model policy inference SHALL package the observation encoder, posterior/state estimator, actor and required state initialization or normalization. Training-only imagination losses, decoders and critics SHALL be excluded unless the selected deployed computation genuinely requires them.

#### Scenario: A latent policy is loaded without a learner
- **WHEN** a fixed observation/action/reset sequence is replayed
- **THEN** its latent updates and decoded actions match reference inference with simulation and training components blocked

### Requirement: Planning and learned dynamics remain bounded tools

Learned-dynamics planning SHALL use explicit horizons, model identities and controller contracts with an independently evaluated reference. Model uncertainty SHALL NOT replace physical safety controls, and unsupported pretrained world-model adapters SHALL remain unadvertised. A dynamics-only example SHALL NOT establish hierarchical control support.

#### Scenario: A learned planner is used for a simulated skill selector
- **WHEN** the selector invokes a low-level policy
- **THEN** task success and model-error failures are measured separately and no action bypasses the low-level controller's declared limits

### Requirement: Hierarchical control has its own executable proof

A separate hierarchical reference SHALL execute a high-level selector over at least two compatible low-level skills with explicit transition, reset and controller contracts. Selector and skill identities SHALL be recorded. A single dynamics model or low-level actor SHALL NOT substitute for the selector-plus-controller workflow.

#### Scenario: A composed task requires a skill transition
- **WHEN** the high-level selector changes from one low-level skill to another
- **THEN** the actual selected skills, transition conditions and task outcome are retained
- **AND** invalid skill targets are rejected, episode reset clears pending state and all commands retain the low-level controller's limits
