# Spec Delta

## Purpose

Close the robot-learning loop using maintained human corrections, offline-to-online reinforcement learning and learned reward signals while keeping physical success, privacy, deployment authorization and low-level robot safety independent from learning.

## ADDED Requirements

### Requirement: Correction episodes support DAgger-style training

LeRobot-compatible intervention and rollout data SHALL preserve episode/clock/source identity, pre- and post-intervention actions, policy checkpoint, operator labels and separate collection/training/held-out splits. A bounded SO-101 simulation reference SHALL collect mistakes, train a correction-aware student and evaluate against no-correction/equal-budget baselines without creating a replacement teleoperation UI.

#### Scenario: A user takes over a rollout
- **WHEN** the selected capture workflow records operator intervention
- **THEN** the original policy command, executed control, intervention interval and provenance remain distinguishable, and evaluation episodes are not silently added to training

### Requirement: Offline-to-online RL has one truthful replay contract

An RLPD-style reference SHALL combine validated offline demonstration replay and collected simulation experience with explicit mixture scheduling, importance of action units, replay sampling, policy staleness, target updates, warm-up, optimizer counters and resumes. It SHALL build on existing SAC/offline data seams, not implement a second opaque RL runner.

#### Scenario: An offline-pretrained agent begins online training
- **WHEN** a partially trained agent resumes from demonstrations
- **THEN** the run records actual offline and new interaction samples, replay ratios, fresh policy updates, returned actor artifacts and separately frozen comparison results

### Requirement: HIL-SERL is simulation-first and explicitly safety gated

A human-intervention RL reference SHALL integrate bounded corrective actions, replay and a trainable task reward classifier. Its first implementation SHALL run in simulation or approved recordings; real-hardware online RL, controller replacement, unattended exploration and automatic actuator writes remain outside ordinary CI and require their own physical safety approval.

#### Scenario: A policy attempts a disallowed action
- **WHEN** the simulation intervention wrapper constrains or replaces the command
- **THEN** executed action, intervention and consequence are recorded; no claim is made that learned bounds or classifiers certify real hardware safety

### Requirement: Learned rewards are optional training signals, never task truth

Support one maintained HIL-SERL reward-classifier training round trip and assess or integrate published Robometer/TOPReward processors against approved data and compute. Reward signals SHALL carry calibration, uncertainty, model/version/processor identity and held-out reward-quality tests. Reward training SHALL NOT overwrite measured native success, task constraints or release gates.

#### Scenario: The learned reward is high but the vial remains held
- **WHEN** task evaluation detects no seated/released/held success
- **THEN** learned reward remains a separate diagnostic and task success is false

### Requirement: Flow-based policies have complete temporal actions

A maintained small flow-matching or Diffusion Transformer action policy SHALL support masked future-action training, conditioning and iterative sampler, normalization and stochastic RNG identity, chunk execution/replanning, resume, package export and clean-process deterministic/seeded parity through existing NexuML data and policy surfaces. It SHALL use matched SO-101 task/data/protocol comparisons with ACT and Diffusion rather than claiming VLA architecture equivalence.

#### Scenario: Inference is resumed after export
- **WHEN** the same observation/history/reset schedule and sampling seed are used
- **THEN** normalized and physical action chunks agree within recorded tolerances with no hidden training-only state

### Requirement: Temporal inference supports deadlines and safe state transitions

An opt-in policy-serving interface SHALL support asynchronous inference requests, explicit observation timestamps, per-stream state/chunks, bounded buffers, timeouts/stale-observation rejection and held-action/fallback delegation to the controller. Offline replay and ROS 2 reference inference SHALL reproduce action timing and reset behavior. A remote policy process SHALL not be required for ordinary local inference.

#### Scenario: A VLA chunk arrives too late
- **WHEN** the request misses its configured control deadline
- **THEN** the runtime records the missed deadline, discards incompatible stale chunks and delegates bounded fallback to the supported external controller rather than applying obsolete commands

### Requirement: Adaptation has rigorous counterfactual and forgetting controls

All corrective, offline-to-online and learned-reward comparisons SHALL freeze task versions and held-out episodes, record multi-seed/condition sample counts, measure task success, intervention frequency and old-task degradation, and return pass/fail/insufficient-evidence independently of safety authorization.

#### Scenario: Reward optimization increases returns but reduces task success
- **WHEN** the new policy is compared on a frozen suite
- **THEN** failure modes and lower success block unsupported claims of policy improvement
