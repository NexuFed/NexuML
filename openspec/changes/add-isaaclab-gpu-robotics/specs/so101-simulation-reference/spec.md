# Spec Delta

## Purpose

Include SO-101 as a required later Phase 2 simulation reference, using the maintained robot asset and a minimal reward-bearing joint-reaching task rather than claiming an existing demonstration task is PPO-ready.

## ADDED Requirements

### Requirement: SO-101 is required after the initial references

Phase 2 SHALL include an executable SO-101 reference after Cartpole and Franka validation. SO-101 SHALL NOT be satisfied by roadmap text, asset discovery, fake tests, substituting another arm, or making the whole reference optional to change completion.

#### Scenario: Cartpole and Franka pass but SO-101 has not executed
- **WHEN** Phase 2 completion is assessed
- **THEN** SO-101 remains an open acceptance gate and the change is not reported complete

#### Scenario: SO-101 is exercised after the shared bridge is verified
- **WHEN** the approved asset and runtime prerequisites are available
- **THEN** the SO-101 task executes through the same supported GPU provider and NexuML workflow, not a separate training stack

### Requirement: The maintained SO-101 asset retains explicit provenance

The reference SHALL reuse the pinned upstream SO-101 asset with its Isaac Sim PhysX variant. It SHALL record asset source/version or content identity, license/access prerequisites, joint names/order, authored limits, actuator configuration, fixed gripper position, and units. Missing or incompatible assets SHALL fail clearly without automatic model replacement.

#### Scenario: An approved SO-101 asset is loaded
- **WHEN** native task construction completes
- **THEN** the expected five arm joints and gripper are validated against the actual articulation and the selected physics variant is recorded

#### Scenario: The asset identity or articulation differs
- **WHEN** a replacement asset changes joint ordering, limits, or actuator semantics
- **THEN** validation rejects the mismatch before policy actions are applied

### Requirement: The first task is five-joint reaching with a fixed gripper

The task SHALL command the five arm joints toward seeded, validated reachable joint targets while holding the gripper at a documented fixed position. Observations SHALL expose arm positions, velocities, and targets with declared units/order. Continuous action bounds and target decoding SHALL be explicit. The task SHALL NOT imply arbitrary six-degree-of-freedom pose control or hardware calibration.

#### Scenario: Native joint response is validated
- **WHEN** bounded scripted controls are applied to approved targets
- **THEN** intended joints move with the declared sign/scale, targets respect articulation limits, and the gripper remains fixed

#### Scenario: An incompatible action is supplied
- **WHEN** input has an incorrect width/order, nonfinite values, or unsupported gripper/pose semantics
- **THEN** the task rejects it rather than silently dropping or remapping controls

### Requirement: Reaching has a real reward and observable outcome

The task SHALL define a finite reward from joint-target error, explicit reset/target sampling, termination and time-limit rules, and a documented success tolerance. It SHALL report reaching error and task success separately from episode return. The upstream reward-free cube-stack configuration SHALL NOT be presented as a trained manipulation reference.

#### Scenario: Target tracking and timeout are measured
- **WHEN** controlled native episodes reach a target or exhaust the horizon
- **THEN** reward/error/success and termination/truncation agree with the configured task rules and remain traceable to the same target/seed

#### Scenario: A native demonstration task has no reward
- **WHEN** its configuration is inspected during integration
- **THEN** it is identified as unsuitable for the PPO acceptance path rather than supplying fabricated zero-reward learning evidence

### Requirement: SO-101 completes the same deployable-policy round trip

SO-101 acceptance SHALL demonstrate native execution, real NexuML PPO updates, finite metrics, intermediate-checkpoint resume, complete-policy export/reload, simulator-free policy inference, fresh-process simulation replay, and recorded behavior. It SHALL retain the shared state/action mapping and resource cleanup without requiring successful learned manipulation.

#### Scenario: The bounded SO-101 acceptance run completes
- **WHEN** native and NexuML tests run in the verified profile
- **THEN** actor changes, remaining-frame resume, deterministic reload equality, replay metrics/media, and complete simulator shutdown are evidenced
- **AND** no reward-convergence or cube-stacking claim is made

### Requirement: SO-101 stays within simulation-only scope

The reference SHALL document that five-joint simulated control does not validate real-robot calibration, collision safety, actuators, or deployment. Leader-arm teleoperation, demonstrations, ACT/VLA training, inverse-kinematics controller development, and hardware execution SHALL remain outside this change.

#### Scenario: Users consult the SO-101 instructions
- **WHEN** they follow the reference workflow
- **THEN** commands target simulation only and prerequisites/limitations distinguish this task from later LeRobot or supervised hardware work
