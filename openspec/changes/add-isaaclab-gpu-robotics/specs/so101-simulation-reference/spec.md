# Spec Delta

## Purpose

Include required SO-101 simulation references for five-joint reaching and upstream contact-based vial placement, preserving distinct task/profile/control identities and complete NexuML policy round trips.

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

The joint-reaching reference SHALL reuse the pinned upstream SO-101 asset with its Isaac Sim PhysX variant. It SHALL record asset source/version or content identity, license/access prerequisites, joint names/order, authored limits, actuator configuration, fixed gripper position, and units. Missing or incompatible assets SHALL fail clearly without automatic model replacement.

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

Each SO-101 reference SHALL demonstrate native execution, real NexuML PPO updates, finite metrics, intermediate-checkpoint resume, complete-policy export/reload, simulator-free inference, fresh-process replay and recorded behavior. Each SHALL retain its declared state/action/profile meaning and cleanup. Bounded update proofs SHALL NOT imply successful learned manipulation.

#### Scenario: The bounded SO-101 acceptance run completes
- **WHEN** native and NexuML tests run in the verified profile
- **THEN** actor changes, remaining-frame resume, deterministic reload equality, replay metrics/media, and complete simulator shutdown are evidenced
- **AND** no reward-convergence or cube-stacking claim is made

### Requirement: SO-101 stays within simulation-only scope

Both references SHALL document that simulation does not validate real-robot calibration, collision safety, actuators or deployment. Leader-arm teleoperation, demonstration collection, ACT/VLA training, camera-policy training/distillation, sim2real presets, inverse-kinematics controller development and hardware execution SHALL remain outside this change.

#### Scenario: Users consult the SO-101 instructions
- **WHEN** they follow the reference workflow
- **THEN** commands target simulation only and prerequisites/limitations distinguish this task from later LeRobot or supervised hardware work

### Requirement: Upstream vial placement is a separate required reference

The integration SHALL expose the pinned upstream `IsaacTutorial-Place-Vial-SO101` state task through ordinary NexuML configuration and training in its explicit Newton MJWarp profile. It SHALL reuse upstream task logic, assets and reset data rather than rename reaching or replace NexuML with the upstream learner. Existing EA/PhysX references and their artifacts SHALL retain their meanings.

#### Scenario: A user selects the vial-placement scenario
- **WHEN** its approved isolated tutorial profile and pinned assets are available
- **THEN** the normal session, Lightning, TorchRL and PPO path trains the upstream contact-based task
- **AND** selecting the existing joint-reaching scenario still produces five arm actions with a fixed gripper

### Requirement: Vial controls retain six relative-joint action semantics

Vial placement SHALL command five ordered arm joints plus the gripper through bounded continuous actions, with verified joint limits, physical units and actuator identity. Nominal targets SHALL follow upstream measured-position-relative semantics with arm scale 0.033 rad and gripper scale 0.02 rad, 120 Hz physics and 30 Hz policy control. EA absolute-target or fixed-gripper semantics SHALL NOT be substituted.

#### Scenario: Native arm and gripper response is checked
- **WHEN** bounded scripted actions execute under the nominal tutorial configuration
- **THEN** all six declared controls respond with the expected order, sign, scale and timing within the validated articulation constraints
- **AND** packaged policies reject incompatible action/profile mappings rather than relying only on tensor width

### Requirement: Manipulation is achieved through physical contact

The vial task SHALL preserve upstream contact-based grasping, lifting, transport, insertion and release. After reset it SHALL NOT attach the vial to the gripper, teleport the object, bypass required contacts or substitute reaching reward. Acceptance SHALL include a bounded native contact-based reference trajectory from a canonical home start, separately attributed from any learned NexuML policy.

#### Scenario: A native reference trajectory places the vial
- **WHEN** the arm picks up the tabletop vial and moves it into a rack hole
- **THEN** retained state/contact evidence demonstrates physical pickup, lift, insertion and release without post-reset object-pose writes
- **AND** this validates task feasibility, not that a short NexuML PPO run learned the behavior

### Requirement: Placement success uses the upstream physical predicate

Vial placement SHALL preserve pinned rewards, termination and timeout rules and report task success separately from return. Success SHALL require the upstream progress history and an upright, mechanically seated, released and nearly still vial for the pinned ten-control-step hold. Held, airborne or merely nearby vials SHALL NOT count. Per-episode outcomes SHALL survive autoreset without inheriting the next episode's flags.

#### Scenario: Success and false positives are tested
- **WHEN** controlled native episodes exercise valid placement, insufficient hold, held/nearby vials, drops and timeout
- **THEN** outcome, reward and termination/truncation agree with the pinned rules and reset at the next episode
- **AND** the evaluator reports placement success and available grasp/lift/insertion outcomes without fabricated joint-reaching error

### Requirement: Curriculum training is distinct from home-start evaluation

Training SHALL retain the pinned eight-phase reset curriculum and nominal tutorial variation with verified reset-data provenance. Replay SHALL start from canonical home states with upstream nominal play settings and traceable dataset rows/seeds. Curriculum-phase success SHALL NOT be reported as full home-to-placement success. Original reset data SHALL NOT be overwritten, and policy determinism SHALL NOT enable unsupported deterministic physics kernels.

#### Scenario: A policy trained with late-phase resets is replayed
- **WHEN** fresh-process evaluation begins
- **THEN** it uses the documented home-start distribution rather than a release-phase reset and reports complete-episode placement success separately from training return

#### Scenario: A deterministic policy is requested
- **WHEN** the normal evaluator chooses deterministic policy actions
- **THEN** the pinned Newton physics mode remains unchanged rather than enabling the tutorial's documented failing Warp-wide deterministic mode
