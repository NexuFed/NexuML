# Spec Delta

## Purpose

Prove that the same NexuML architecture trains and evaluates distinct obtainable machines with exact task, asset, actuator and deployment identities rather than generic scene-loading claims.

## ADDED Requirements

### Requirement: Every reference has an exact machine manifest

Reference manifests SHALL identify hardware revision and procurement/BOM route, asset source/hash/license, simulator/task/backend, sensors, joint order, controls, units, calibration assumptions and intended runtime. Asset loading, rollout, learning, evaluation and real-deployment support SHALL be separate statuses.

#### Scenario: A generic humanoid asset loads
- **WHEN** Unitree G1 support is assessed
- **THEN** that asset is not counted as the exact 29-DoF G1 training or deployment proof

### Requirement: Manipulation references include contact and bimanual learning

The arm workflow SHALL include the established Cartpole/Franka/SO-101 references, physical-contact SO-101 vial placement and maintained OpenArm unimanual/bimanual tasks. It SHALL preserve each task's distinct gripper, absolute/relative control and reset semantics and include actual learning/evaluation artifacts.

#### Scenario: Joint reaching succeeds with a fixed gripper
- **WHEN** pick-and-place acceptance is evaluated
- **THEN** reaching does not satisfy grasp, lift, transport, insertion and release requirements

### Requirement: Humanoid references cover locomotion and manipulation

An exact Unitree G1 29-DoF task SHALL have a trained standing/walking or velocity-tracking policy with fall, tracking and held-out-condition evidence. A maintained humanoid manipulation/imitation reference SHALL independently validate demonstration and policy execution; locomotion alone SHALL NOT establish manipulation support.

#### Scenario: Humanoid policies are exported
- **WHEN** they are replayed on the declared embodiment
- **THEN** action ordering and actuator behavior match each task and task-appropriate quality results are retained

### Requirement: Quadruped support requires a learned task

A Unitree Go2 velocity/terrain reference SHALL demonstrate native compatibility, NexuML learning, resume, export and held-out evaluation with its actual actuator/action conventions. A different robot or a scripted gait SHALL NOT count as this trained reference.

#### Scenario: Terrain friction changes during evaluation
- **WHEN** the frozen Go2 robustness suite runs
- **THEN** tracking, falls and constraint violations are reported separately from training reward

### Requirement: Drone identities and flight-controller boundaries remain distinct

A Crazyflie 2.1+ reference SHALL use a verified matching model and control timing for learned hover/goal tracking. PX4 software-in-the-loop SHALL use a PX4-compatible airframe and independent failsafe proof; it SHALL NOT be presented as the Crazyflie deployment path.

#### Scenario: A generic quadrotor or PX4 vehicle flies in simulation
- **WHEN** Crazyflie capability is reported
- **THEN** the matching-model learning gate remains open until its own physical/control contract is validated

### Requirement: PX4 accepts bounded policy commands through a supported interface

A separate PX4-compatible SITL reference SHALL execute an exported NexuML policy through a supported offboard or trajectory interface. It SHALL verify command bounds, frames, rate, heartbeat and offboard-loss failsafe behavior, then compare tracking and failsafe results against the classical controller under matched conditions. Native SITL startup alone SHALL NOT satisfy acceptance.

#### Scenario: A policy controls the PX4 simulated vehicle
- **WHEN** the frozen target-tracking cases run through the declared supported interface
- **THEN** actual policy-to-vehicle motion, tracking error and constraint/failsafe outcomes are recorded alongside the matched classical-controller reference
- **AND** neither result is presented as a Crazyflie deployment proof

#### Scenario: Offboard commands or heartbeat are lost
- **WHEN** the test deliberately interrupts the policy command stream
- **THEN** the configured native PX4 failsafe executes with recorded timing and state
- **AND** the policy runner reports the loss without bypassing the flight controller or claiming physical flight readiness

### Requirement: Mobile robotics includes a learned policy and classical baseline

A TurtleBot3 Waffle Pi or explicitly reviewed exact supported alternative SHALL have compatible assets, learned navigation/waypoint control and matched Nav2 comparison. Isaac-backed training and the retained Gazebo/ROS integration SHALL have separate evidence; localization alone SHALL NOT count as learning.

#### Scenario: Nav2 reaches goals successfully
- **WHEN** learned mobile-policy acceptance is assessed
- **THEN** independent NexuML parameter updates, held-out navigation results and a policy artifact are still required

### Requirement: Industrial references preserve task starting assumptions

A maintained gear-insertion task SHALL identify its exact UR10e or Rizon embodiment, pre-grasp assumption, perception inputs and controller. The roadmap's UR5e/URSim reference SHALL remain independently validated; neither model nor an insertion-only task SHALL silently substitute for the other.

#### Scenario: Gear insertion starts with an already grasped object
- **WHEN** task success is reported
- **THEN** the report does not claim a complete perception-to-pick-to-insert workflow unless those additional stages were actually exercised

### Requirement: Industrial intelligence does not require unsafe motion learning

A LumenPnP or approved versioned machine-data reference SHALL demonstrate offline prediction, inspection or anomaly detection using ordinary NexuML data/model workflows. Read-only OPC UA integration SHALL be exercised where available; absent maintained dynamics SHALL require an explicit feasibility assessment, not a fabricated simulator.

#### Scenario: No validated LumenPnP dynamics model exists
- **WHEN** the industrial example is delivered
- **THEN** its offline utility remains measurable and the simulator-engineering gap is separately recorded without commanding hardware

### Requirement: All embodiment proofs are attributable and comparable

Each required embodiment SHALL retain native-first checks, real learning, checkpoint/export/replay, resource/latency evidence and task-specific held-out metrics. Simulation proofs SHALL NOT imply hardware calibration, safety or shared weights across embodiments.

#### Scenario: One arm workflow is complete
- **WHEN** full multi-embodiment completion is requested
- **THEN** unverified humanoid, quadruped, drone and mobile references remain incomplete rather than inheriting the arm's result

### Requirement: Every example family has its own versioned proof record

reference-catalogue.md SHALL enumerate SO-101 state/reach/contact/camera/IL/VLA examples, OpenArm unimanual/bimanual, G1 exact locomotion/Mimic, Go2, Crazyflie/PX4, TurtleBot3, UR10e/Rizon, UR5e and LumenPnP with exact pinned assets and separate native/training/resume/export/evaluation/quality/hardware statuses. Bounded Allegro/Shadow/DexSuite reorientation and GR-1 Mimic examples remain explicitly optional until native source/task/license feasibility is proven.

#### Scenario: SO-101 five-joint reaching PPO passes
- **WHEN** whole-program multi-embodiment completion is considered
- **THEN** SO-101 vial physical contact, six-action grasp/release, visual control and all other required embodiment tasks remain open until independently tested
