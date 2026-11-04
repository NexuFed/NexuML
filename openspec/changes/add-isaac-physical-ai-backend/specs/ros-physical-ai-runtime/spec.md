# Spec Delta

## Purpose

Connect exported NexuML policies to ROS simulation, accelerated perception and supported robot runtimes while retaining explicit middleware, timing and controller boundaries.

## ADDED Requirements

### Requirement: ROS execution is inference only

The ROS runtime SHALL load a complete approved policy and map validated observations to bounded high-level controller commands without constructing a training learner. Simulator and robot transport resources SHALL remain outside policy artifacts and serializable task definitions.

#### Scenario: A policy runs against a simulated arm controller
- **WHEN** observations arrive at the declared rate
- **THEN** inference produces correctly ordered and bounded commands with traceable policy, processor and embodiment identities

### Requirement: Distro and Python boundaries are explicitly validated

The target accelerated runtime SHALL use Isaac ROS 5.0 and ROS 2 Lyrical. Any Isaac Sim bridge requiring another supported distro SHALL use a separately validated process/workspace boundary. Cross-distro communication and custom-message Python ABI compatibility SHALL NOT be assumed from shared installation.

#### Scenario: The selected Sim bridge documents Jazzy while perception uses Lyrical
- **WHEN** an integrated workflow is prepared
- **THEN** standard message, discovery, QoS, timing and failure behavior are tested across the chosen boundary or the workflow remains blocked

### Requirement: Timing and controller failure behavior are observable

The runtime SHALL validate timestamps, simulation clock, frames, input freshness, command rate, stale commands, reset and shutdown. External controllers SHALL own watchdogs, safe fallback, collision protection and emergency stops; a model's action bounds SHALL NOT be described as equivalent safeguards.

#### Scenario: Required joint state becomes stale
- **WHEN** the configured freshness limit is exceeded
- **THEN** policy command publication stops or follows the explicitly validated controller fallback and the fault is recorded

### Requirement: The independent Gazebo proof remains required

A live headless Gazebo and ros2_control reference SHALL demonstrate reset, repeated start/stop and actual policy-driven simulated movement through the existing ROS adapter. Isaac Sim bridge success SHALL NOT substitute for this original outstanding system acceptance.

#### Scenario: Only Isaac Sim ROS execution has passed
- **WHEN** original ROS/Gazebo completion is assessed
- **THEN** the Gazebo gate remains open with its separate prerequisites and evidence

### Requirement: Accelerated perception supplies standard contracted inputs

FoundationPose or a maintained equivalent SHALL supply a manipulation observation; cuVSLAM and nvBlox SHALL support a mobile reference where compatible. Frames, units, availability, latency and resource use SHALL be measured against a plain ROS baseline. Classical perception/navigation SHALL NOT count as a learned robot policy.

#### Scenario: Pose estimates are unavailable or in another frame
- **WHEN** manipulation inference requires that pose
- **THEN** the observation is rejected or explicitly transformed through the calibrated frame contract rather than using a stale oracle substitute

### Requirement: Recording uses maintained ROS and episode tooling

ROS experience SHALL be captured with maintained rosbag2/MCAP tools and converted through a validated MCAP-to-LeRobot workflow where applicable. Episode boundaries, dropped messages, calibration and action/observation alignment SHALL be verified before offline learning.

#### Scenario: Several recording sessions are converted
- **WHEN** their timelines or calibration revisions differ
- **THEN** conversion retains session identity and validates the explicit alignment/split policy without silently joining incompatible streams

### Requirement: Accelerated planning and deployment remain capability checked

Supported cuMotion/MoveIt planning and Isaac ROS Deploy integration SHALL remain separate from learning. LEAPP, ONNX, Triton or controller-specific deployment SHALL require a compatible artifact contract and parity proof; arbitrary ACT, VLA or Dreamer packages SHALL NOT be accepted by assumption.

#### Scenario: A policy has no valid accelerated converter
- **WHEN** deployment is requested
- **THEN** the supported reference inference runtime remains available and the unsupported optimized target is reported explicitly

### Requirement: Native ROS provisioning is optional for library users

ROS and Isaac ROS system packages SHALL be provisioned through the coordinated container/workspace setup, not base Python dependencies or import-time installers. Isaac ROS release limitations and removed APIs SHALL be reflected in version-specific documentation and tests.

#### Scenario: A normal CPU test environment has no ROS installed
- **WHEN** NexuML imports and fake-transport tests execute
- **THEN** they require no middleware installation, hardware access or implicit simulator download
