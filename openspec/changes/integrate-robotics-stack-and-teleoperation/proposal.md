# Proposal

## Why

The user selected ten Isaac-based robot/control examples and Unitree G1 as the first physical deployment target. NexuML needs a reusable workflow connecting native robot software, human demonstrations, simulation learning, deployment-equivalent testing and local adaptation. Build on the independently qualified interaction/inference handoff in decouple-simulation-learning-and-evaluation; reuse the broader Physical AI data, learning, safety and evaluation requirements rather than duplicate them.

## What Changes

- Establish full-stack simulation with ROS 2, ros2_control, optional Isaac ROS/Nav2/MoveIt 2 and Unitree SDK2/DDS alongside native GPU learning. Reuse native launch/configuration and isolate simulation from physical transports.
- Reuse native keyboard/gamepad/SpaceMouse/leader/XR tools and recording/conversion through thin CLI/SDK entrypoints. Define action authority, human takeover, requested/executed actions, timing, remote input and disconnect handling.
- Integrate optional bounded native asynchronous navigation, manipulation and whole-body skills. Support modular/frozen perception, trainable PyTorch perception-policy pipelines and hybrid control. Qualify optional ROSIDL Buffer/Torch interoperability against ordinary ROS.
- Deliver ten exact Isaac references in reference-catalogue.md, with native tasks/assets, actual learning, export/reload, evaluation and play. Require only selected matching MuJoCo and Gazebo/ROS examples, not all robots in every simulator.
- Deliver LeKiwi as the primary mobile manipulator simulation reference; G1 29-DoF body as the primary humanoid reference (exact EDU hardware and hands to confirm). Use official native simulator/DDS/whole-body/teleop tools, locomotion and manipulation learning, demonstrations, sim-to-sim and gated supervised physical G1 deployment. Provide compatible local adaptation evidence for NexuFL without production federation code.
- Preserve foundation play/viewer modes, local fit and existing umbrella algorithms, data and deployment safeguards. No new trainer, universal robot DSL, custom middleware, teleop UI, mandatory ROS dependency or sibling repository edits.

## Capabilities

### New Capabilities

- `robot-system-simulation`: native deployment-equivalent simulation and isolated clock/transport ownership.
- `native-robot-runtime`: optional native ROS/Unitree interfaces and inference/controller parity.
- `teleoperation-and-recording`: native input, action authority, interventions, recording/conversion.
- `hierarchical-robot-control`: typed native asynchronous skills and failure semantics.
- `perception-learning-integration`: modular, end-to-end and hybrid learning composition.
- `accelerated-ros-tensor-transport`: optional measured standard GPU buffer interoperability.
- `mobile-manipulator-reference`: selected Isaac catalogue and LeKiwi full-stack reference.
- `unitree-g1-reference`: exact native G1, DDS, locomotion/manipulation/IL, sim-to-sim and gated physical workflow.

### Modified Capabilities

No published baseline spec is replaced. The active umbrella's physical priority is explicitly updated to G1, retaining historical evidence and other planned references. See design for ownership.

## Impact

Optional library providers, existing CLI/SDK, data/interaction/policy components, native task/launch assets, existing CI image/profile mechanism, documentation and qualification. No hardware purchase, asset download, infrastructure change or physical actuation is authorized by this proposal.
