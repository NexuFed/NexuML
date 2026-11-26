# Tasks

All unchecked. Reuse matching foundation/umbrella evidence and fixtures; native checks on approved prepared resources only. No procurement or physical actuation authorized.

## 0. Foundation and source gate
- [ ] 0.1 Consume qualified foundation purpose/capabilities/action-source/timing/inference handoff (§1.7–1.8) and relevant native gate. Preserve local fit/export/play; do not wait for all Ray scaling.
- [ ] 0.2 Freeze R01–R10 exact native sources/tasks/assets/licenses/hashes/controls; inspect official LeKiwi URDF and LeIsaac assets, exact 29-body-DoF G1 and hand mapping. Separate asset-only from existing task; record blockers.
- [ ] 0.3 Audit pinned ROS/Isaac ROS, Sim bridge, SDK2 and native teleop versions/ABI in prepared profiles. Define simulated/physical domain isolation, no-hardware prerequisites and controller fallback.

## 1. Full-stack simulation and inference
- [ ] 1.1 Extend existing ROS transport with minimal typed observation/command/action/service binding; standard messages, TF/clock/QoS, freshness/deadlines, explicit simulated endpoints. Test actual system as well as fakes.
- [ ] 1.2 Native deployment-equivalent Isaac Sim ROS/ros2_control launch; preserve separate Gazebo fixture, swap simulated driver behind compatible controller, verify actual policy motion/recording, reset/failure/cancel/repeated cleanup; no ROS hot-loop overhead for RL.
- [ ] 1.3 Native Unitree SDK2/DDS simulator provider with isolated domains, no physical discovery/commands, actual controller state, freshness and safe failure; no redundant ROS wrapper.
- [ ] 1.4 Complete exported policy parity on recorded sequences and simulated hardware, native command timing; optional optimized runtime only after parity.

## 2. Teleoperation and recording
- [ ] 2.1 Thin CLI/Python teleop and optional record via native tools; keyboard/gamepad/SpaceMouse/leader/XR capability checks, simulated operator session and play/lifecycle reuse.
- [ ] 2.2 Explicit source ownership, takeover, requested/executed actions, bounded controls, remote input latency/deadman/disconnect and cleanup independently of viewer. No custom UI/protocol/network management.
- [ ] 2.3 Native HDF5/MCAP/LeRobot conversion and ingestion, original data, timestamps/calibration/splits/outcomes/interventions; actual human demos in simulated LeKiwi/G1 when approved devices available; scripted data separate.
- [ ] 2.4 Offline BC/ACT or compatible learner actual updates/resume/export/play and held-out outcomes; reuse umbrella Mimic/DAgger/IL acceptance.

## 3. Perception and skills
- [ ] 3.1 Modular external/frozen ROS perception observation, compare native features/timing/availability; no privileged deployed actor input.
- [ ] 3.2 Trainable PyTorch vision encoder and actor, explicit frozen/shared scope, processor/action parity; reuse visual-policy/VLA evidence; no ROS/TensorRT autograd assumption.
- [ ] 3.3 Native bounded asynchronous Nav2 and MoveIt 2/compatible planning and G1 whole-body skill goals; verify duration/timeout/failure/cancel, no fixed-step misrepresentation; controller safety/balance independent.

## 4. Required Isaac examples
- [ ] 4.1 R01 Cartpole, R02 calibrated SO-101, R03 Franka Panda: requalify exact existing references and outstanding contact proof against foundation; reuse valid evidence.
- [ ] 4.2 R04 Kaya, R05 RidgebackFranka: native control, actual learning/resume/export, held-out holonomic/base-arm outcomes and play; classical comparison.
- [ ] 4.3 R06 Spot with arm, R07 A2D: exact licensed native assets, arm/mimic controls and learned skills, complete round trip and outcomes; no bare Spot or different humanoid substitute.
- [ ] 4.4 R09 Crazyflie: exact hover/goal task with disturbance and control evidence; separate PX4 roadmap.
- [ ] 4.5 R10 LeKiwi: reuse LeIsaac native task/model or validate official URDF conversion; human teleop and recording, actual learned manipulation, resume/export/play, full-stack navigate/pick/transport/place and classical baseline.
- [ ] 4.6 R08 G1 (may begin independently of R04–R07/R09): official exact 29-body-DoF and hand mapping, locomotion and separate manipulation/IL, actual updates, fall/tracking/contact outcomes, resume/export/play.

## 5. G1 system and selected alternate simulator gates
- [ ] 5.1 Official Unitree Isaac and matching MuJoCo simulator/SDK, actor/control parity and actual sim-to-sim outcomes with isolated DDS. Reuse foundation matched learning/local-fit; no full catalogue port.
- [ ] 5.2 Native Unitree XR/LeRobot or compatible NVIDIA Physical AI simulated reference, retained qualified balance/whole-body control, human manipulation demos, conversion/training/replay and intervention evidence.
- [ ] 5.3 Full-stack LeKiwi and G1 simulated hardware, actual policy/perception/controller/skill commands, clock/reset/failure/disconnect/cancel, isolation from real robot.
- [ ] 5.4 Prepare later physical G1 EDU gate (first humanoid, after SO-101/LeKiwi physical milestones; software readiness can proceed independently): exact body/hands/firmware/compute/access, approved protective workspace/support, independent stop and qualified native controller; recorded-input and sim-to-sim/full-stack proof. No motion without separately explicit authorization; hardware absence blocks physical gate only.
- [ ] 5.5 When separately authorized, supervised bounded G1 manipulation and capture, real data validation, bounded offline local fit, old/new-domain evaluation/rollback. Feed foundation three-domain/mixed-objective gate with compatible real experience (earlier SO-101/LeKiwi evidence may satisfy it where contracts match); no auto deployment or production NexuFL implementation.

## 5a. First physical learning milestones (independently scheduled before G1 physical trials)
- [ ] 5a.1 Qualify calibrated SO-101 leader/follower and camera recording, matching native sim and recorded-input parity, bounded controls, independent stop, operator takeover and explicit physical authorization; no actuation on ordinary CI or missing enablement.
- [ ] 5a.2 When separately authorized, demonstrate bounded SO-101 task execution, ACT baseline and compatible pretrained SmolVLA fine-tuning from approved demonstrations/corrections; compare frozen held-out sim and physical tasks, log actual actions/outcomes/interventions and retain known-good rollback. No uncontrolled physical RL.
- [ ] 5a.3 After SO-101, qualify exact LeKiwi V1 base + arm calibration, synchronized controls, safe stop/disconnect and separately authorized supervised mobile manipulation; collect real episodes and demonstrate bounded offline adaptation and held-out evaluation. Reuse compatible receipts for umbrella tasks and foundation local-fit gate; unavailable hardware blocks only its physical proof. Bimanual systems are a later candidate, not a prerequisite for G1.

## 6. Optional acceleration and release
- [ ] 6.1 Standard ROS baseline then probe rosidl::Buffer/CUDA/Torch, device residency/copies/latency/throughput/output parity and lifetime/synchronization. Explicit fallback/unsupported status; acceleration does not block ordinary ROS.
- [ ] 6.2 Optional profiles and existing CI image workflow; document native tasks and train/evaluate/play/teleop/record, full-stack launch and private network/hardware prerequisites; no mandatory ROS/system installer.
- [ ] 6.3 Publish ten-row revision-bound evidence matrix (asset, native rollout, actual learning, resume, export, held-out quality, play/media; separate system/hardware status); run focused regression and `openspec validate integrate-robotics-stack-and-teleoperation --strict`. Reconcile overlaps before archival; reuse existing receipts, no skipped or mock-only native passes.
