# Proposal

## Why

NexuML has a verified, deliberately narrow Isaac/PPO integration, but not a complete Physical AI development backend. Make the Isaac ecosystem the primary robotics simulation platform while retaining NexuML's compiler, session, TorchRL learning, data, evaluation and portable-inference architecture across reinforcement learning, imitation, VLA, humanoids, world models and ROS deployment.

## What Changes

- Deliver one staged integration program, not a second framework or a single oversized implementation increment. "Full backend" means the explicitly specified workflow families and their executable acceptance references; it does not promise that every upstream task, robot, solver or NVIDIA product works interchangeably.
- **BREAKING development-environment migration:** target one automatically provisioned NexuVibe NVIDIA DevContainer environment using Python 3.12, one pinned CUDA-13 PyTorch family and a jointly resolved Isaac Lab/Newton/Isaac Sim/NexuML/TorchRL stack. Keep ordinary NexuML installation simulator-free. Reconcile the existing cu128 EA references and incompatible Meta-World requirements explicitly rather than install conflicting packages or silently drop requested capabilities.
- Reuse portable definitions and execution-local Isaac resources; cover manager/direct workflows, native vectorization, explicit physics/renderer presets, asset import, state/vision/depth/contact/IMU/range/tactile observations, randomization, curricula, deformable/coupled-physics references and real-to-sim scene provenance where the pinned upstream supports them.
- Extend the existing learner seams for on-policy PPO, off-policy learning, recurrence, asymmetric critics and multi-agent/scaled execution with real tests. Native RSL-RL and other upstream runners remain baselines, not replacement NexuML training engines.
- Add validated LeRobot/MCAP/Mimic experience ingestion, teleoperation and corrective demonstrations, ACT, teacher-to-vision distillation, pretrained SmolVLA and GR00T integration, full/parameter-efficient adaptation and algorithm-appropriate VLA post-training.
- Include actual learned-dynamics/world-model training and a DreamerV3 workflow, with sequence-aware replay, latent-state inference, imagination, held-out predictive/control evaluation and a conventional-policy baseline. Older Dreamer implementations or generic world-model documentation do not establish DreamerV3 support.
- Provide reproducible common release evaluation and Isaac Lab-Arena integration, task/embodiment/processor contracts, complete policy packaging and numerical/semantic deployment parity.
- Include ROS 2 execution, Isaac Sim bridge validation, Isaac ROS 5.0/Lyrical perception and supported deployment, MCAP-to-LeRobot ingestion and the independently required Gazebo/ros2_control proof. Retain explicit distro/process boundaries where native libraries cannot safely coexist.
- Deliver trained arm, humanoid, quadruped, drone and mobile-robot references, plus industrial insertion and a non-actuating industrial-data reference. Preserve exact obtainable hardware/asset identity; no generic humanoid or quadrotor substitutes for a named machine.
- Provide secure SSH/browser/WebRTC inspection and bounded recording, followed by operator-supervised real-arm validation, local offline adaptation, regression comparison, promotion and rollback. No autonomous real-hardware exploration or fleet/federation implementation.
- Preserve all existing completion evidence. Track every roadmap checkbox and Isaac capability family to requirements, tasks and evidence; prerequisites, blockers and manually gated proofs remain visible and do not count as completed capabilities.

## Capabilities

### New Capabilities

- `physical-ai-environment-composition`: One automatic shared development environment, dependency authority, native prerequisites and compatibility migration.
- `physical-policy-contracts`: Versioned physical-system, multimodal/temporal, processor, state and portable-artifact semantics.
- `isaac-simulation-workflows`: Upstream task/asset/physics/sensor/randomization and advanced-scene execution through the existing environment boundary.
- `isaac-torchrl-learning`: Correct native rollouts and algorithm-appropriate learning through the existing session/Lightning/TorchRL lifecycle.
- `robot-experience-workflows`: Demonstrations, synthetic trajectories, multi-rate data, corrective experience and maintained interchange formats.
- `robot-imitation-learning`: ACT, behavior cloning, teacher/student distillation and complete offline-to-simulation round trips.
- `vision-language-action-workflows`: Pretrained VLA inference, adaptation, chunk execution and supported RL post-training.
- `physical-world-model-learning`: Learned dynamics, DreamerV3, imagination and stateful policy export with comparative evidence.
- `physical-policy-evaluation`: Frozen release suites, Arena, robustness, forgetting and explicit candidate decisions.
- `ros-physical-ai-runtime`: ROS/Gazebo/Isaac bridges, accelerated perception, recording and supported runtime integrations.
- `isaac-embodiment-workflows`: Native-validated, actually trained multi-embodiment and industrial references.
- `physical-policy-deployment`: Inference parity, supervised sim-to-real and a local offline improvement loop.
- `isaac-execution-and-inspection`: Process ownership, remote viewing, distributed execution and optional GPU/system evidence tiers.

### Modified Capabilities

None of the published specifications changes requirements. Existing export, dependency-manifest, CLI, discovery and supervised contracts remain authoritative. The unarchived Isaac, RL and MuJoCo changes are dependencies/coordination points, not silently rewritten specifications; see `design.md` and `coverage.md`.

## Impact

- NexuML: targeted additions to existing core contracts/compiler/policy/export, data, reinforcement/training, interaction, evaluation/CLI and library providers/models/scenarios. No `IsaacSession`, alternative compiler, duplicate trainer, robotics superclass or mandatory simulator dependency in core.
- Tooling: optional extras, repository lock, focused tests, opt-in native/GPU/ROS/multi-device/hardware gates, documentation and durable evidence. Maintain one implementation owner for shared outcome, temporal-policy and learner extensions.
- Coordinated external delivery: NexuVibe owns the consumer lock, Python/backend selection, Docker/system dependencies, automatic setup and interpreter settings. Its reviewed changes must be delivered in that repository; this repo-local proposal does not edit it or authorize rebuilding an active container.
- Scope: roadmap Phases 0–10 and X1–X5 are traced, with Isaac-related/supporting work included and independent work explicitly assigned. World models and humanoids are required workstreams, not omitted future wishes. Hardware proofs remain manual; NexuFL implementation, fleet operations, simulator/driver rewrites and safety certification remain excluded.
- Proposal creation installs nothing, downloads no simulator/assets, launches no GPU job, changes no existing proposal and performs no hardware/system operation.
