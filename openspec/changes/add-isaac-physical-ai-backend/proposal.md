# Proposal

## Focused native configuration, sensors, Viser and MARL ownership

[generalize-isaac-native-configuration-and-marl](../generalize-isaac-native-configuration-and-marl/proposal.md)
owns implementation and acceptance of its early Viser feasibility gate and
milestones A–D. This umbrella remains the broader roadmap, not a duplicate queue
for that slice. Its focused requirements govern the current unmerged native
schema, strict read-only viewing, native TorchRL IPPO/MAPPO and preserved SO-101
SDK/provenance boundaries. Historical evidence and completed tasks remain intact;
MuJoCo, ROS 2, CARLA, ACT/Diffusion/VLA/world-model, robotics and independent
federation obligations are neither removed nor completed by that focused change.

## Linear issue and PR linkage

- **Implementation issue:** [NEX-234](https://linear.app/nexufed/issue/NEX-234) — SO-101 local imitation learning and NexuML Python SDK handoff (Gate 0 / F01).
- **PR branch (recommended from Linear):** `feature/nex-234-so-101-local-imitation-learning-and-nexuml-python-sdk`. PR title example: `NEX-234: SO-101 local behavior cloning and Python SDK handoff`.
- **PR body:** include `Closes NEX-234` **only** on the PR that actually satisfies verified tasks 0.1–0.8. Planning-only PRs should say `Related to NEX-234`; do not close the issue.
- Keep the PR small: do not merge the entire unfinished Physical AI umbrella to deliver the handoff. Later FL implementation is tracked by [NEX-235](https://linear.app/nexufed/issue/NEX-235).
- **Status automation:** linking alone is not a guarantee of transitions; the Linear↔GitHub integration and team's workflow automations must be configured to move the issue on PR creation/review/merge.


## Why

NexuML has a verified, deliberately narrow Isaac/PPO integration, but not a complete Physical AI development backend. Make the Isaac ecosystem the primary robotics simulation platform while retaining NexuML's compiler, session, TorchRL learning, data, evaluation and portable-inference architecture across reinforcement learning, imitation, VLA, humanoids, world models and ROS deployment.

## Priority: early SO-101 local-learning / NexuFL handoff

Before the full Isaac/Physical AI programme, deliver a **small, real SO-101 state-based behavior-cloning proof** with verified native expert episodes, actual NexuML model updates, clean-process export and fresh closed-loop task success. Expose the existing NexuML training/session lifecycle through a **bounded Python SDK local-fit and portable model-state contract** usable by a trusted, installed external caller. No second trainer, `FederatedTrainer`, mandatory REST server, gRPC or federation logic in NexuML. The same source shall work without a web server; NexuFL owns its own outbound `ClientProgram`/gRPC orchestration.

**Stop for an early, narrow NexuML PR** after the local training/data/export/SDK interface is accepted, *not* after ACT, Diffusion, world models, ROS, VLA, other embodiments or the whole umbrella change. Pin the reviewed wheel/commit and handoff contract for NexuFL's separately planned `federate-so101-imitation-learning`. The full programme and all independently required proof remain open. See `federation-handoff.md`, `tasks.md` §0 and the `local-learning-handoff` spec. Planning alone is not implementation evidence.

## What Changes

- Deliver one staged integration program, not a second framework or a single oversized implementation increment. "Full backend" means the explicitly specified workflow families and their executable acceptance references; it does not promise that every upstream task, robot, solver or NVIDIA product works interchangeably.
- **Versioned compatibility instead of one impossible all-dependencies installation:** provision one primary, automatically reproducible Python-3.12/CUDA-13 Isaac-first NexuVibe environment. Keep ordinary NexuML simulator-free and do not constrain the entire distribution for optional Meta-World. Use a separately pinned/approved execution profile only when tested upstream MuJoCo/Meta-World/JAX/ROS constraints genuinely cannot coexist, with explicit portable contract and independent evidence.
- **Consumer-selected versions, minimally restrictive packages:** Python 3.12/CUDA 13 and exact dependency/source versions describe this development/validation environment, not universal NexuML requirements. Published packages retain genuine minimums and only evidence-backed incompatibility bounds; the consuming project's root uv configuration owns Python/backend/index selection, exact pins, approved overrides and lock. Put the optional Physical AI stack in library-owned `robots` with root forwarding, outside `all`; the development consumer explicitly requests both. Real upstream wheel/API/ABI restrictions remain visible rather than being ignored or exported as arbitrary framework-wide caps.
- Permit temporary consumer-owned constraint exceptions for the recorded `packaging`, Transformers, NumPy, urllib3 and isolated-importer usd-exchange conflicts, with exact version/source/lock identities, original/replacement bounds, targeted tests and removal conditions. Preserve Lab's urllib3 security floor and require NumPy/native-USD compatibility proof; resolution alone does not validate an exception. No blanket `--no-deps`, edited upstream metadata, implicit MuJoCo physics override or misleading clean dependency-check claim.
- Bring the previous ordinary MuJoCo Humanoid/Reacher PPO and Meta-World MT1 reach workflows into this change with native transitions, success-only metrics, EGL/OSMesa recording, resume/export/replay and a frozen three-seed 50-goal comparison. Add three bounded GPU MuJoCo Playground/MJWarp references (small control, articulated robot and vision); upstream JAX learning is explicitly baseline-only until genuine NexuML learning and checkpoint ownership are verified.
- Reuse portable definitions and execution-local Isaac resources; cover manager/direct workflows, native vectorization, explicit physics/renderer presets, asset import, state/vision/depth/contact/IMU/range/tactile observations, randomization, curricula, deformable/coupled-physics references and real-to-sim scene provenance where the pinned upstream supports them.
- Extend the existing learner seams for on-policy PPO, off-policy learning, recurrence, asymmetric critics and multi-agent/scaled execution with real tests. Native RSL-RL and other upstream runners remain baselines, not replacement NexuML training engines.
- Make reference-catalogue.md the required machine/example inventory: preserve Cartpole, Franka, SO-101 Reach and six-action physical-contact Vial Placement, wrist-camera teacher/student and camera-PPO-from-scratch, ACT/Diffusion/flow policies and compatible SmolVLA/GR00T N1.7/pi0.5. Preserve OpenArm, G1, Go2, Crazyflie/PX4, TurtleBot3, UR5e/UR10e/Rizon and LumenPnP; add a bounded dexterous-hand reference and optional GR-1 exercise after source/license feasibility.
- Add validated LeRobot/MCAP/Mimic experience ingestion, teleoperation and corrective demonstrations, ACT, teacher-to-vision distillation, pretrained SmolVLA and GR00T integration, full/parameter-efficient adaptation and algorithm-appropriate VLA post-training.
- Expand the LeRobot examples with a complete Diffusion Policy round trip and a resource-gated pretrained π0.5 adaptation/evaluation reference. Compare against existing policies only on matching data, embodiment and observation/action contracts; reuse NexuML training and inference rather than an upstream trainer.
- Add a simulation-first DAgger/correction workflow, offline-to-online RLPD and reward-classifier/HIL-SERL proof, a distinct assessment of Robometer/TOPReward and bounded asynchronous chunk inference; learned rewards never replace measured task success or safety boundaries.
- Include actual learned-dynamics/world-model training and a DreamerV3 workflow, with sequence-aware replay, latent-state inference, imagination, held-out predictive/control evaluation and a conventional-policy baseline. Older Dreamer implementations or generic world-model documentation do not establish DreamerV3 support.
- Assess VLA-JEPA, FastWAM and SARM as explicitly bounded research candidates: inspect dependency/API, model/license, data/embodiment and resource feasibility before proposing integration. Their required deliverable is a reviewable feasibility report, not full training or advertised support. Auxiliary video prediction does not substitute for DreamerV3 imagination; learned progress never replaces measured task success or safety controls.
- Include pinned compatible LeRobot evaluation routes (native LIBERO/LIBERO-plus and optional RoboTwin subset), not a copied generic benchmark engine. Retain native Meta-World MT1 scoring and independently frozen matched task/sensor/action comparisons.
- Provide reproducible common release evaluation and Isaac Lab-Arena integration, task/embodiment/processor contracts, complete policy packaging and numerical/semantic deployment parity.
- Include ROS 2 execution, Isaac Sim bridge validation, Isaac ROS 5.0/Lyrical perception and supported deployment, MCAP-to-LeRobot ingestion and the independently required Gazebo/ros2_control proof. Retain explicit distro/process boundaries where native libraries cannot safely coexist.
- Deliver trained arm, humanoid, quadruped, drone and mobile-robot references, plus industrial insertion and a non-actuating industrial-data reference. Preserve exact obtainable hardware/asset identity; no generic humanoid or quadrotor substitutes for a named machine.
- Provide secure SSH/browser/WebRTC inspection and bounded recording, followed by operator-supervised real-arm validation, local offline adaptation, regression comparison, promotion and rollback. No autonomous real-hardware exploration or fleet/federation implementation.
- Preserve all existing completion evidence. Track every roadmap checkbox and Isaac capability family to requirements, tasks and evidence; prerequisites, blockers and manually gated proofs remain visible and do not count as completed capabilities.
- Create one focused local Git commit after each verified implementation milestone, containing only milestone-owned source, tests, user documentation, configuration/locks and planning changes. Do not commit execution evidence, reports, logs, generated run results/media, checkpoints or weights, including evidence embedded in otherwise committable files. Retain it locally or in an explicitly approved artifact location; preserve existing committed history. Push, review publication, merge and archive still require separate direction.
- **Current autonomous campaign:** the user authorizes direct task-related NexuVibe changes while preserving its other work, and installs/experiments in one isolated venv in this NexuML worktree. Do not rebuild the active DevContainer, replace `/env`, change drivers or disturb other jobs; retain disk headroom. Prioritize Isaac and defer large-model downloads/training for now. Keep their original required gates open rather than claiming full-program completion. Further documented dependency selections/overrides may be evaluated with security/API/ABI checks; do not blindly force physics compatibility or create hidden runtime environments.

## Unified implementation authority

This folder is the single implementation queue for the Isaac, native MuJoCo/Meta-World, MuJoCo Playground and complete Physical AI reference programme. The earlier add-mujoco-reference-workflows change is preserved as a historical source but **must not be implemented in parallel**: its three capability specs and acceptance obligations are imported here. Earlier add-isaaclab-gpu-robotics native results remain historical and must be re-proven on a changed stack. Treat the 236-checkbox roadmap snapshot and reference-catalogue.md as coverage/evidence inventories, not implementation completion claims.

"Implement in one go" means a **single coherent, staged campaign** using tasks.md with separate vertical-slice tests and a final integrated acceptance pass. It does not mean one massive framework rewrite, forcing incompatible physics/JAX wheels into one Python installation, or bypassing real hardware/resource/license approval. Each reference reports native operation, NexuML-owned training, resume, exported policy, fresh replay, multi-seed quality and separately gated hardware status.

## Capabilities

### New Capabilities

- `physical-ai-environment-composition`: One automatic shared development environment, dependency authority, native prerequisites and compatibility migration.
- `physical-policy-contracts`: Versioned physical-system, multimodal/temporal, processor, state and portable-artifact semantics.
- `isaac-simulation-workflows`: Upstream task/asset/physics/sensor/randomization and advanced-scene execution through the existing environment boundary.
- `isaac-torchrl-learning`: Correct native rollouts and algorithm-appropriate learning through the existing session/Lightning/TorchRL lifecycle.
- `robot-experience-workflows`: Demonstrations, synthetic trajectories, multi-rate data, corrective experience and maintained interchange formats.
- `robot-imitation-learning`: ACT, Diffusion Policy, behavior cloning, teacher/student distillation and complete offline-to-simulation round trips.
- `local-learning-handoff`: Versioned bounded Python SDK local-fit, numeric model-state handoff and SO-101 behavior-cloning acceptance/early PR gate.
- `vision-language-action-workflows`: Pretrained SmolVLA, GR00T and π0.5 inference/adaptation, chunk execution and supported RL post-training.
- `physical-world-model-learning`: Learned dynamics, DreamerV3, imagination and stateful export, plus separate VLA-JEPA/FastWAM feasibility assessments.
- `physical-policy-evaluation`: Frozen release suites, Arena, robustness, forgetting, explicit candidate decisions and bounded SARM feasibility.
- `ros-physical-ai-runtime`: ROS/Gazebo/Isaac bridges, accelerated perception, recording and supported runtime integrations.
- `isaac-embodiment-workflows`: Native-validated, actually trained multi-embodiment and industrial references.
- `physical-policy-deployment`: Inference parity, supervised sim-to-real and a local offline improvement loop.
- `isaac-execution-and-inspection`: Process ownership, remote viewing, distributed execution and optional GPU/system evidence tiers.
- `mujoco-reference-workflows`: MuJoCo Humanoid/Reacher and Meta-World reach reference round trips, media and benchmark.
- `rl-task-outcomes`: Independently validated native task success and optional task-specific error outcomes.
- `physical-ai-dependency-composition`: Truthful optional extras and tested native compatibility/execution lanes.
- `mujoco-gpu-playground`: GPU MuJoCo/MJWarp environment integration and clearly separated JAX-native baselines.
- `interactive-robot-learning`: Corrective data, offline-to-online RL, learned rewards, flow policies and temporal inference.

### Modified Capabilities

None of the published specifications changes requirements. Existing export, dependency-manifest, CLI, discovery and supervised contracts remain authoritative. The original RL/Isaac changes remain inherited foundation, while the earlier MuJoCo proposal is now an implementation-superseded historical input whose required specs are imported here. No separate MuJoCo task queue is to be executed; see design.md, reference-catalogue.md and coverage.md.

## Impact

- NexuML: targeted additions to existing core contracts/compiler/policy/export, data, reinforcement/training, interaction, evaluation/CLI and library providers/models/scenarios, now including one shared MuJoCo/Meta-World outcome owner and GPU-Playground/reward/correction adapters. No `IsaacSession`, alternative compiler, duplicate trainer, robotics superclass or mandatory simulator dependency in core.
- Tooling: optional `robots` extras separate from `all`, minimally restrictive published compatibility metadata, consumer-root version/backend/override authority, repository development lock, focused tests, opt-in native/GPU/ROS/multi-device/hardware gates, documentation and durable evidence. A repository lock or validated reference pin is not a version requirement imposed on downstream installations. Maintain one implementation owner for shared outcome, temporal-policy and learner extensions.
- Coordinated external delivery: NexuVibe owns the consumer lock, Python/backend selection, Docker/system dependencies, automatic setup and interpreter settings. The user authorizes task-related changes directly there while preserving its other work. This does not authorize rebuilding the active DevContainer or modifying its shared environment.
- Scope: roadmap Phases 0–10 and X1–X5 are traced, with Isaac, MuJoCo, Meta-World, Playground and compatible benchmark/example work included here; genuinely independent CARLA and manual hardware gates remain explicit. World models and humanoids are required workstreams, not omitted future wishes. Hardware proofs remain manual; NexuFL implementation, fleet operations, simulator/driver rewrites and safety certification remain excluded.
- Model additions are plan-only: exact releases/weights and measured training/inference budgets remain acceptance gates. No implicit `lerobot[all]`, catalogue-wide support, foundation-model pretraining, large-model download or paid-compute submission. On 2026-10-19 the user approved the D3 exceptions, subsequently including NumPy, urllib3 and isolated-importer USD, and starting implementation. Simulator-independent slices may proceed while full primary resolution/provisioning remains open; dependent native/model slices retain every prerequisite. This supersedes earlier policy snapshots, not their recorded failures. No runtime exception is validated by this revision. Meta-World remains required via its separately approved native-lane route; physics constraints are not relaxed.
- Proposal creation installs nothing, downloads no simulator/assets, launches no GPU job, changes no existing proposal and performs no hardware/system operation.
