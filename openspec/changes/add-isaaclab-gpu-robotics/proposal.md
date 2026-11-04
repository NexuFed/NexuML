# Proposal

## Why

The initial Isaac integration verifies GPU-native robotics and SO-101 joint reaching through NexuML's existing learning stack. The approved extension must make that workflow usable inside the existing development container, audit it against current TorchRL guidance, and add the upstream SO-101 contact-based vial pick-and-place task rather than presenting joint reaching as manipulation.

## What Changes

- Establish an isolated, reproducible Isaac Lab **v3.0.0-EA** / Isaac Sim **6.1.0** / Python **3.12** / PyTorch **2.11.0+cu128** Linux x86_64 integration profile. Pin Isaac's exact source revision and resolved dependencies, validate native operation first, and keep the ordinary project environment unchanged.
- Add a library-owned Isaac environment definition/runtime with simulator-free portable description and execution-local ownership of launch, native batched GPU tensors, reset, and shutdown. Select only `isaacsim_physx` initially.
- Integrate through the existing direct TorchRL collector, preferring the maintained `IsaacLabWrapper` after proving its terminal-observation and auto-reset semantics. Preserve correct PPO time-limit bootstrapping, actual frame budgets, device placement, and policy synchronization without double-vectorizing.
- Deliver state-policy references in order: **Isaac Cartpole → Franka Reach → SO-101 Joint Reach**. Isaac Cartpole is continuous-force control, not categorical Gymnasium CartPole.
- Reuse the pinned upstream SO-101 asset for a small reward-bearing five-arm-joint reaching task with a fixed gripper. Its upstream cube-stack reference has no reward definition and is not claimed as an existing PPO task. SO-101 requires native execution, real updates, resume, export/reload, and replay, not documentation or fake-only acceptance.
- Support fresh-process exported-policy replay through the shared evaluator and native logging, including bounded GIF/MP4/both capture in that process. Reject unsupported in-fit Isaac evaluation before allocation rather than resetting training or opening a second simulator context.
- Document prerequisites, approved asset access, action/control semantics, runnable native/NexuML commands, artifacts, and limitations. Keep simulator/GPU tests opt-in and preserve supervised, Gymnasium, media, packaging, and inference-import boundaries.
- Install and run inside the existing NexuVibe NVIDIA DevContainer using isolated Python 3.12 environments and editable NexuML packages; retain normal development tools and `/env`. No separate development profile or worker container is required. Rebuild/recreation timing remains operator-controlled.
- Add the upstream `IsaacTutorial-Place-Vial-SO101` state task as a fourth reference, proposed registry name `isaac-so101-vial-place-ppo`. Reuse its assets, reward/termination rules and reset dataset; six continuous controls include the gripper. Placement requires physical grasp, transport, insertion and release, not proximity or object attachment.
- Per the user's explicit selection, introduce a separately pinned **IsaacLabTutorial / Isaac Lab 3.0 / Newton MJWarp / cu130** profile. Preserve the tested EA/PhysX profile and its three references; never mix dependencies or silently reinterpret existing policies/checkpoints under the new backend.
- Audit startup/import order, process ownership, native batch/device mapping, wrapper selection, mutable buffers, final/reset observations, numerical GAE, on-policy collection/synchronization and replay against the linked TorchRL guide and actual pinned sources. Record passes, justified version-specific deviations and unresolved blockers separately.
- Preserve the original **42 completed tasks** as historical baseline evidence; all extension tasks remain unchecked until DevContainer-native setup, the audit and the fourth reference's train/resume/export/replay/media round trip are verified.

## Capabilities

### New Capabilities

- `isaaclab-runtime`: Isolated named compatibility profiles, existing-DevContainer installation, portable description, native batches, and profile-aware lifecycle/diagnostics.
- `isaaclab-rollout-collection`: Single-GPU direct collection, reset/GAE/PPO correctness and a version-aware TorchRL integration audit for both profiles.
- `isaaclab-reference-workflows`: Native/NexuML reference round trips, DevContainer-native commands, profile-specific evidence and test tiers.
- `so101-simulation-reference`: Existing joint reaching plus upstream contact-based vial placement, distinct controls/assets/reset distributions and simulation-only acceptance.
- `isaaclab-policy-replay`: Fresh-process replay for all four references, task-specific results/media, portable inference and unsupported in-fit evaluation diagnostics.

### Modified Capabilities

None of the published capabilities change requirements. The existing CLI override, logs-root, export/manifest, and supervised contracts remain authoritative; this adds Isaac-specific behavior beside the two unarchived RL changes rather than rewriting their requirements or completion claims.

## Impact

- Library: Isaac provider and profile-specific task construction/contracts, existing references and a fourth scenario factory; optional task/backend imports stay execution-local.
- Reinforcement/session: a narrow provider-to-TorchRL bridge plus lifecycle/preflight integration in existing modules; no second trainer, collector engine, or general plugin hierarchy.
- Evaluation/export: reuse existing complete-policy loading, CLI overrides, evaluator, media helpers, logger instances, staging, and provenance. Simulator/learner handles stay outside inference packages and checkpoints.
- Tooling: coordinated NexuVibe NVIDIA DevContainer setup/docs, two isolated dependency manifests, portable launch paths, focused CPU tests and opt-in native GPU/media evidence. Preserve already staged NexuVibe work; changes there belong to that repository, not hidden NexuML patches.
- Generating this proposal performs no installation, asset download, rebuild/restart, driver change or implementation. Existing EULA approval stands; optional telemetry remains off. No NexuFL, IL/VLA, camera-policy training/distillation, sim2real preset, ROS/hardware, multi-GPU/Ray, general backend matrix, convergence target or unrelated baseline repair is included. Newton support is limited to the explicitly selected tutorial reference.
