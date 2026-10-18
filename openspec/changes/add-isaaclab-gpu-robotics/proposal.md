# Proposal

## Why

Phases 0–1 verify NexuML's CPU Gymnasium workflows, but not GPU-native robotics simulation, same-step auto-reset, or Isaac's process-owned simulator lifecycle. Phase 2 must prove these boundaries without replacing the existing session, policy, PPO, configuration, export, or logging stack, and must include SO-101 as a required later reference.

## What Changes

- Establish an isolated, reproducible Isaac Lab **v3.0.0-EA** / Isaac Sim **6.1.0** / Python **3.12** / PyTorch **2.11.0+cu128** Linux x86_64 integration profile. Pin Isaac's exact source revision and resolved dependencies, validate native operation first, and keep the ordinary project environment unchanged.
- Add a library-owned Isaac environment definition/runtime with simulator-free portable description and execution-local ownership of launch, native batched GPU tensors, reset, and shutdown. Select only `isaacsim_physx` initially.
- Integrate through the existing direct TorchRL collector, preferring the maintained `IsaacLabWrapper` after proving its terminal-observation and auto-reset semantics. Preserve correct PPO time-limit bootstrapping, actual frame budgets, device placement, and policy synchronization without double-vectorizing.
- Deliver state-policy references in order: **Isaac Cartpole → Franka Reach → SO-101 Joint Reach**. Isaac Cartpole is continuous-force control, not categorical Gymnasium CartPole.
- Reuse the pinned upstream SO-101 asset for a small reward-bearing five-arm-joint reaching task with a fixed gripper. Its upstream cube-stack reference has no reward definition and is not claimed as an existing PPO task. SO-101 requires native execution, real updates, resume, export/reload, and replay, not documentation or fake-only acceptance.
- Support fresh-process exported-policy replay through the shared evaluator and native logging, including bounded GIF/MP4/both capture in that process. Reject unsupported in-fit Isaac evaluation before allocation rather than resetting training or opening a second simulator context.
- Document prerequisites, approved asset access, action/control semantics, runnable native/NexuML commands, artifacts, and limitations. Keep simulator/GPU tests opt-in and preserve supervised, Gymnasium, media, packaging, and inference-import boundaries.

## Capabilities

### New Capabilities

- `isaaclab-runtime`: Isolated compatibility profile, portable provider description, native batched simulation, and execution-local lifecycle/diagnostics.
- `isaaclab-rollout-collection`: Single-GPU direct collection, device/batch mapping, same-step reset and terminal-observation correctness, PPO/frame/resume integration.
- `isaaclab-reference-workflows`: Native and NexuML Cartpole/Franka state-policy round trips, documented controls, provenance, and test tiers.
- `so101-simulation-reference`: Required later SO-101 joint-reaching task using the maintained asset, explicit embodiment semantics, and complete simulation-only acceptance.
- `isaaclab-policy-replay`: Fresh-process Isaac policy evaluation, native results/media, portable inference separation, and clear unsupported in-fit evaluation behavior.

### Modified Capabilities

None of the published capabilities change requirements. The existing CLI override, logs-root, export/manifest, and supervised contracts remain authoritative; this adds Isaac-specific behavior beside the two unarchived RL changes rather than rewriting their requirements or completion claims.

## Impact

- Library: new Isaac provider, SO-101 task configuration, and three scenario factories; lazy imports prevent optional simulator discovery from initializing Isaac.
- Reinforcement/session: a narrow provider-to-TorchRL bridge plus lifecycle/preflight integration in existing modules; no second trainer, collector engine, or general plugin hierarchy.
- Evaluation/export: reuse existing complete-policy loading, CLI overrides, evaluator, media helpers, logger instances, staging, and provenance. Simulator/learner handles stay outside inference packages and checkpoints.
- Tooling: isolated environment/container configuration and version evidence, focused fake/CPU tests, explicitly opted-in GPU/simulator tests, and Phase 2 documentation/reporting.
- No Isaac installation, assets, EULA acceptance, system/driver change, or implementation is authorized by generating this proposal. No NexuFL, IL/VLA, ROS/hardware, multi-GPU/Ray, Newton/OV backend rollout, convergence target, or unrelated baseline repair is included.
