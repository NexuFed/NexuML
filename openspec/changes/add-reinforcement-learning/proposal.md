## Why

NexuML already has a clean typed pipeline, TensorDict-based execution, Lightning training, portable configuration, export, tracking, and Ray placement. What it cannot express today is interactive learning: `ScenarioSpec.training` assumes dataset batches, supervised loss keys, epochs, and the current `NexuLightningModule`.

The immediate requirement is reinforcement learning, but the architectural boundary should be useful beyond PPO. Physical-AI workloads combine simulation, real sensors/actuators, multimodal observations, demonstrations, online interaction, and deployable policies. A foundation that equates a NexuML environment with a TorchRL `EnvBase`, or equates the deployable policy with only the raw neural pipeline, would require another refactor for imitation learning, VLA policies, world models, sim-to-real, and real ROS 2 systems.

This change therefore remains RL-first in implementation while introducing the smallest reusable Physical-AI seams: a framework-neutral interaction contract, a typed environment definition that can be described without owning a live runtime, a complete deployable policy wrapper, and worker-local materialization of stateful environments/collectors.

The old PRISMA codebase already demonstrated that TorchRL can live cleanly inside a Lightning `LightningModule` using manual optimization. Reusing that idea remains preferable to creating a second top-level trainer in NexuML. Lightning stays the training engine; TorchRL provides RL-specific collectors, losses, replay, value estimation, and policy-distribution primitives.

This also creates the correct boundary for later NexuFL support. The federated/deployable model is a complete policy artifact, while trajectories, simulator state, ROS handles, replay buffers, and training-only critics stay local unless a future federated-learning contract explicitly requests them.

## What Changes

- Extend `ScenarioSpec.training` to accept either the existing `TrainingSpec` or a new `ReinforcementLearningSpec` without renaming or breaking the existing supervised API.
- Add an optional scenario-level `InteractionSpec`. RL requires it, but it is deliberately separate from the RL algorithm so later imitation/VLA/world-model workflows can reuse the same environment and policy I/O contract.
- Add a typed `InteractionContract` describing observation and action fields, including shape, dtype, modality, bounds/space semantics, vectorization, and control-period metadata.
- Add typed `EnvironmentDefinition` components with two distinct responsibilities:
  - `describe()` returns the portable interaction contract without creating a long-lived training runtime or connecting to real hardware.
  - `build(context)` creates a mutable environment runtime at the execution site that owns reset/step/close behavior.
- Do not require the core environment runtime to subclass TorchRL `EnvBase`. Add a reinforcement integration adapter that converts the framework-neutral NexuML environment runtime into TorchRL semantics for collectors.
- Add a typed `ActionAdapterDefinition` and `PolicySpec` so `scenario.pipeline` remains the trainable neural graph while a `CompiledPolicy` represents the complete deployable mapping from observation TensorDict to bounded/discrete actions.
- Keep `lightning.Trainer` as the only NexuML trainer. Add a dedicated `NexuRLLightningModule` with manual optimization and a checkpointable algorithm runtime.
- Add typed `RLAlgorithmDefinition` components. Algorithm construction receives the compiled policy and interaction contract, not a live environment, keeping learner state independent of simulator/robot ownership.
- Build live environments and collectors worker-locally after Lightning/Ray placement. Do not serialize active simulators, ROS nodes, sockets, or collector processes through `NexuSession`.
- Add explicit environment/collector teardown and policy-weight synchronization for non-direct collectors so process/Ray rollout workers do not continue sampling with stale learner weights.
- Add a compile-context seam so a pipeline can be compiled from an interaction contract instead of assuming `DataSpec`.
- Drive RL by environment frames and collector batches rather than pretending RL epochs are dataset epochs. The first implementation is finite-frame training.
- Add explicit episode-evaluation configuration: deterministic/stochastic action mode, seed, episode count, max steps, and optional evaluation-environment override.
- Preserve the distinction between deployable policy artifact and resumable training checkpoint. Training-only critic/Q/target state participates in checkpoints but not normal policy export.
- Add a `nexuml[reinforcement]` optional dependency group for the core Python RL stack. Heavy simulator/system integrations remain optional and layered separately.
- Add maintained library environments/policy adapters/PPO plus executable scenarios covering:
  - continuous control;
  - discrete control;
  - visual observations;
  - humanoid control;
  - robot/manipulator control;
  - ROS 2 against simulation;
  - autonomous-driving simulation.
- Implement a maintained ROS 2 reference adapter rather than documentation only. Normal CI uses a fake transport; an opt-in integration path uses ROS 2 + Gazebo/`ros2_control` so the same NexuML-facing adapter can later drive real hardware.
- Keep hard-real-time control, safety interlocks, emergency-stop logic, vendor drivers, and low-level servo loops outside NexuML.
- Keep imitation learning, VLA training, world-model training, and federated interactive RL out of this implementation while ensuring `data` and `interaction` can coexist and the new contracts do not block those follow-up modes.

## Capabilities

### New Capabilities

- `reinforcement-learning-training`: Defines RL scenario configuration, Lightning/TorchRL lifecycle, worker-local collection, frame budgets, algorithm state, policy synchronization, checkpointing, and episodic evaluation.
- `reinforcement-learning-components`: Defines interaction contracts, environment/action-adapter/RL-algorithm components, compile context, runtime ownership, discovery, persistence, vectorization, and optional dependency boundaries.
- `reinforcement-learning-library`: Defines maintained Gymnasium/PPO/ROS 2 integrations, Physical-AI-oriented example scenarios, simulation tiers, and dependency layering.
- `reinforcement-learning-export`: Defines the complete deployable policy artifact, training-only state boundary, provenance, resume semantics, and later NexuFL compatibility.

### Modified Capabilities

None of the existing supervised capabilities change semantically. Existing supervised training, data loading, export, Ray execution, and NexuFL package-loading contracts must continue to pass unchanged.

## Impact

- Core types: `src/nexuml/core/types.py` gains `InteractionSpec`, interaction field/action contracts, `PolicySpec`, `ReinforcementLearningSpec`, collector/evaluation specs, and the additive `ScenarioSpec.interaction/policy` fields.
- Component definitions/discovery: `src/nexuml/core/components.py`, `discovery.py`, and `registry.py` gain environment, action-adapter, and RL-algorithm roles.
- Compilation: `src/nexuml/core/compiler.py` gains an explicit `PipelineCompileContext` derived either from `DataSpec` or `InteractionContract`.
- Policy runtime: a small `CompiledPolicy` wrapper composes `CompiledPipeline` + action adapter + portable interaction contract.
- Reinforcement runtime: new modules adapt NexuML environments/policies to TorchRL, create collectors at the execution site, synchronize policy weights, and own lifecycle cleanup.
- Training: `src/nexuml/training/lightning.py` keeps supervised behavior; a focused RL module/runtime is added beside it while `NexuSession` remains the public lifecycle owner.
- Library: new reinforcement, environment, action-adapter, ROS 2 transport/reference-task, and scenario modules.
- Packaging: new `reinforcement` optional extra; simulator-specific extras/prerequisites stay separate; ROS 2/Isaac/CARLA are not mandatory base dependencies.
- Tests: registry/config/compiler tests; fast Pendulum/CartPole smoke tests; fake-ROS tests; opt-in simulator integration tests; export/resume tests; policy-sync tests; existing supervised regression suite.
- Documentation: Physical-AI/RL mental model, environment contract, simulator/ROS 2 paths, scenario matrix, deployment boundary, and future imitation/VLA/world-model extension notes.
- Downstream NexuFL: no implementation in this change, but learned policy state is separable from local experience/runtime state and the interaction contract is portable enough for future client-side execution.
