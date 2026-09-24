## Why

NexuML already has a clean typed pipeline, TensorDict-based execution, Lightning training, portable configuration, export, tracking, and Ray placement. What it cannot express today is interactive learning: `ScenarioSpec.training` assumes dataset batches, supervised loss keys, epochs, and the current `NexuLightningModule`. Robotics and automation training instead needs an environment loop, policies, trajectories/replay, actor/critic losses, and frame-based stopping.

The old PRISMA codebase already demonstrated that TorchRL can live cleanly inside a Lightning `LightningModule` using manual optimization. Reusing that idea is preferable to creating a second top-level trainer in NexuML: Lightning can remain the single training engine while TorchRL provides the RL-specific environment, collector, replay, objective, and policy primitives.

This also creates the right boundary for later NexuFL support. The deployable NexuML pipeline remains the policy/model artifact, while trajectories, environments, replay buffers, and training-only critic state stay local unless a future federated-RL contract explicitly says otherwise.

## What Changes

- Extend `ScenarioSpec.training` to accept either the existing `TrainingSpec` or a new `ReinforcementLearningSpec` without renaming or breaking the existing supervised API.
- Keep `lightning.Trainer` as the only NexuML trainer. Add a dedicated `NexuRLLightningModule` that uses `automatic_optimization = False`, Lightning-owned optimizers, and TorchRL building blocks.
- Add typed `EnvironmentDefinition` and `RLAlgorithmDefinition` component roles using the same immutable definition -> runtime materialization pattern and discovery/serialization registry as layers, data sources, evaluators, and loader backends.
- Add a small compile-context seam so a pipeline can be compiled from environment observation shapes instead of assuming `DataSpec`. The ordinary `compile(scenario)` supervised path remains intact.
- Treat `scenario.pipeline` as the deployable policy. Training-only critic/Q networks belong to the chosen RL algorithm runtime/config and SHALL NOT silently become part of the exported deployable policy.
- Drive RL by environment frames and collector batches rather than pretending RL epochs are dataset epochs. The first implementation is finite-frame training.
- Use TorchRL's current `Collector` front door for rollouts. Collection topology is an RL concern; NexuML execution placement remains a separate concern.
- Add a `nexuml[rl]` optional dependency group for TorchRL and Gymnasium. ROS 2/`rclpy` SHALL NOT become a mandatory PyPI dependency.
- Add initial library components for a Gymnasium environment and PPO, plus a small Pendulum scenario that is suitable for CPU smoke tests and documentation.
- Define the ROS 2 extension boundary and include an example custom library environment, but do not make NexuML own robot drivers, hard-real-time control, safety logic, or a generic reward function.
- Preserve current train-package export semantics for the deployable policy and record RL provenance separately enough that downstream systems can distinguish an RL-trained policy. Federated interactive RL inside NexuFL is explicitly a follow-up; this change only keeps the artifact and runtime boundaries compatible with it.

## Capabilities

### New Capabilities

- `reinforcement-learning-training`: Defines RL scenario configuration, the Lightning/TorchRL training lifecycle, frame-based collection, optimization, checkpointing, and evaluation semantics.
- `reinforcement-learning-components`: Defines typed environment and RL-algorithm components, compile context, discovery, persistence, and runtime ownership.
- `reinforcement-learning-library`: Defines the initial Gymnasium + PPO library implementation, Pendulum scenario, extension examples, and optional dependency boundary.
- `reinforcement-learning-export`: Defines which RL-trained state is the portable deployable policy, which state is training-only, and how exports remain usable by later NexuFL integration.

### Modified Capabilities

None in the first change. Existing supervised training, data loading, export, Ray execution, and NexuFL package-loading contracts must continue to pass unchanged.

## Impact

- Core types: `src/nexuml/core/types.py`.
- Component definitions/discovery: `src/nexuml/core/components.py`, `src/nexuml/core/discovery.py`, `src/nexuml/core/registry.py`.
- Compilation: `src/nexuml/core/compiler.py` gains an explicit pipeline compile context rather than reading every shape exclusively from `DataSpec`.
- Training: `src/nexuml/training/lightning.py` keeps the supervised implementation and session; a focused RL module/runtime is added beside it.
- Library: new `library/src/nexuml_library/reinforcement/`, `environments/`, and `scenarios/reinforcement/` modules.
- Packaging: new `rl` optional extra and inclusion in `all`; no hard ROS dependency.
- Tests: focused component/serialization tests plus one tiny end-to-end Pendulum PPO smoke test; no large algorithm benchmark suite.
- Documentation: one RL mental-model/how-to path and a robotics/ROS extension example.
- Downstream NexuFL: no required implementation in this change, but actor/policy export identity must remain stable and RL-only local state must not leak into the ordinary portable model artifact.
