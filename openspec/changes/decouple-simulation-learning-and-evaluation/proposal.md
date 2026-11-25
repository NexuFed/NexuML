# Proposal

## Why

The robotics branch already has a portable interaction contract, native Isaac Lab tasks, TorchRL collection, Lightning optimization and an evaluation-only Arena integration. At baseline c745927b505b1ca2402186217949c7d454e8ee45, Isaac requires a single direct collector on the learner's CUDA device; Ray rollout collection starts a local CPU-only Ray runtime; Arena-composed tasks cannot enter normal training. Remote job submission is not distributed simulation. These qualification restrictions prevent researchers from selecting the best simulator and independently scaling simulation, training and evaluation.

The strategic goal is not to outperform or replace Isaac Lab as a simulator/trainer. NexuML is the common local learning backend for NexuFL: compatible clients using Isaac Lab/Arena, MuJoCo and physical ROS 2 robots must be able to start from the same versioned global policy, learn locally and return compatible numeric updates for one global model without transferring raw experience. Native training remains a valid entry point.

## What Changes

- Run an early, matched native Isaac Lab versus NexuML decision gate on a reward-bearing SO-101 task. Compare correctness, learning, wall-clock/GPU/memory performance, code and dependency burden, export and suitability for repeated external-weight local fitting. Reuse or remove redundant abstractions where upstream preserves semantics; do not treat all later stages as unconditional backlog.
- Extend the existing bounded local-fit and manifest contract toward interactive RL and heterogeneous local experience, not a new federation API. Separate shared actor/parameter compatibility from client-specific task, recipe and runtime provenance; qualify numeric update production from a common base using Isaac, MuJoCo and ROS 2 experience. Keep training-only critics, optimizer state and raw data local by default; NexuFL owns rounds, aggregation, transport and promotion. Prioritize a common-objective three-domain learning proof using authorized demonstrations, followed by an explicitly measured mixed-objective experiment (local simulator PPO and real-robot imitation/offline learning). Neither experiment depends on full distributed GPU qualification.
- Qualify one narrow native-first policy import/initialization path from a maintained Isaac Lab learner, with explicit actor architecture, preprocessing, controls and parameter mapping. Do not promise arbitrary checkpoint conversion or import of optimizer state.

- Keep ScenarioSpec, compiler, CompiledPolicy, EnvironmentDefinition/Runtime, TensorDict, TorchRL and Lightning. Native scene, task, sensor, physics and control configuration remains provider-owned.
- Make native Isaac Lab and Arena task construction converge on the same runtime/bindings/lifecycle; add ordinary Arena-composed training while retaining maintained native benchmark evaluation.
- Separate environment, rollout collection and learner placement. Qualify direct same-GPU and separate-GPU operation, then synchronous GPU Ray collectors on an existing cluster with worker-local inference, versioned batched rollouts and explicit resources.
- Execute evaluation on immutable policy snapshots in an independent process/worker, with explicit simulator placement, native benchmark results and no disturbance to training.
- Separate policy semantic compatibility, exact training-resumption identity and asset/runtime provenance. Require explicit mappings rather than treating same-shaped tensors as portable.
- Qualify a native MuJoCo GPU/Warp or maintained Playground path as a second provider. Retain Gymnasium MuJoCo. Plan Gazebo, CARLA and ROS 2 expansion behind the same interaction boundary, without claiming all upstream tasks work.
- Extend existing CI/image/release and execution mechanisms; measure actual native throughput, learning, network costs, failure handling and cleanup. No new trainer, simulator language, RPC protocol, mandatory service or framework.

## Capabilities

### New Capabilities

- `portable-simulator-interaction`: simulator-neutral interaction and physical semantics, provider-owned native configuration, compatibility versus provenance.
- `arena-task-learning`: shared native Isaac/Arena construction, lifecycle, ordinary learning and native suite integration.
- `independent-rollout-placement`: distinct devices and worker resources, GPU-native direct and distributed synchronous TorchRL collection.
- `snapshot-policy-evaluation`: independent evaluation placement, immutable versioned snapshots, native results and bounded concurrency.
- `cross-simulator-qualification`: second native provider and staged system-level backends, runtime/image compatibility and measured evidence.
- `native-learning-comparison`: measured native Isaac Lab baseline, decision gate and narrow native-first interoperability.
- `heterogeneous-local-learning-handoff`: common global-policy identity, bounded compatible local fitting and numeric update receipts across simulation and authorized real-world experience.

### Modified Capabilities

None of the published baseline capabilities is replaced. The existing active changes are unarchived; this is the focused follow-up that extends their implementation, not a second queue for their historical work. See design.md for precedence and reconciliation before archival.

## Impact

Core interaction/contract validation, existing collector and evaluation paths, Lightning lifecycle, library-native environment adapters and examples, execution resources, CI containers and documentation. Extend the existing NexuML local-learning SDK and document the consumer contract for NexuFL. NexuFL aggregation, protocol, deployment and integration changes, NexuVibe and cluster GitOps changes are outside this change; document their prerequisites without editing them. Phase gates require native evidence, not mocked success; a local deterministic aggregation fixture is not a deployed NexuFL round or proof of real-robot safety.
