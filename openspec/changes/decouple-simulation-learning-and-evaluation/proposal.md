# Proposal

## Why

The robotics branch already has a portable interaction contract, native Isaac Lab tasks, TorchRL collection, Lightning optimization and an evaluation-only Arena integration. At baseline c745927b505b1ca2402186217949c7d454e8ee45, Isaac requires a single direct collector on the learner's CUDA device; Ray rollout collection starts a local CPU-only Ray runtime; Arena-composed tasks cannot enter normal training. Remote job submission is not distributed simulation. These qualification restrictions prevent researchers from selecting the best simulator and independently scaling simulation, training and evaluation.

## What Changes

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

### Modified Capabilities

None of the published baseline capabilities is replaced. The existing active changes are unarchived; this is the focused follow-up that extends their implementation, not a second queue for their historical work. See design.md for precedence and reconciliation before archival.

## Impact

Core interaction/contract validation, existing collector and evaluation paths, Lightning lifecycle, library-native environment adapters and examples, execution resources, CI containers and documentation. NexuFL, NexuVibe and cluster GitOps changes are outside this change; document their integration prerequisites without editing them. Phase gates require native evidence, not mocked success.
