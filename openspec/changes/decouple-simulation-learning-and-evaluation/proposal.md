# Proposal

## Why

The robotics branch already has a portable interaction contract, native Isaac Lab tasks, TorchRL collection, Lightning optimization and an evaluation-only Arena integration. At baseline c745927b505b1ca2402186217949c7d454e8ee45, Isaac requires a single direct collector on the learner's CUDA device; Ray rollout collection starts a local CPU-only Ray runtime; Arena-composed tasks cannot enter normal training. Remote job submission is not distributed simulation. These qualification restrictions prevent researchers from selecting the best simulator and independently scaling simulation, training and evaluation.

The strategic goal is not to outperform or replace Isaac Lab as a simulator/trainer. NexuML is the common local learning backend for NexuFL: compatible clients using Isaac Lab/Arena, MuJoCo and physical ROS 2 robots must be able to start from the same versioned global policy, learn locally and return compatible numeric updates for one global model without transferring raw experience. Native training remains a valid entry point.

This change qualifies the simulator-only handoff. The real-robot learning
experiments below are retained as follow-up requirements, not release gates;
the existing ROS 2 adapter and safety checks are unchanged.

## What Changes

- Run an early, matched native Isaac Lab versus NexuML decision gate on a reward-bearing SO-101 task. Compare correctness, learning, wall-clock/GPU/memory performance, code and dependency burden, export and suitability for repeated external-weight local fitting. Reuse or remove redundant abstractions where upstream preserves semantics; do not treat all later stages as unconditional backlog.
- Extend the existing bounded local-fit and manifest contract toward interactive RL and heterogeneous local experience, not a new federation API. Separate shared actor/parameter compatibility from client-specific task, recipe and runtime provenance; qualify numeric update production from a common base using Isaac, MuJoCo and ROS 2 experience. Keep training-only critics, optimizer state and raw data local by default; NexuFL owns rounds, aggregation, transport and promotion. Prioritize a common-objective three-domain learning proof using authorized demonstrations, followed by an explicitly measured mixed-objective experiment (local simulator PPO and real-robot imitation/offline learning). Neither experiment depends on full distributed GPU qualification.
- Qualify one narrow native-first policy import/initialization path from a maintained Isaac Lab learner, with explicit actor architecture, preprocessing, controls and parameter mapping. Do not promise arbitrary checkpoint conversion or import of optimizer state.

- Keep ScenarioSpec, compiler, CompiledPolicy, EnvironmentDefinition/Runtime, TensorDict, TorchRL and Lightning. Native scene, task, sensor, physics and control configuration remains provider-owned.
- Make native Isaac Lab and Arena task construction converge on the same runtime/bindings/lifecycle; add ordinary Arena-composed training while retaining maintained native benchmark evaluation. Qualify reuse of an independently runnable, externally authored native Python task/configuration (including a custom direct environment) and its assets without a parallel NexuML scene schema or task rewrite.
- Add simulator-neutral `nexuml play` and Python playback using the existing policy inference/rollout/runtime lifecycle: continuous and bounded playback, native parallel environments, per-environment reset/state, real-time pacing, selected visible instances, native interactive simulation controls and optional recording. Keep scientific evaluation distinct. Support ordinary local execution and independent remote Ray Jobs playback on qualified workers.
- Expose typed Isaac viewer modes `headless`, `viser`, `webrtc` and `local` through existing scenario configuration and CLI. Reuse native Isaac visualizers/Kit streaming and the existing CI-distributed Isaac image; preserve private, read-only training/evaluation viewing and explicit opt-in. External NetBird, TCP forwarding, X11 and TigerVNC are infrastructure, not NexuML-managed networking or desktop services.
- Make co-located GPU-native simulation, inference, rollout storage and TorchRL/Lightning PPO optimization the primary single-GPU and distributed execution path. Extend the existing Lightning and Ray Train execution strategy to qualified rank-local DDP/NCCL (one native simulator and learner replica per GPU; synchronize gradients, not rollouts). Preserve independently selectable placement as an optional memory/resource/throughput strategy: same-host GPU split and remote GPU collectors with an independent GPU learner, persistent worker-local inference, versioned batched rollouts and explicit resources. CPU learning/inference is explicit and optional, never a condition for GPU collection. A replica that cannot fit on a GPU needs an independently qualified inference/model partition or memory-efficient strategy, not merely DDP or a remote learner. Keep TorchRL; do not add a competing learner or NCCL protocol.
- Execute evaluation on immutable policy snapshots in an independent process/worker, including qualified native Isaac evaluation during fit; keep in-process native restrictions. Distributed live viewing defaults off; optionally select exactly one read-only actual-training rank (Viser preferred). Native Kit/WebRTC is independent play/authoring first, using a qualified prepared endpoint or display; it must never be a prerequisite for training. Preserve local Kit/TigerVNC evidence; Windows X11 and external WebRTC remain separately qualified optional profiles.
- Separate policy semantic compatibility, exact training-resumption identity and asset/runtime provenance. Require explicit mappings rather than treating same-shaped tensors as portable.
- Qualify a native MuJoCo GPU/Warp or maintained Playground path as a second provider. Retain Gymnasium MuJoCo. Plan Gazebo, CARLA and ROS 2 expansion behind the same interaction boundary, without claiming all upstream tasks work.
- Extend existing CI/image/release mechanisms with immutable prepared simulation/learner/interactive profiles and verified worker dependency admission. Probe kit-less native runtime where compatible; keep full Kit for required sensors/authoring/play. Profile matched native PPO at realistic native batch sizes and a memory-pressure/visual-policy workload; small state MLP and CPU learner evidence do not qualify VLA performance. Require explicit memory/transfer costs, no silent CPU/offload/fallback. Keep existing safety and semantic gates, but external GPU/data/display/operator prerequisites block only their own acceptance lanes; implement and test all independent work without waiting for user intervention. See execution-plan.md. No new trainer, simulator DSL, RPC, mandatory service or framework.

- Freeze a small reusable interaction handoff for the follow-up `integrate-robotics-stack-and-teleoperation`: purpose-aware lifecycle, action-source-independent execution, provider-declared supported operations, observation/command timing and simulator-free inference. Implement only the seams needed by current training/evaluation/playback and a small non-policy action-source fixture; actual teleoperation, recording orchestration, DDS/ROS skills and robot deployment belong to the follow-up.

## Capabilities

### New Capabilities

- `portable-simulator-interaction`: simulator-neutral interaction and physical semantics, provider-owned native configuration, compatibility versus provenance.
- `arena-task-learning`: shared native Isaac/Arena construction, lifecycle, ordinary learning and native suite integration.
- `independent-rollout-placement`: distinct devices and worker resources, GPU-native direct and distributed synchronous TorchRL collection.
- `snapshot-policy-evaluation`: independent evaluation placement, immutable versioned snapshots, native results and bounded concurrency.
- `cross-simulator-qualification`: second native provider and staged system-level backends, runtime/image compatibility and measured evidence.
- `native-learning-comparison`: measured native Isaac Lab baseline, decision gate and narrow native-first interoperability.
- `heterogeneous-local-learning-handoff`: common global-policy identity, bounded compatible local fitting and numeric update receipts across simulation and authorized real-world experience.
- `policy-playback`: shared inference-only continuous/bounded native vectorized playback, interactive simulation controls, pacing, recording and independent local/remote execution.
- `native-viewer-selection`: typed viewer modes, provider-specific native launch and capability validation, private connectivity, purpose-specific control isolation and actual display/stream acceptance.

### Modified Capabilities

None of the published baseline capabilities is replaced. The existing active changes are unarchived; this is the focused follow-up that extends their implementation, not a second queue for their historical work. See design.md for precedence and reconciliation before archival.

## Impact

Core interaction/contract validation, existing collector and evaluation/playback paths, Lightning lifecycle, CLI/SDK, library-native environment adapters and examples, execution resources, CI containers and documentation. Extend the existing NexuML local-learning SDK and document the consumer contract for NexuFL. NexuFL aggregation, protocol, deployment and integration changes, NexuVibe and cluster GitOps changes are outside this change; document their prerequisites without editing them. Phase gates require native evidence, not mocked success; a local deterministic aggregation fixture is not a deployed NexuFL round or proof of real-robot safety.
