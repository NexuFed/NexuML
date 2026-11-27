## ADDED Requirements

### Requirement: Simulation and learner placement are independent

Collector configuration SHALL independently express environment, inference and rollout storage devices and worker resource requests. Learner placement SHALL remain in existing training/execution settings. Co-located GPU simulation, inference, storage and optimization SHALL be the default native PPO path without forced host copies. Separate GPU placement SHALL be optional for memory or measured bottlenecks, preserve native batching, and never require a CPU learner. Unsupported combinations SHALL fail before resource allocation.

#### Scenario: An Isaac learner uses a second local GPU
- **GIVEN** Explicit compatible simulation and learner devices are configured
- **WHEN** the corresponding supported workflow is executed
- **THEN** Simulation and worker inference occur on the selected simulation GPU, the learner updates on its own GPU, and native frame/transition semantics remain unchanged.

### Requirement: Ray GPU collectors use an existing cluster

Ray collection SHALL use existing/native Ray connection and resource scheduling, persistent execution-local environment factories and worker-local inference. GPU reservations and supported resource/label selectors SHALL be explicit, including worker-visible device mapping. It SHALL NOT silently start a CPU-only local Ray cluster or shut down a borrowed cluster. Ray Jobs and Ray Train SHALL retain distinct responsibilities.

#### Scenario: Two GPU collectors and one GPU learner are requested
- **GIVEN** The existing cluster has qualified simulator and learner resources on distinct nodes
- **WHEN** the corresponding supported workflow is executed
- **THEN** Ray schedules the declared roles, workers return batched rollouts, and the learner has no native simulator dependency.

### Requirement: Synchronous collection preserves on-policy semantics

For PPO/IPPO/MAPPO, each update SHALL use complete rollouts from one policy version. Rollouts SHALL preserve time and environment axes, actual frame counts, behavior statistics, critic inputs or values, terminal versus reset observations, termination versus truncation, and required masks. Workers SHALL acknowledge new weights before collecting again. The remote-collector learner SHALL reject stale or incomplete batches and unsupported asynchronous or multi-learner configurations; co-located rank-local DDP is a distinct qualified strategy.

#### Scenario: A worker returns data from an outdated rollout policy
- **GIVEN** The learner is assembling a synchronous update
- **WHEN** the corresponding supported workflow is executed
- **THEN** The inconsistent batch is rejected and no partial optimizer update is performed.

### Requirement: Collection resources and payloads remain bounded

Remote collection SHALL transfer batched TensorDict experience and versioned policy weights by maintained mechanisms rather than defaulting to per-step network RPC. In-flight batches, object-store memory and cleanup SHALL be bounded; failures SHALL be actionable and cancellation SHALL release owned resources. Native vectorization and dtype SHALL be preserved.

#### Scenario: A remote collector fails or is cancelled
- **GIVEN** Other workers have outstanding rollout batches
- **WHEN** the corresponding supported workflow is executed
- **THEN** The incomplete update is not consumed, owned actors/native resources and pending references are released, and the borrowed cluster remains running.


### Requirement: Co-located distributed learners retain the native GPU fast path

NexuML SHALL extend existing TorchRL/Lightning and Ray Train execution for synchronous single-agent PPO with one native simulator, local inference, CUDA rollout buffer and learner per rank. Supported PyTorch DDP/NCCL SHALL synchronize gradients via existing strategy, not network rollout batches or a new protocol. Global frame budgets, deterministic distinct seeds, equal minibatch/update geometry, normalization/reduction semantics, manual optimization, rank-zero artifact/evaluation ownership, resume and collective failure cleanup SHALL be verified. Unsupported algorithms and model sharding SHALL fail closed. CPU/Gloo is software evidence only; qualify actual CUDA NCCL rank placement separately.

#### Scenario: Two GPU ranks learn one policy
- **WHEN** qualified native rank-local PPO executes on two assigned GPUs
- **THEN** each rank collects local CUDA rollouts and contributes to one synchronized update, global physical frames are counted once, and one coherent checkpoint/export is produced without forced CPU staging.

### Requirement: Memory-driven placement is explicit and measurable

An independent GPU learner SHALL be supported with qualified GPU collectors; simulator/inference separation SHALL be available through a qualified existing device boundary where combined memory is insufficient. Model replication, optimizer state, activations, sensors and rollout memory SHALL be measured separately. DDP SHALL NOT be described as solving an oversized single replica. Model-specific sharding, PEFT, precision or frozen modules require explicit selection and compatible objective/export/local-fit semantics. No silent CPU offload, lossy observations, stale on-policy batches or arbitrary VLA PPO assumption. Transfer/latency, memory and throughput SHALL be reported on visual and memory-pressure references; real VLA claims require actual model evidence.

#### Scenario: A requested GPU policy cannot fit with simulation
- **WHEN** the explicit memory/placement profile is evaluated
- **THEN** qualified separate GPU roles are used or the configuration fails with actionable diagnostics; CPU learning or unqualified model conversion is not silently selected.

### Requirement: Placement admission and native runtime inputs are bounded

GPU worker/learner/evaluator requests SHALL use explicit existing Ray/Lightning resources, verified worker-visible device identity and bounded complete-topology admission. Prepared immutable compatible runtime profiles SHALL be preferred; actor startup SHALL NOT depend on unbounded installation or cloning a heavy environment. Missing external inputs SHALL block only the affected live qualification; the independent software and tests remain deliverable.

#### Scenario: An existing cluster cannot admit the full topology
- **WHEN** an independent GPU learner or DDP ranks cannot obtain required resources
- **THEN** submission fails or terminates within declared admission limits with the missing role/resource identified and owned resources released; unrelated implementation is not blocked.
