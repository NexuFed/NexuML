## ADDED Requirements

### Requirement: Simulation and learner placement are independent

Collector configuration SHALL independently express environment, inference,
rollout storage and worker resources. Existing training settings SHALL own
learner placement. Native PPO SHALL default to co-location without forced
host copies. Unsupported combinations SHALL fail before allocation.

#### Scenario: An Isaac learner uses a second local GPU
- **GIVEN** Explicit compatible simulation and learner devices are configured
- **WHEN** the corresponding supported workflow is executed
- **THEN** Optional separation preserves native batches and transitions, gives each role its selected device, and does not require CPU learning.

### Requirement: Ray GPU collectors use an existing cluster

Ray collection SHALL use existing/native Ray connection and resource scheduling, persistent execution-local environment factories and worker-local inference. GPU reservations and supported resource/label selectors SHALL be explicit, including worker-visible device mapping. It SHALL NOT silently start a CPU-only local Ray cluster or shut down a borrowed cluster. Ray Jobs and Ray Train SHALL retain distinct responsibilities.

#### Scenario: Two GPU collectors and one GPU learner are requested
- **GIVEN** The existing cluster has qualified simulator and learner resources on distinct nodes
- **WHEN** the corresponding supported workflow is executed
- **THEN** Ray schedules the declared roles, workers return batched rollouts, and the learner has no native simulator dependency.

### Requirement: Synchronous collection preserves on-policy semantics

PPO/IPPO/MAPPO SHALL update only on complete rollouts from one policy
version. Workers SHALL acknowledge new weights before collecting again.
Remote collectors SHALL reject unsupported asynchronous or multi-learner
configurations; rank-local DDP is distinct.

#### Scenario: A worker returns data from an outdated rollout policy
- **GIVEN** The learner is assembling a synchronous update
- **WHEN** the corresponding supported workflow is executed
- **THEN** The inconsistent batch is rejected and no partial optimizer update is performed.

#### Scenario: An acknowledged batch is consumed
- **THEN** Native time/env axes, actual frames, behavior statistics, critic data, terminal vs reset state, termination vs truncation and masks are preserved.

### Requirement: Collection resources and payloads remain bounded

Remote collection SHALL transfer batched TensorDict experience and versioned policy weights by maintained mechanisms rather than defaulting to per-step network RPC. In-flight batches, object-store memory and cleanup SHALL be bounded; failures SHALL be actionable and cancellation SHALL release owned resources. Native vectorization and dtype SHALL be preserved.

#### Scenario: A remote collector fails or is cancelled
- **GIVEN** Other workers have outstanding rollout batches
- **WHEN** the corresponding supported workflow is executed
- **THEN** The incomplete update is not consumed, owned actors/native resources and pending references are released, and the borrowed cluster remains running.


### Requirement: Co-located distributed learners retain the native GPU fast path

TorchRL and Lightning SHALL support synchronous single-agent PPO with
rank-local native simulation, inference, rollout storage, GAE and learning.
Ray Train can place the ranks; PyTorch DDP/NCCL SHALL exchange gradients,
not rollouts. Unsupported tasks, geometry and sharding SHALL fail closed.

#### Scenario: Two GPU ranks learn one policy
- **WHEN** qualified native rank-local PPO executes on two assigned GPUs
- **THEN** each rank collects local CUDA rollouts and contributes to one synchronized update, global physical frames are counted once, and one coherent checkpoint/export is produced without forced CPU staging.

#### Scenario: Equal rank-local shards are optimized
- **WHEN** The ranks consume one global PPO batch
- **THEN** Advantage normalization uses global minibatch geometry, optimizer steps stay in lockstep, and rank zero owns export and evaluation. CPU/Gloo parity does not qualify GPU placement.

#### Scenario: The learner resumes with new simulator instances
- **GIVEN** A compatible checkpoint and remaining global frame budget
- **THEN** Fresh independently seeded simulators restore shared weights, optimizer and counters; failed ranks abort and release owned resources.

### Requirement: Memory-driven placement is explicit and measurable

Qualified GPU collectors SHALL support an independent GPU learner and
explicit inference placement through existing device boundaries. DDP
SHALL NOT be treated as a solution for an oversized replica. No silent
CPU offload, lossy observation conversion or asynchronous PPO.

#### Scenario: A requested GPU policy cannot fit with simulation
- **WHEN** the explicit memory/placement profile is evaluated
- **THEN** qualified separate GPU roles are used or the configuration fails with actionable diagnostics; CPU learning or unqualified model conversion is not silently selected.

#### Scenario: A visual or memory-pressure lane is qualified
- **THEN** Report transfer, latency, throughput, and separate model, optimizer, activation, sensor and rollout memory. Sharding, PEFT, precision and freezing require explicit compatible objective, export and local-fit behavior; general VLA claims require real model evidence.

### Requirement: Placement admission and native runtime inputs are bounded

GPU worker/learner/evaluator requests SHALL use explicit existing Ray/Lightning resources, verified worker-visible device identity and bounded complete-topology admission. Prepared immutable compatible runtime profiles SHALL be preferred; actor startup SHALL NOT depend on unbounded installation or cloning a heavy environment. Missing external inputs SHALL block only the affected live qualification; the independent software and tests remain deliverable.

#### Scenario: An existing cluster cannot admit the full topology
- **WHEN** an independent GPU learner or DDP ranks cannot obtain required resources
- **THEN** submission fails or terminates within declared admission limits with the missing role/resource identified and owned resources released; unrelated implementation is not blocked.
