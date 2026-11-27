## ADDED Requirements

### Requirement: Simulation and learner placement are independent

Collector configuration SHALL independently express environment, policy inference and rollout storage devices and worker resource requests. Learner placement SHALL remain in existing training/execution settings. Qualified direct single-device and separate-device paths SHALL preserve native batching and avoid unnecessary host copies. Unsupported combinations SHALL fail before resource allocation.

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

For PPO/IPPO/MAPPO, each update SHALL use complete rollouts from one policy version. Rollouts SHALL preserve time and environment axes, actual frame counts, behavior statistics, critic inputs or values, terminal versus reset observations, termination versus truncation, and required masks. Workers SHALL acknowledge new weights before collecting again. The learner SHALL reject stale or incomplete batches and unsupported asynchronous or multi-learner configurations.

#### Scenario: A worker returns data from an outdated rollout policy
- **GIVEN** The learner is assembling a synchronous update
- **WHEN** the corresponding supported workflow is executed
- **THEN** The inconsistent batch is rejected and no partial optimizer update is performed.

### Requirement: Co-located DDP does not transmit rollout storage

For synchronous single-agent PPO, Lightning DDP ranks SHALL own their
simulator, inference, rollout storage, GAE and updates. Ray Train MAY place
ranks but SHALL NOT send their rollouts between them. Unsupported tasks,
geometry and placements SHALL fail explicitly.

#### Scenario: Ranks consume one global PPO batch
- **WHEN** The learners update on equal complete local shards
- **THEN** Frame counts and advantage normalization use the global batch
  geometry, optimizer steps stay in lockstep, and rank zero owns export
  and evaluation.

#### Scenario: The learner resumes with new simulator instances
- **GIVEN** A checkpoint from a supported DDP run with compatible world size,
  task and batch geometry
- **WHEN** The run resumes with a remaining frame budget
- **THEN** Each rank starts a new, independently seeded simulator and restores
  shared weights, optimizer state and global counters, without claiming
  continuation of the physical trajectories.

### Requirement: Collection resources and payloads remain bounded

Remote collection SHALL transfer batched TensorDict experience and versioned policy weights by maintained mechanisms rather than defaulting to per-step network RPC. In-flight batches, object-store memory and cleanup SHALL be bounded; failures SHALL be actionable and cancellation SHALL release owned resources. Native vectorization and dtype SHALL be preserved.

#### Scenario: A remote collector fails or is cancelled
- **GIVEN** Other workers have outstanding rollout batches
- **WHEN** the corresponding supported workflow is executed
- **THEN** The incomplete update is not consumed, owned actors/native resources and pending references are released, and the borrowed cluster remains running.
