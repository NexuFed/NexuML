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

For supported PPO/IPPO/MAPPO topologies, each learner update SHALL consume complete policy-version-consistent rollouts, preserving time/environment axes, actual frame counts, behavior statistics, critic inputs or values, terminal versus reset observations, terminated/truncated signals and required masks. Updated rollout weights SHALL be acknowledged before the next on-policy collection. Stale, incomplete or unsupported asynchronous/multi-learner requests SHALL be rejected rather than silently alter learning.

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
