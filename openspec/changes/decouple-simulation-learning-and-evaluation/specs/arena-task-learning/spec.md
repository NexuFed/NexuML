## ADDED Requirements

### Requirement: Arena composition supports ordinary learning

Native tools and ordinary NexuML training and playback SHALL use the same trusted, importable Arena scene, embodiment and task without copying them into a NexuML schema. The normal EnvironmentDefinition/build and TorchRL collection path SHALL use the maintained native builder and share compatible Isaac Lab runtime, bindings, observation/action mapping and lifecycle. Plain Isaac Lab SHALL remain usable without Arena.

#### Scenario: An Arena reward-bearing task is selected for training
- **GIVEN** A compatible pinned native environment, task and policy contract are provided
- **WHEN** the corresponding supported workflow is executed
- **THEN** The canonical Lightning learner performs actual optimizer updates from native transitions and exports a policy usable for fresh replay.

### Requirement: Native simulation lifecycle belongs to its execution owner

Each native simulator SHALL be started, stepped and closed by its owning process, with valid launch/import order and cleanup on normal completion, exception and cancellation. Direct execution SHALL retain necessary finalization scope; distributed learners SHALL NOT launch a remote worker's simulator. Conflicting native scopes SHALL be rejected.

#### Scenario: An Isaac or Arena environment is collected remotely
- **GIVEN** A simulation worker is allocated its declared GPU
- **WHEN** the corresponding supported workflow is executed
- **THEN** The worker starts one qualified native scope, preserves native vectorization and closes all owned resources when collection ends.

### Requirement: Arena benchmarks retain suite authority

Native Arena evaluation SHALL use maintained suite composition, condition identities, success criteria, seeds, native recorder and reports. Shared policy bindings/lifecycle SHALL NOT replace benchmark-specific scoring with generic replay. Generic video and benchmark media capabilities SHALL be validated explicitly.

#### Scenario: A trained compatible policy is evaluated on held-out Arena conditions
- **GIVEN** A pinned suite and immutable policy are selected
- **WHEN** the corresponding supported workflow is executed
- **THEN** Native per-condition/per-episode outcomes and failure evidence are returned via existing tracking/artifacts without retraining or fabricating results.
