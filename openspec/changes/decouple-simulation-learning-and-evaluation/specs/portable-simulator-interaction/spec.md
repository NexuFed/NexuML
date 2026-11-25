## ADDED Requirements

### Requirement: Native tasks use the existing interaction boundary

Native providers SHALL implement or reuse the existing serializable EnvironmentDefinition, execution-local EnvironmentRuntime and InteractionContract. Native scene, embodiment, physics, task, sensor and reward configuration SHALL remain provider-owned. Compiling and describing a declared environment SHALL NOT require retaining a simulator or connecting to physical hardware.

#### Scenario: A project-owned native task is selected
- **GIVEN** A trusted importable native definition and explicit contract are supplied
- **WHEN** the corresponding supported workflow is executed
- **THEN** NexuML compiles the same policy without loading a native simulator on the coordinator, and builds the runtime only at its execution site.

### Requirement: Policy compatibility is distinct from runtime provenance

Compatibility SHALL validate observation/action semantics, shape excluding runtime batch width, dtype, units, coordinate frames, joints and order, normalization, control rate and mode, temporal/reset behavior and agent groups. Native asset/physics/runtime identities SHALL be retained in provenance. Differences in provider metadata or prose alone SHALL NOT determine compatibility. Actual semantic differences SHALL require an explicit validated mapping or fail closed; identical tensor shapes SHALL NOT suffice.

#### Scenario: A policy is evaluated with a different simulator
- **GIVEN** The new provider has a validated compatible semantic contract and different native provenance
- **WHEN** the corresponding supported workflow is executed
- **THEN** Evaluation accepts the compatible policy and records both identities; a changed joint order or control rate without a validated mapping is rejected.

### Requirement: Checkpoint continuation retains stricter identity

Resume SHALL retain existing learner, optimizer, counters, task/asset/physics and algorithm compatibility rules, with explicitly supported rollout geometry and placement changes only. It SHALL construct fresh live resources and SHALL NOT claim restoration of native physical trajectories.

#### Scenario: A checkpoint is resumed on a supported placement
- **GIVEN** The model, algorithm and exact continuation contract are valid
- **WHEN** the corresponding supported workflow is executed
- **THEN** Learning state and remaining frame budget are restored, the environment is freshly built, and changed placement is recorded.
