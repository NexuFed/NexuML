## ADDED Requirements

### Requirement: Native tasks use the existing interaction boundary

Native providers SHALL implement or reuse the existing serializable EnvironmentDefinition, execution-local EnvironmentRuntime and InteractionContract. Native scene, embodiment, physics, task, sensor and reward configuration SHALL remain provider-owned. An independently runnable installed native task and Python configuration/registration, including compatible custom direct environments and referenced assets, SHALL be usable by native tools and NexuML without rewriting their simulation logic; only the learning scenario, semantic contract and bindings are additional. Qualification SHALL extend the existing NC1–NC6 registered-task acceptance with a real custom direct task reused by both paths; unsupported version, native reset or I/O semantics SHALL fail explicitly. Compiling and describing a declared environment SHALL NOT require retaining a simulator or connecting to physical hardware.

#### Scenario: A project-owned native task is selected
- **GIVEN** A trusted importable native definition and explicit contract are supplied
- **WHEN** the corresponding supported workflow is executed
- **THEN** NexuML compiles the same policy without loading a native simulator on the coordinator, and builds the runtime only at its execution site.

### Requirement: Policy compatibility is distinct from runtime provenance

Policy compatibility SHALL check observation/action semantics, shapes excluding runtime batch width, dtype, units, frames, joint identities and order, normalization, control rate and mode, temporal/reset behavior, and agent groups. Native asset, physics and runtime identities SHALL be recorded as provenance. For federated local fitting, a common actor semantic contract and exact shared tensor/parameter identity SHALL be validated separately from client-specific recipe and environment identity, as specified in heterogeneous-local-learning-handoff. Provider metadata or prose alone SHALL NOT decide compatibility. Semantic differences SHALL require a validated explicit mapping; matching shapes alone SHALL NOT suffice.

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


### Requirement: Interaction ownership and capabilities are independent of policy source

The existing environment definition/runtime SHALL expose a minimal purpose-aware lifecycle and provider-supported operations, validate incompatible purpose, vectorization, reset, pacing and viewer requests, and accept contracted actions independently of a CompiledPolicy. Requested/executed action and source identity SHALL remain representable. Train/evaluate/play SHALL reuse the same interaction boundary; teleoperation, recording and authorized physical deployment SHALL have an explicit extensibility boundary but are not implemented by this requirement. Physical hardware SHALL NOT be treated as a resettable, pausable or accelerated simulator.

#### Scenario: A native runtime receives scripted rather than learned actions
- **WHEN** an approved scripted source supplies the same contracted action as the policy
- **THEN** the same native execution and validation path is used and resource ownership remains unchanged.

### Requirement: Timing and standalone inference form a reusable handoff

Interaction contracts SHALL preserve available native clock/timestamps, sensor validity/freshness, control period and action timing, and distinguish synchronous fixed-step RL from asynchronous bounded goals. Unsupported operations SHALL fail explicitly rather than inventing transitions or timing. The existing complete policy export SHALL demonstrate installed inference and temporal-state/action parity independent of simulator, ROS, learner, collectors, Ray and NexuFL, subject to documented inference dependencies. A generic skill executor is not required.

#### Scenario: A deployment consumer loads a qualified policy
- **WHEN** the complete exported actor is evaluated on representative recorded observations in an inference-only installation
- **THEN** processor, action and reset/state parity are retained without a learner or live simulator; missing required inputs or unsupported timed operations are rejected.
