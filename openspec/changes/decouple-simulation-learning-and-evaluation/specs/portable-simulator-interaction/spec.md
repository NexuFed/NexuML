## ADDED Requirements

### Requirement: Native tasks use the existing interaction boundary

Native providers SHALL use the existing serializable EnvironmentDefinition, execution-local EnvironmentRuntime and InteractionContract. Native tools and NexuML SHALL reuse an installed task, its Python configuration/registration and assets without rewriting simulation logic. Providers retain scene, embodiment, physics, task, sensor and reward configuration. Compiling or describing a declared environment SHALL NOT require retaining a simulator or connecting to hardware.

#### Scenario: A project-owned native task is selected
- **GIVEN** A trusted importable native definition and explicit contract are supplied
- **WHEN** the corresponding supported workflow is executed
- **THEN** NexuML compiles the same policy without loading a native simulator on the coordinator, and builds the runtime only at its execution site.

#### Scenario: An installed custom direct environment extends NC1–NC6
- **WHEN** native tools and NexuML run the same independently runnable custom direct task and its referenced assets
- **THEN** NexuML adds only the learning scenario, semantic contract and bindings; both paths run the real task as part of registered-task qualification.
- **AND** NexuML explicitly rejects unsupported versions, native reset behavior or I/O semantics.

### Requirement: Policy compatibility is distinct from runtime provenance

Policy compatibility SHALL check observation/action semantics, shapes excluding runtime batch width, dtype, units, coordinate frames, joint identities and order, normalization, control rate and mode, temporal/reset behavior, and agent groups. Native asset, physics and runtime identities SHALL be recorded as provenance. Provider metadata or prose alone SHALL NOT decide compatibility. Semantic differences SHALL require a validated explicit mapping; matching shapes alone SHALL NOT suffice.

#### Scenario: A policy is evaluated with a different simulator
- **GIVEN** The new provider has a validated compatible semantic contract and different native provenance
- **WHEN** the corresponding supported workflow is executed
- **THEN** Evaluation accepts the compatible policy and records both identities; a changed joint order or control rate without a validated mapping is rejected.

#### Scenario: Local-fit clients use different recipes or environments
- **WHEN** clients check a shared actor for federated local fitting
- **THEN** they validate its common semantic contract and exact shared tensor/parameter identity separately from each client's recipe and environment identity, as specified in `heterogeneous-local-learning-handoff`.

### Requirement: Checkpoint continuation retains stricter identity

Resume SHALL retain existing learner, optimizer, counters, task/asset/physics and algorithm compatibility rules, with explicitly supported rollout geometry and placement changes only. It SHALL construct fresh live resources and SHALL NOT claim restoration of native physical trajectories.

#### Scenario: A checkpoint is resumed on a supported placement
- **GIVEN** The model, algorithm and exact continuation contract are valid
- **WHEN** the corresponding supported workflow is executed
- **THEN** Learning state and remaining frame budget are restored, the environment is freshly built, and changed placement is recorded.


### Requirement: Interaction ownership and capabilities are independent of policy source

Train, evaluate and play SHALL use the same environment definition/runtime and purpose-aware lifecycle. It SHALL accept contracted actions independently of a CompiledPolicy, expose provider-supported operations and reject incompatible purpose, vectorization, reset, pacing or viewer requests. This boundary SHALL allow extension for teleoperation, recording and authorized physical deployment without requiring those implementations.

#### Scenario: A native runtime receives scripted rather than learned actions
- **WHEN** an approved scripted source supplies the same contracted action as the policy
- **THEN** the same native execution and validation path is used, resource ownership remains unchanged, and both the requested and executed actions and the source identity are representable.

#### Scenario: A caller requests simulator controls on physical hardware
- **WHEN** the caller requests simulator reset, pause or accelerated time for a physical runtime
- **THEN** the runtime rejects the request rather than treating the hardware as a simulator.

### Requirement: Timing and standalone inference form a reusable handoff

Interaction contracts SHALL preserve available native clocks and timestamps, sensor validity and freshness, control period and action timing. They SHALL distinguish synchronous fixed-step RL from bounded asynchronous goals and reject unsupported operations without inventing transitions or timing. The existing complete policy export SHALL preserve temporal-state and action parity with only its documented inference dependencies. No generic skill executor is required.

#### Scenario: A deployment consumer loads a qualified policy
- **WHEN** the complete exported actor is evaluated on representative recorded observations in an inference-only installation without a simulator, ROS, learner, collectors, Ray or NexuFL
- **THEN** processor, action and reset/state parity are retained without a learner or live simulator; missing required inputs or unsupported timed operations are rejected.
