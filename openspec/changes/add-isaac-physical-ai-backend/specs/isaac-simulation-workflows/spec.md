# Spec Delta

## Purpose

Expose the pinned Isaac ecosystem's simulation workflow families through NexuML while retaining task-specific support limits and verified native physical behavior.

## ADDED Requirements

### Requirement: Native task workflows remain selectable and portable

The backend SHALL support native manager-based and direct task definitions using explicit registered task identities and reconstructible parameters. Discovery and configuration restoration SHALL NOT launch simulation; native task creation SHALL occur only at execution and validate the declared policy contract.

#### Scenario: A task is resolved without Isaac installed
- **WHEN** a user discovers and serializes an available reference definition
- **THEN** task identity and its static contract remain available without importing the simulator

### Requirement: Physics and rendering selection are independent

Supported task presets SHALL select Newton, Isaac Sim PhysX or standalone OvPhysX and their supported renderers explicitly. Unsupported task/backend/renderer combinations SHALL fail rather than substitute physics. Incompatible native runtimes SHALL NOT coexist in one process.

#### Scenario: A standalone PhysX task requests a Kit runtime in the same process
- **WHEN** the selected upstream marks that combination incompatible
- **THEN** execution rejects it before allocating a conflicting native context

### Requirement: Native batching preserves transition truth

One simulator instance SHALL expose its native batch without an additional per-environment simulator vectorizer. Reset behavior SHALL preserve final and initial observations, immutable collected values, termination, truncation and independent row identity across no-done, partial-done and all-done transitions.

#### Scenario: A task auto-resets a timed-out row
- **WHEN** the final observation differs from the new episode's initial observation
- **THEN** learning receives the former for bootstrapping and the latter for the next action with exactly one native reset

### Requirement: Sensors are actual task inputs when requested

Declared cameras, depth, range/ray, IMU, contact, wrench and tactile sensors SHALL be constructed through supported upstream APIs and expose their real measurements, units, frames, sampling and validity. An inspection camera or synthetic tensor SHALL NOT establish sensor-policy support.

#### Scenario: A vision policy trains across a native batch
- **WHEN** camera observations are selected with a supported renderer
- **THEN** batched image data reaches the policy with measured layout, timing and preprocessing provenance
- **AND** missing sensor support fails explicitly

### Requirement: Assets and imported scenes retain provenance

Robot and scene imports SHALL record source revision, content hashes, license/access conditions, conversion parameters, collision and actuator settings. Supported URDF, MJCF and USD conversion SHALL reuse maintained tools. A converted asset SHALL NOT count as a trained task or calibrated physical machine.

#### Scenario: A cached robot description changes
- **WHEN** asset content no longer matches the recorded identity
- **THEN** construction fails before policy execution or requires an explicit new version and validation

### Requirement: Randomization and curriculum are reproducible experiment inputs

Task-owned domain randomization, sensor/actuator delay and noise, curriculum and system-identification parameters SHALL be recorded in resolved experiments. Evaluation SHALL use explicit held-out distributions and distinguish curriculum resets from complete-task starts.

#### Scenario: A curriculum starts close to task completion
- **WHEN** the learned policy is evaluated for end-to-end task success
- **THEN** the evaluation uses the frozen complete-task start distribution and does not count curriculum-only success as full completion

### Requirement: Advanced physics and reconstructed scenes have executable references

Deformable, coupled-solver, dexterous contact and reconstructed-scene workflows SHALL each have a pinned supported reference and task-specific validity evidence before being advertised as supported. Native material behavior or scene rendering alone SHALL NOT establish learned control or real-world fidelity.

#### Scenario: A reconstructed scene is used for navigation evaluation
- **WHEN** a policy is evaluated in a real-to-sim reconstruction
- **THEN** the report identifies capture/scene provenance, collision geometry, sensor assumptions and held-out task results

#### Scenario: A soft-body reference has unsupported contact behavior
- **WHEN** its selected solver cannot provide the required stable interaction
- **THEN** the capability remains blocked with the failing native probe retained rather than replacing the object with a rigid proxy silently
