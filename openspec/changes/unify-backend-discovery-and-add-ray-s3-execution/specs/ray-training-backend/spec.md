# Ray Training Backend

## Purpose

Run the canonical NexuML Lightning session on Ray with the smallest possible adapter.

## ADDED Requirements

### Requirement: Ray execution reuses the canonical NexuML lifecycle
Ray execution SHALL use `TorchTrainer` for worker allocation. Every worker SHALL reconstruct the validated `ScenarioSpec`, create the real `NexuSession`, and call `NexuSession.run()` exactly once. Ray-specific code SHALL NOT duplicate fit, validation, post-train fitting, testing, or evaluation logic.

#### Scenario: Ray worker starts training
- **WHEN** a Ray worker receives a serialized scenario
- **THEN** it validates the scenario, creates a `NexuSession`, injects the Ray-aware Lightning trainer seam, and calls `run()` once

### Requirement: Training strategy has one source of truth
DDP, FSDP, and DeepSpeed SHALL be selected by `TrainingSpec.strategy`; `RayExecutionSpec` SHALL NOT contain a second strategy setting. Ray execution SHALL map those values to the official `RayDDPStrategy`, `RayFSDPStrategy`, or `RayDeepSpeedStrategy` and SHALL use `RayLightningEnvironment`, `RayTrainReportCallback`, and `prepare_trainer`.

#### Scenario: FSDP is selected
- **WHEN** `training.strategy` is `fsdp` and execution is Ray
- **THEN** the Ray Trainer uses `RayFSDPStrategy` while the rest of the NexuML session remains unchanged

#### Scenario: DeepSpeed is selected
- **WHEN** `training.strategy` is `deepspeed` and execution is Ray
- **THEN** the Ray Trainer uses `RayDeepSpeedStrategy` without a NexuML-specific DeepSpeed implementation

### Requirement: Ray configuration describes placement only
Ray execution configuration SHALL describe target, worker count, resources per worker, and Ray run storage. Fixed and elastic worker counts SHALL map directly to Ray `ScalingConfig`; no separate fixed/elastic topology abstraction is introduced.

#### Scenario: Elastic workers are configured
- **WHEN** `workers` is a two-value minimum/maximum range
- **THEN** the values are passed to Ray's elastic `ScalingConfig` representation without a NexuML topology profile

### Requirement: Existing clusters use native Ray APIs
Direct existing-cluster execution SHALL connect through `ray.init(address=...)` and execute `TorchTrainer`. Detached submission SHALL be documented through native Ray Jobs using `ray job submit --working-dir . -- uv run ...`. NexuML SHALL NOT implement its own Ray Jobs bundle, handle, status, logs, cancel, detach, or reattach APIs.

#### Scenario: Detached job is required
- **WHEN** a user wants detached execution on an existing cluster
- **THEN** documentation directs the user to the native Ray Jobs CLI with the repository working directory and `uv run`

### Requirement: Local execution remains direct
Local execution SHALL remain the existing direct `NexuSession.run()` path. The Ray feature SHALL NOT add a local backend wrapper or provider-neutral execution result model.

#### Scenario: No Ray execution is configured
- **WHEN** a scenario uses the default local execution
- **THEN** NexuML runs the existing local session path with no Ray dependency imported

### Requirement: KubeRay remains a template boundary
Temporary KubeRay execution MAY be supported through a small user/infrastructure-owned RayJob template adapter. NexuML SHALL NOT model Kubernetes topology, Kueue policy, worker images, service accounts, tolerations, or node selectors as Ray backend domain objects.

#### Scenario: Temporary Ray cluster is requested
- **WHEN** a KubeRay target is selected
- **THEN** NexuML fills only the required code-working-directory and entrypoint fields of the selected RayJob template and leaves platform scheduling configuration in the template
