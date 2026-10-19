# Spec Delta

## Purpose

Make the Isaac-first Physical AI stack reproducible and automatically available in the existing development container without making robotics mandatory for ordinary NexuML users.

## ADDED Requirements

### Requirement: One shared primary development environment

The configured NVIDIA development container SHALL provide Python 3.12 and one pinned CUDA-13 PyTorch family in a consumer-owned environment containing NexuML and the compatible selected Physical AI packages. Simulator-specific environments SHALL NOT substitute for this shared-environment acceptance.

#### Scenario: A clean configured container is created
- **WHEN** approved container creation and dependency synchronization finish
- **THEN** the same interpreter runs NexuML, TorchRL, Newton, the selected Isaac Sim integration, demonstration learning and world-model references
- **AND** the resolved package identities and authoritative lock digest are recorded

### Requirement: Installation is automatic and repeatable

Approved container creation SHALL provision the selected Python packages, native prerequisites, pinned sources and minimum reference assets automatically. Repeated setup SHALL verify or reuse matching content without overwriting unrelated work, performing unrequested upgrades or requiring manual package injection.

#### Scenario: Setup is repeated with a matching installation
- **WHEN** the same lock, sources and asset manifests are requested again
- **THEN** setup preserves matching content and completes dependency and import-origin checks
- **AND** it neither starts training nor downloads a complete unrelated asset collection

#### Scenario: Setup cannot complete
- **WHEN** a required download, hash, dependency or native prerequisite check fails
- **THEN** setup exits unsuccessfully with the failing component identified and does not report a ready backend

### Requirement: Python and native dependency ownership are explicit

NexuML SHALL declare optional integration requirements in its packages. The development consumer SHALL own interpreter, backend selection and the combined lock. Container tooling SHALL own system libraries and ROS packages. Base NexuML imports and non-robot workflows SHALL NOT require these optional integrations.

#### Scenario: NexuML is installed without robotics extras
- **WHEN** a user imports NexuML, discovers scenarios or runs a supervised example
- **THEN** no simulator, ROS runtime, model-weight download or graphics initialization is required

### Requirement: Conflicting requirements remain visible

The complete requested composition SHALL be resolved without unsupported overrides, ignored dependencies or silent capability omission. Mutually incompatible optional selections SHALL be explicit. A required unresolved dependency SHALL keep its capability and full-program acceptance incomplete.

#### Scenario: Legacy Meta-World conflicts with the selected Newton stack
- **WHEN** one selection requires MuJoCo 3.3.0 and another requires the incompatible 3.12 release family
- **THEN** resolution reports both requirements instead of forcing a downgrade or claiming both capabilities work
- **AND** any proposed version change, exclusive selection or deferral is reviewable rather than automatic

### Requirement: Canonical migration preserves historical identity

Migration from the verified EA/cu128 stack SHALL preserve historical evidence and identify changed task, physics, asset and dependency versions. New-stack support SHALL require new native and artifact-parity evidence; old execution results SHALL NOT count as new-stack acceptance.

#### Scenario: An older exported policy is selected
- **WHEN** its recorded control or task profile is incompatible with the new stack
- **THEN** execution rejects it or uses an explicitly validated migration without silently reinterpreting its actions

### Requirement: Native readiness is checked independently of Python installation

Readiness SHALL distinguish package resolution, CUDA execution, graphics/rendering, media encoding and ROS availability. Driver, GPU architecture, toolkit, ABI, storage and memory prerequisites SHALL be checked against the selected components, not inferred from a successful package import or GPU management command.

#### Scenario: Compute is available but graphics are absent
- **WHEN** a user requests rendered simulation or WebRTC
- **THEN** the failing graphics or encoder prerequisite is diagnosed before a long run
- **AND** renderer-free capability is reported separately

### Requirement: Recreation and licensing remain operator controlled

Setup SHALL preserve existing staged work and active experiments and SHALL NOT rebuild an active container, replace its environment or change host drivers without approval. Standing EULA approval SHALL NOT imply optional telemetry consent or acceptance of unrelated model and asset licenses.

#### Scenario: Prepared container settings differ from the running instance
- **WHEN** required OS or GPU passthrough settings are not active
- **THEN** setup identifies the required approved recreation and does not pretend a Python install applied those settings
