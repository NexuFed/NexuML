# Spec Delta

## Purpose

Make the Isaac-first Physical AI stack reproducible and automatically available in the existing development container without making robotics mandatory for ordinary NexuML users.

## ADDED Requirements

### Requirement: One shared primary development environment

The configured NVIDIA development container SHALL provide one consumer-owned Python-3.12 environment with a pinned CUDA-13 PyTorch family and the selected NexuML/Physical AI stack. Approved pinned optional native lanes MAY host incompatible Meta-World, JAX or ROS capabilities under physical-ai-dependency-composition; they SHALL NOT replace primary shared-environment acceptance or imply a conflicting all-extras closure.

#### Scenario: A clean configured container is created
- **WHEN** approved container creation and dependency synchronization finish
- **THEN** the same interpreter runs NexuML, TorchRL, Newton, the selected Isaac Sim integration, demonstration learning and world-model references
- **AND** the resolved package identities and authoritative lock digest are recorded

#### Scenario: An optional native lane is required
- **WHEN** a separately pinned Meta-World, JAX or ROS profile is approved
- **THEN** its NexuML-facing policy/data/metrics boundary is tested independently
- **AND** it neither substitutes for primary interpreter acceptance nor becomes a hidden environment

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

NexuML SHALL declare minimally restrictive compatibility requirements and optional robots dependencies outside all. The consuming root uv project SHALL own interpreter, backend/indexes, exact package/source versions, approved overrides and combined lock. Python 3.12/CUDA 13 SHALL be this development target, not a universal package restriction. Container tooling SHALL own system/ROS libraries; base/non-robot workflows SHALL NOT require robotics.

#### Scenario: NexuML is installed without robotics extras
- **WHEN** a user imports NexuML, discovers scenarios or runs a supervised example
- **THEN** no simulator, ROS runtime, model-weight download or graphics initialization is required

#### Scenario: The implementation environment enables the robotics stack
- **WHEN** the development consumer selects its backend and nexuml[all,dev,robots]
- **THEN** its root pyproject.toml and uv.lock select exact versions and overrides for the requested integrations
- **AND** those consumer choices do not become pins or backend-index requirements for downstream NexuML users

### Requirement: Conflicting requirements remain visible

The requested composition SHALL resolve without ignored dependencies or silent omissions. Only documented exceptions under physical-ai-dependency-composition MAY relax bounds, retaining security/ABI validation. Incompatible or campaign-deferred selections SHALL remain explicit. Required unresolved dependencies or unverified behavior SHALL keep full-program acceptance incomplete; independent implementation need not wait for unrelated native or deferred model work.

#### Scenario: Legacy Meta-World conflicts with the selected Newton stack
- **WHEN** one selection requires MuJoCo 3.3.0 and another requires the incompatible 3.12 release family
- **THEN** resolution reports both requirements instead of forcing a downgrade or claiming both capabilities work
- **AND** any proposed version change, exclusive selection or deferral is reviewable rather than automatic

#### Scenario: An exception permits primary resolution
- **WHEN** the complete primary selection resolves using a declared constraint exception
- **THEN** setup retains original constraints, exact exception and lock identities, known checker discrepancies and separate pending behavior tests
- **AND** no native/model-readiness claim follows until the affected validation passes under approved prerequisites

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

### Requirement: Optional legacy benchmarks do not narrow the whole Python distribution

Classic MuJoCo, Meta-World and GPU Playground SHALL be selectable through verified named optional dependency paths, not an always-on robotics dependency union. Python interpreter restrictions of one optional benchmark SHALL NOT reduce core NexuML's declared compatibility solely to accommodate the benchmark. Incompatible native packages SHALL receive explicit profile/lock and negative-resolution proofs before their native acceptance can pass.

#### Scenario: A base-only user has an otherwise supported Python interpreter
- **WHEN** an optional Meta-World release does not support that interpreter
- **THEN** core supervised/import/inference workflows remain available and only the incompatible optional benchmark reports its specific unsupported selection
