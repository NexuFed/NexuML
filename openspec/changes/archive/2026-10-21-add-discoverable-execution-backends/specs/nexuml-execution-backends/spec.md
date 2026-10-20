# NexuML Execution Backends

## Purpose

Let the selected NexuML Python installation define execution modes and their behavior, with the same configuration, discovery and launch semantics in Python, CLI and Studio.

## ADDED Requirements

### Requirement: Installation-owned execution catalog
NexuML SHALL expose registered execution backend identities, versions, configuration schemas, provenance, dependency diagnostics and supported capabilities without requiring the local API package. CLI and API consumers SHALL use that catalog rather than maintain independent backend lists. Installing a valid backend library SHALL make its backend discoverable without adding API or Studio provider-specific code.

#### Scenario: Python installation has no API extra
- **WHEN** a user lists execution backends from Python or CLI without FastAPI installed
- **THEN** the catalog succeeds and local execution remains usable without importing optional cluster dependencies

#### Scenario: Installed library contributes another backend
- **WHEN** the selected installation discovers a valid additional execution backend definition
- **THEN** Python, CLI and API catalog reads expose its identity, actual schema and capabilities without an API-specific registration step

#### Scenario: Optional runtime is missing
- **WHEN** an installation contains a backend definition but lacks its optional execution dependency
- **THEN** the definition remains inspectable with a missing-dependency diagnostic and cannot be launched through silent fallback

### Requirement: Registered execution configuration
Execution selection SHALL use NexuML's registered `type`, `version` and `params` representation. Loading, saving and frozen launch records SHALL preserve the chosen backend identity and its validated parameters. Unknown backend identities and invalid parameters SHALL produce actionable diagnostics without replacing the selection. The previous closed `execution.kind` representation SHALL NOT be maintained as a second accepted schema.

#### Scenario: Backend configuration round-trip
- **WHEN** a configuration selecting an installed backend is saved and reopened
- **THEN** the same backend definition and parameter values are restored, including fields not rendered by Studio

#### Scenario: Definition or syntax cannot be resolved
- **WHEN** a configuration names an unavailable backend identity/version or uses the previous execution representation
- **THEN** loading fails explicitly without launching work or rewriting the configuration to Local

### Requirement: Shared validation and dispatch
Python, CLI and API SHALL use the same backend-owned preflight and execution behavior. Preflight SHALL validate execution parameters, scenario semantics, target access and launch inputs without allocating training workers or submitting jobs. Launch SHALL revalidate the exact frozen selection and SHALL NOT silently change backend, target or resource request.

#### Scenario: Scenario is incompatible with selected execution
- **WHEN** any interface requests a backend that cannot preserve the scenario's training or evaluation semantics
- **THEN** preflight reports the incompatibility before worker allocation or job creation

#### Scenario: Capacity changes after review
- **WHEN** a target has less immediately available capacity at launch than during review
- **THEN** the backend reports the changed snapshot or native pending state without reducing the user's request or choosing another target

### Requirement: Local and RayCluster retain native training behavior
Local execution SHALL call the canonical NexuML session lifecycle directly. RayCluster execution SHALL connect to an explicitly resolved existing Ray cluster and reuse NexuML's Ray Train implementation. Training strategy SHALL remain in the ordinary training configuration. Existing restrictions on distributed stateful evaluation, post-training fitting, checkpoint resume and exports SHALL remain explicit and SHALL NOT be bypassed by discovery or the new catalog.

#### Scenario: Local scenario is launched
- **WHEN** Local is selected with valid settings
- **THEN** the existing session lifecycle and supported checkpoint/export behavior run without Ray or Kubernetes dependencies

#### Scenario: Existing Ray cluster is selected
- **WHEN** RayCluster is selected with a valid reachable address and worker/resource request
- **THEN** Ray owns placement and the existing NexuML Ray worker lifecycle runs once per worker

#### Scenario: Stateful distributed phase is unsupported
- **WHEN** a RayCluster or RayJob scenario contains unsupported rank-sharded evaluation or post-training fitting
- **THEN** preflight rejects it rather than returning plausible rank-local results

### Requirement: RayJob is a KubeRay submission mode
RayJob SHALL mean the KubeRay RayJob resource, distinct from generic Ray Jobs submission. Execution SHALL submit an explicitly selected infrastructure-owned template to the selected Kubernetes context and namespace. The template SHALL determine whether the job creates a job-scoped Ray cluster or uses an existing cluster; discovery SHALL NOT create either resource. The remote training entrypoint SHALL reuse NexuML's Ray execution and SHALL NOT recursively submit another RayJob.

#### Scenario: Job-scoped Ray cluster is requested
- **WHEN** a reviewed RayJob template defines a new Ray cluster
- **THEN** confirmed launch submits one RayJob and KubeRay owns creation and the template's cleanup policy

#### Scenario: Template selects an existing cluster
- **WHEN** a reviewed RayJob template selects an existing Ray cluster
- **THEN** launch preserves that selection and does not create or delete the shared cluster as a Studio policy

### Requirement: Kubernetes Job and PyTorchJob execution
Kubernetes Job SHALL run the canonical NexuML lifecycle in one training pod using an infrastructure-owned template; unsupported parallel-job topologies SHALL fail preflight. Kubernetes PyTorchJob SHALL use the served PyTorchJob API and an infrastructure-owned distributed launcher template, preserving training strategy and validating rank/world-size behavior. A cluster serving only a different training resource SHALL NOT be presented as PyTorchJob support. Unsupported distributed phases SHALL fail before submission.

#### Scenario: Single Kubernetes training job is selected
- **WHEN** a valid Kubernetes Job template, target and remote configuration handoff are reviewed
- **THEN** launch creates one job whose training entrypoint runs the existing NexuML lifecycle rather than submitting another outer job

#### Scenario: Distributed PyTorch job is selected
- **WHEN** the selected cluster supports PyTorchJob and the reviewed template supplies a compatible distributed launcher
- **THEN** native replica processes execute the same NexuML lifecycle with the validated distributed environment

#### Scenario: Only TrainJob is served
- **WHEN** Kubernetes API discovery reports TrainJob but no supported PyTorchJob resource
- **THEN** PyTorchJob is unavailable with a precise explanation and is not silently substituted with TrainJob

### Requirement: Template and remote input ownership
Job execution SHALL preserve infrastructure-owned images, scheduling policy, storage mounts, service accounts and cleanup settings. Only explicitly supported launch substitutions and reviewed resource bindings SHALL be modified. Preflight SHALL identify how the exact worker configuration, code, dependencies, data and outputs are made available remotely. Launch SHALL NOT assume a local path exists remotely, automatically install packages, upload the whole working directory or rewrite infrastructure policy.

#### Scenario: Launch applies resource overrides
- **WHEN** a user changes a supported worker count or resource field
- **THEN** the reviewed manifest changes only the declared target fields while unrelated template fields remain unchanged

#### Scenario: Required remote configuration handoff is missing
- **WHEN** the selected template cannot receive the frozen worker configuration or references unresolved remote inputs
- **THEN** preflight reports the missing setup and no remote resource is created

### Requirement: Honest execution observation and controls
Backends SHALL report supported status, logs, metrics, cancellation, resume and artifact capabilities separately. Native remote references SHALL include sufficient target and resource identity to inspect the submitted job without confusing a same-named replacement. Submission acceptance SHALL NOT mean training success. Remote terminal status SHALL come from the native workload, not from the local submitter's exit. Unsupported controls SHALL remain unavailable, and losing local observation SHALL NOT falsely cancel remote execution.

#### Scenario: Native job is waiting for resources
- **WHEN** a remote job exists but its training workload has not started
- **THEN** observation identifies its submitted or pending state rather than reporting completed training or fabricated progress

#### Scenario: Remote cancellation is supported
- **WHEN** a user explicitly cancels a job through a backend that supports cancellation
- **THEN** the backend acts only on the verified submitted workload and reports cancellation as confirmed only after native termination is verified

#### Scenario: Submitter exits or observation is lost
- **WHEN** the local submitter finishes while the remote job is active or the job cannot be inspected
- **THEN** the run remains pending/running when verified or becomes ownership-unknown when unverified, never successful or cancelled solely because the submitter exited

#### Scenario: Native identity has been replaced
- **WHEN** a recorded job name now resolves to a different native resource identity
- **THEN** inspection and cancellation refuse to treat the replacement as the original run
