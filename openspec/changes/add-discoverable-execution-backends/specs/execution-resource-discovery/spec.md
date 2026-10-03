# Execution Resource Discovery

## Purpose

Expose trustworthy execution targets and resource snapshots from NexuML so users can select a compatible destination without confusing host capacity, workload eligibility and immediate scheduling availability.

## ADDED Requirements

### Requirement: Scoped read-only discovery
NexuML SHALL discover local resources, known/configured Ray endpoints and kubeconfig context identities through backend-owned Python operations. Network probing SHALL be limited to the explicitly selected context/namespace or known Ray endpoint, with bounded time and results. Discovery SHALL NOT scan arbitrary networks, change the user's current kubeconfig context, create workloads, reserve resources or install operators.

#### Scenario: Run dialog opens
- **WHEN** execution discovery is requested for the selected target
- **THEN** bounded read-only probes return discovered data without creating a Ray runtime, cluster or training job

#### Scenario: Multiple Kubernetes contexts exist
- **WHEN** kubeconfig contains several contexts
- **THEN** their identities can be listed locally and remote queries use the selected context explicitly without switching the global current context or probing every cluster

#### Scenario: Target probe times out
- **WHEN** one target fails to answer within the discovery deadline
- **THEN** that target receives a timeout diagnostic and independent successful discovery remains available

### Requirement: Availability has distinct dimensions
Discovery SHALL distinguish backend definition/dependency support, target reachability, resource API support, permissions, scenario compatibility and capacity visibility. A registered custom resource API SHALL NOT by itself prove controller health. Missing permissions to inspect nodes SHALL NOT be reported as zero nodes or missing submission permissions.

#### Scenario: Submission allowed but node visibility denied
- **WHEN** a user can create the selected workload but cannot list cluster nodes
- **THEN** discovery reports usable submission access with unknown node capacity instead of declaring the cluster empty or disabling execution solely for missing node visibility

#### Scenario: Operator support cannot be verified
- **WHEN** the workload API is served but controller health is not observable
- **THEN** discovery identifies API support and unknown controller health without claiming the controller is healthy

### Requirement: Effective local resources
Local discovery SHALL report usable CPU/memory limits and visible supported accelerators where observable, accounting for process affinity and container limits rather than treating host inventory as automatically usable. Accelerator type and memory SHALL be included when available. Unobservable limits or device availability SHALL be marked unknown, and discovery SHALL NOT execute training to estimate model memory.

#### Scenario: Local process is container-limited
- **WHEN** host CPU/memory inventory exceeds the effective process or container limit
- **THEN** the summary identifies effective limits separately from host totals

#### Scenario: Accelerator probe is unavailable
- **WHEN** the selected runtime cannot inspect accelerator information
- **THEN** discovery preserves known CPU/memory results and reports the accelerator diagnostic rather than inventing zero or unrestricted GPUs

### Requirement: Ray capacity comes from the selected cluster
RayCluster discovery SHALL expose live Ray nodes, declared scheduling resources and currently available resources through read-only inspection of the selected cluster. Ray worker nodes SHALL be distinguished from underlying Kubernetes nodes. Discovery SHALL NOT implicitly start a local Ray cluster. RayJob templates creating new clusters SHALL show Kubernetes prospective capacity separately from capacity of already-running Ray clusters.

#### Scenario: Ray workers occupy only part of Kubernetes
- **WHEN** a Kubernetes cluster has eight nodes but the selected Ray cluster has three live Ray nodes
- **THEN** Ray execution capacity identifies those three Ray nodes without presenting all eight Kubernetes nodes as active Ray capacity

#### Scenario: RayJob will create its own cluster
- **WHEN** a selected RayJob template has no running target Ray cluster
- **THEN** discovery reports template requests and prospective Kubernetes capacity without fabricating live Ray workers

### Requirement: Kubernetes capacity and workload eligibility
Kubernetes discovery SHALL distinguish node inventory, allocatable resources, workload-matching nodes and estimated unallocated resources based on effective native pod requests rather than runtime utilization alone. Eligibility SHALL account for observable readiness, unschedulable nodes, resource fit, selectors/required affinity, taints/tolerations and namespace quota. Capacity estimates SHALL include required head, master, submitter and worker roles where applicable. GPU resource units and known device types SHALL remain distinguishable; shared GPU units SHALL NOT be mislabeled as dedicated physical devices.

#### Scenario: Only some nodes match
- **WHEN** eight visible nodes include three matching the selected GPU/resource request and template constraints
- **THEN** discovery reports eight visible nodes and three matching nodes as different values

#### Scenario: Aggregate capacity cannot fit a worker
- **WHEN** resources are sufficient in aggregate but no eligible node can fit one worker's request
- **THEN** discovery reports the per-node fit limitation rather than declaring the workload immediately schedulable

#### Scenario: Namespace quota limits a large cluster
- **WHEN** cluster capacity exceeds the selected namespace's remaining quota
- **THEN** discovery exposes the quota constraint separately and preflight reports a known violating request

#### Scenario: Device type or advanced scheduling information is absent
- **WHEN** a device's model, volume feasibility, gang scheduling or another relevant constraint cannot be determined
- **THEN** the estimate identifies the unresolved constraint and does not claim complete scheduler feasibility

### Requirement: Resource snapshots are advisory and fresh
Snapshots SHALL carry source, observation time, scope and completeness/diagnostic information. Observed zero SHALL be distinct from unknown. Total/allocatable, estimated unallocated, runtime utilization and autoscaler/template maxima SHALL remain separately labeled. Neither discovery nor preflight SHALL guarantee that advisory capacity is reserved or that a workload will start immediately.

#### Scenario: Target has zero available GPUs
- **WHEN** an authoritative snapshot reports no currently unallocated GPUs
- **THEN** discovery represents a known zero rather than unknown and permits a native pending outcome where policy allows submission

#### Scenario: Snapshot becomes stale
- **WHEN** an earlier snapshot is displayed after its freshness window or a failed refresh
- **THEN** consumers can identify its age and stale/partial state and launch rechecks authoritative constraints rather than treating it as a reservation

### Requirement: Partial results do not erase known information
Discovery SHALL retain successful backend/target results when other dependencies, credentials, APIs or probes fail. Restricted or unsupported probes SHALL provide explicit diagnostics. Enumeration exceeding result bounds SHALL identify truncation rather than report an incomplete total as exact.

#### Scenario: Node listing succeeds but pod accounting fails
- **WHEN** allocatable node capacity is visible but pod requests cannot be read
- **THEN** allocatable capacity remains available while currently unallocated estimates are unknown with the pod-accounting diagnostic

#### Scenario: Result set is truncated
- **WHEN** discovery reaches its result bound before enumeration completes
- **THEN** counts and estimates identify their partial scope instead of claiming a complete cluster total

### Requirement: Credentials and explicit launch boundaries
Cluster credentials SHALL remain in the selected Python runtime's normal credential mechanisms and SHALL NOT be returned in browser-facing schemas, discovery results, saved configurations, observations or logs. Discovery inputs SHALL reference authorized contexts/endpoints/templates rather than grant arbitrary remote access or filesystem reads. Ordinary catalog reads SHALL NOT execute kubeconfig authentication helpers; target probes requiring such helpers SHALL occur only in the explicitly selected trusted context. Resource creation SHALL require a separate confirmed launch.

#### Scenario: Browser inspects a Kubernetes target
- **WHEN** discovery returns context, namespace and capacity information
- **THEN** the browser receives safe identifiers and diagnostics but no kubeconfig, token, certificate, secret contents or credential-bearing URL

#### Scenario: Discovery input requests an unauthorized file
- **WHEN** an API caller supplies a template or configuration path outside its authorized boundary
- **THEN** the request is rejected without reading the file or probing a target
