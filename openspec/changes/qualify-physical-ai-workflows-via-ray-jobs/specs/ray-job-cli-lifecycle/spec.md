# Spec Delta

## Purpose

Run the existing CLI lifecycle through this branch's SDK Jobs behavior ported into `dev`'s registered `ray-jobs-sdk` backend, allowing independent acceptance cases to overlap without consuming the submitter's GPU or losing their outputs. The existing KubeRay `ray-job` remains a distinct backend.

## ADDED Requirements

### Requirement: R1 Acceptance execution stays remote

All application, native, browser and SDK acceptance runs SHALL use registered `ray-jobs-sdk` on `pool_gpu_workers`, never DGX or separate CPU-worker pools. The submitter SHALL perform control-plane work only with CUDA disabled. Missing capacity or prerequisites SHALL NOT trigger local simulation, a local Ray cluster, another pool or local GPU fallback.

#### Scenario: A CPU control workload is selected
- **WHEN** a control case performs CPU-only computation
- **THEN** it still runs in a default one-GPU job on `pool_gpu_workers`
- **AND** it may hide CUDA inside that allocation without selecting another worker pool

#### Scenario: Remote resources are unavailable
- **WHEN** a submitted acceptance case cannot obtain its required worker resources
- **THEN** it remains queued or is reported blocked with a bounded deadline
- **AND** no replacement training, simulator or native graphics process starts on the submitter

#### Scenario: Local static validation is needed
- **WHEN** lint, documentation, specification or hermetic submission tests run locally
- **THEN** they use CPU-only execution with CUDA disabled and do not execute the acceptance workloads

### Requirement: R2 Remote jobs use the public CLI lifecycle

The backend SHALL support ordinary training, checkpoint resume, checkpoint export and model/policy evaluation using the existing public CLI semantics. Remote selection SHALL occur before local model/checkpoint loading, compilation or native allocation. Existing submission-only training behavior and local commands SHALL remain supported.

#### Scenario: A remote policy is evaluated
- **WHEN** evaluation selects `ray_job` and a worker-accessible package path
- **THEN** submission succeeds without reading that package on the submitter
- **AND** a fresh remote process executes the normal inference-only evaluation path

#### Scenario: Training continues from a remote checkpoint
- **WHEN** a remote resume request supplies an explicit execution configuration and worker checkpoint path
- **THEN** the worker restores the existing checkpoint training semantics and remaining budget
- **AND** the submitter neither opens nor implicitly uploads that checkpoint

### Requirement: R3 Resolved requests preserve trust and path ownership

Remote requests SHALL preserve resolved overrides, use quoted argument boundaries and disable recursive job submission. Execution credentials SHALL NOT enter exported scenario metadata. Checkpoints, datasets and output paths SHALL be explicit worker resources; a missing worker input SHALL fail remotely without silently using a different local or remote file.

#### Scenario: Only the submitter has the named input
- **WHEN** a package or checkpoint path is unavailable on the selected worker
- **THEN** the remote command fails with the missing path and command identity
- **AND** no implicit upload, alternate artifact or local execution is attempted

#### Scenario: A request contains spaces and shell metacharacters
- **WHEN** valid paths or scenario values include whitespace, quotes or shell metacharacters
- **THEN** those values retain their literal argument meaning
- **AND** no additional shell command is executed from them

### Requirement: R4 Parallel cases have isolated resource ownership

At most four jobs SHALL be in flight, further limited by qualified capacity; pending/setup jobs SHALL occupy slots. Each job SHALL reserve a whole GPU and use only job-visible devices. Outputs, mutable environments, ports and simulator namespaces SHALL be isolated. Parallel cases SHALL NOT share writable file-based tracking databases or alter another job's resources.

#### Scenario: Two native cases execute concurrently
- **WHEN** capacity permits two independent GPU cases to overlap
- **THEN** each records its own allocation, process identity, ports and output root
- **AND** neither case observes the other's files, control messages, device allocation or tracking state

#### Scenario: Available capacity is smaller than the campaign bound
- **WHEN** further cases cannot obtain a complete requested allocation
- **THEN** they wait or report blocked rather than oversubscribe fractional GPUs or run locally

### Requirement: R5 Dependency setup is private and failure preserving

The SDK backend SHALL default to locked uv setup in a private per-submission venv and support explicit prepared-environment opt-out. Setup SHALL complete before the workload and preserve failure. It SHALL modify only the owned job environment, never the submitter, base worker or another job's interpreter. Only appropriate immutable assets and package caches MAY be shared.

#### Scenario: A prepared remote environment is selected
- **WHEN** `setup_command=None` explicitly opts out of installation
- **THEN** the backend adds no implicit uv settings and the implicit interpreter remains `python`
- **AND** an explicitly configured interpreter/environment retains precedence

#### Scenario: Setup fails before Python starts
- **WHEN** a selected setup command exits unsuccessfully
- **THEN** the workload does not start and the remote job reports setup failure
- **AND** another running job's interpreter and installed packages remain unchanged

### Requirement: R6 Dependent native stages remain sequential

Parallelism SHALL be across independent cases, not conflicting stages of one native runtime. Checkpoint export, resumed training, evaluation and live replay SHALL use the required fresh process boundaries after their prerequisites complete. The acceptance driver SHALL NOT create a second unsupported native context for periodic evaluation.

#### Scenario: An Isaac learning run needs progress recordings
- **WHEN** early, intermediate and final checkpoints are selected for comparison
- **THEN** matching evaluations run sequentially in fresh remote processes after the owning training process releases its native context
- **AND** each recording identifies its actual source checkpoint and physical frame count

### Requirement: R7 Submission and terminal success remain distinct

Submission SHALL return a job identifier without claiming training success. Acceptance monitoring SHALL follow that identifier to a terminal state and retain command exit codes. Cancellation and deadlines SHALL stop only owned jobs and preserve failure or cancellation; retries SHALL require a new explicit attempt identity rather than occur silently.

#### Scenario: A submitted job fails after initialization
- **WHEN** submission succeeded but native construction or a later CLI stage fails
- **THEN** the case is failed, with the failing stage and retained logs
- **AND** its earlier submission identifier is not treated as successful acceptance

### Requirement: R8 Required outputs survive worker cleanup

Every acceptance attempt SHALL declare a unique durable output root and explicit checkpoint, policy-export, tracking and evidence locations. Completion SHALL require retrieval and identity checks after the producing process exits. Setting a log-root variable alone SHALL NOT be represented as redirecting every export. Existing outputs SHALL NOT be overwritten by a new attempt.

#### Scenario: A worker is removed after producing a policy
- **WHEN** its training and evaluation commands have completed
- **THEN** the policy, receipts and finalized selected media remain retrievable from the declared durable store
- **AND** their hashes match the producing attempt rather than files from another run

### Requirement: R9 Workloads have explicit safety budgets

Remote cases SHALL declare wall-time, CPU/GPU, memory and output budgets and monitor free disk with the existing 4 GiB stop floor. Native/display/asset prerequisites SHALL be checked on the selected worker. Exceeding a deadline or resource floor SHALL stop owned work, retain diagnostics and leave unrelated jobs and original datasets untouched.

#### Scenario: A running job reaches its disk floor
- **WHEN** its monitored free disk reaches the recorded minimum
- **THEN** only the owned process group is cancelled and the case reports the resource stop
- **AND** no other run's files, processes or original datasets are deleted to make it pass

### Requirement: R10 SDK behavior uses the registered execution structure

SDK Jobs SHALL register as `ray-jobs-sdk` using `dev`'s `ExecutionBackendDefinition`, `type`/`version`/`params` serialization and shared Python/CLI/API dispatch. The port SHALL preserve this branch's validated SDK defaults and asynchronous submission contract. The closed union and old `execution.kind` schema SHALL NOT remain accepted. KubeRay `ray-job` SHALL remain distinct and unchanged.

#### Scenario: Default and project settings migrate
- **WHEN** the registered SDK definition is constructed or restored
- **THEN** it preserves default `uv sync --locked`, one GPU, worker-local cache/download settings, private `{run_id}` venv/interpreter resolution and explicit override precedence
- **AND** address/TLS, pool tags, dependency extras and durable paths remain project-owned

#### Scenario: Settings or preflight are inspected
- **WHEN** settings, catalog/discovery or preflight are evaluated without launch
- **THEN** no job is submitted, dependencies installed or learner/native runtime allocated

#### Scenario: Default training submission completes
- **WHEN** `submit_ray_job(scenario)` submits through shared dispatch
- **THEN** it returns the asynchronous string job ID, not a training result
- **AND** the worker receives registered local execution without submission credentials and cannot recursively submit

#### Scenario: A scenario is submitted after schema migration
- **WHEN** Python, CLI or API selects the registered SDK backend
- **THEN** shared validation/dispatch reaches the same quoted setup and Ray Jobs SDK transport
- **AND** the remote normal CLI retains this branch's setup, resource, isolation and failure behavior without Ray Train or KubeRay substitution

#### Scenario: A spec is reused for two submissions
- **WHEN** the caller submits the same registered SDK definition twice
- **THEN** each submission resolves a distinct private venv and matching Python command
- **AND** the caller's settings are not mutated and explicit environment overrides remain intact

#### Scenario: Old execution syntax is supplied
- **WHEN** a config or profile uses the previous `execution.kind` representation
- **THEN** loading fails with an actionable registered-schema diagnostic
- **AND** the framework neither accepts a parallel legacy schema nor silently selects local or KubeRay execution
