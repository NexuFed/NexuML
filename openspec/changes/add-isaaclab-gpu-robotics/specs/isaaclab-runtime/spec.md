# Spec Delta

## Purpose

Provide optional Isaac Lab GPU simulation with reproducible prerequisites, portable configuration, and execution-local resource ownership while retaining ordinary NexuML workflows.

## ADDED Requirements

### Requirement: Isaac uses a reproducible isolated compatibility profile

The integration SHALL identify exact source revisions and resolved Python, simulation, PyTorch and CUDA dependencies per isolated profile. The initial profile SHALL remain Linux x86_64 EA/Isaac Sim PhysX. The upstream vial task SHALL use a separately pinned Newton MJWarp profile. Preparation SHALL NOT mutate the ordinary environment or silently switch profiles.

#### Scenario: The approved candidate profile is prepared
- **WHEN** an operator explicitly approves preparing the integration environment
- **THEN** its manifest records the EA source revision, simulator version, Python/PyTorch/CUDA builds, dependency resolution, and operating-system/container identity
- **AND** native execution is verified before the profile is described as tested

#### Scenario: The installation cannot satisfy the profile
- **WHEN** the resolver or runtime requires an incompatible dependency or backend
- **THEN** preparation reports the conflict without modifying the ordinary environment or substituting a different profile

### Requirement: Prerequisites and downloads require explicit operator action

The integration SHALL diagnose missing runtime, GPU/graphics access, assets, or incompatible prerequisites before a long training run. It SHALL NOT install system packages, change drivers, accept licenses, or download simulators/assets during ordinary discovery or CPU tests. Setup documentation SHALL identify approved asset locations, caching, disk requirements, and network/license needs.

#### Scenario: GPU management visibility does not establish simulator usability
- **WHEN** a GPU is listed but CUDA, Vulkan/graphics access, or simulator startup fails
- **THEN** the run reports the failing prerequisite without claiming hardware compatibility or falling back to CPU simulation

#### Scenario: An asset is unavailable
- **WHEN** an approved task asset cannot be resolved from the configured source/cache
- **THEN** the run identifies the missing asset and setup action without substituting another robot

### Requirement: Provider configuration remains portable without Isaac

Isaac scenarios SHALL support registry discovery, resolved configuration round trips, and policy-contract description without importing or launching simulator libraries. Portable definitions SHALL identify task, source revision, physics selection, observations, actions, and control timing without retaining live runtime handles.

#### Scenario: Discovery runs in a normal CPU environment
- **WHEN** Isaac imports are blocked during scenario discovery and configuration restoration
- **THEN** the original three reference definitions, the vial-placement definition and their static contracts remain available without simulator allocation

#### Scenario: Runtime and declared contract disagree
- **WHEN** an instantiated task differs in shape, dtype, action meaning, joint order, or control period from its declared profile
- **THEN** execution fails before collecting training data rather than silently rewriting the policy contract

### Requirement: One runtime owns one native batch

The provider SHALL materialize one native Isaac batch at the execution site using the requested environment count and device. It SHALL expose distinct environment batch axes without adding them to per-observation/action field shapes. Unsupported execution topology SHALL fail before simulator allocation.

#### Scenario: A batched state task starts
- **WHEN** a supported single-GPU direct run requests N environments
- **THEN** one simulation owns N independent episodes with GPU-resident observations/actions and correctly shaped rewards and done flags
- **AND** no outer environment loop or process vectorizer creates N simulator applications

#### Scenario: An unsupported topology is requested
- **WHEN** Isaac requests process/Ray collection, multiple learner GPUs, asynchronous collection, or mismatched simulator/policy devices
- **THEN** a focused error rejects that combination before any simulator starts

### Requirement: Shutdown releases only owned resources

Runtime and launcher resources SHALL close on completion, initialization failure, training failure, and cancellation. Shutdown SHALL be idempotent, preserve a primary exception when cleanup also fails, and avoid terminating caller-owned resources. Live handles SHALL NOT enter scenario serialization, checkpoints, or policy packages.

#### Scenario: Construction fails after launcher startup
- **WHEN** task construction raises after acquiring simulator resources
- **THEN** all acquired owned resources are released and the original failure remains visible

#### Scenario: Cancellation and cleanup both fail
- **WHEN** cancellation is followed by an encoder, environment, or launcher close error
- **THEN** cancellation remains the primary outcome with cleanup failure recorded and no false successful completion

### Requirement: Accepted fast shutdown follows the whole job

The operator-approved fast-shutdown profile SHALL finish environment cleanup, checkpoints/exports and explicit logger finalization before public application close. It SHALL preserve success, failure, explicit exit and interrupt status without modifying upstream or promising Python `atexit` cleanup.

#### Scenario: Kit requires process-terminating shutdown
- **WHEN** an Isaac job completes or raises after application startup
- **THEN** the normal Python job unwinds and finalizes before public fast close, with an explicit truthful exit status
- **AND** a job outside the owned process scope or using an existing caller-owned application is rejected before native allocation

### Requirement: Installation reuses the existing NVIDIA development container

The supported setup SHALL install and run both profiles inside the existing NexuVibe NVIDIA DevContainer without a required worker container or separate development profile. It SHALL preserve normal tools and `/env`, use isolated Python 3.12 environments and editable NexuML packages, and provide explicit interpreter and launch commands independent of worker-image paths.

#### Scenario: An operator prepares a profile inside the development container
- **WHEN** the documented approved installation is executed
- **THEN** native and NexuML commands use the selected isolated interpreter and editable source checkout
- **AND** dependency checks, source/import identities and resolved manifests are recorded without changing `/env`, the ordinary lock or the other profile

#### Scenario: Inherited workspace settings target the shared environment
- **WHEN** environment or library-path settings would redirect installation or load another profile's CUDA libraries
- **THEN** setup explicitly isolates those settings or fails before mutation rather than installing into `/env`

### Requirement: Container recreation remains operator-controlled

Setup SHALL distinguish prepared configuration from the running container, preserve pre-existing staged work and require operator-approved timing for rebuild/recreation. It SHALL diagnose effective GPU/graphics, driver, resource and cache prerequisites without restarting active experiments. Standing EULA approval SHALL NOT imply optional telemetry consent.

#### Scenario: Prepared graphics configuration is not active
- **WHEN** the current container lacks the selected profile's OS or graphics prerequisites
- **THEN** setup identifies the required recreation and remains blocked until the operator approves its timing
- **AND** CUDA-only Newton execution is not reported as successful RTX/Vulkan validation

### Requirement: Startup follows the selected profile's verified launch contract

Simulator-owning jobs SHALL use the selected pinned profile's supported import/launch order with one owned simulation per process. Isaac Sim startup SHALL satisfy the documented application-before-Torch rule or retain a source-justified, native-tested version-specific exception. Kit-less Newton SHALL NOT allocate an Isaac Sim app. Portable operations and completed-budget resume SHALL allocate none.

#### Scenario: The existing entrypoint conflicts with launch guidance
- **WHEN** the TorchRL audit finds imports preceding the application launch
- **THEN** a tested supported bootstrap or an evidenced pinned-version exception resolves the gate
- **AND** an unexplained mismatch remains incomplete rather than receiving a correctness claim

#### Scenario: A kit-less tutorial job exits or is cancelled
- **WHEN** the owned Newton runtime finishes, fails or receives cancellation
- **THEN** public cleanup releases its resources and finalizes loggers with truthful status without invoking the EA Kit fast-close path

### Requirement: Profile mismatches fail without replacing dependencies or assets

Task configuration and artifacts SHALL identify profile, backend and source/asset/reset-data identities. Incompatible task/profile combinations SHALL fail before simulation or policy actions, not upgrade an environment, change physics or reinterpret old checkpoints. Required pinned asset content SHALL be verified before trusted-format loading, with missing Git LFS data diagnosed explicitly.

#### Scenario: A vial-placement task is selected in the EA environment
- **WHEN** its required tutorial revision, dependencies or assets are absent or incompatible
- **THEN** execution identifies the required isolated profile or missing content without modifying the working EA installation
