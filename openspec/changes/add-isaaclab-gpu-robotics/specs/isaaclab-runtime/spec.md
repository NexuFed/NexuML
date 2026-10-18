# Spec Delta

## Purpose

Provide optional Isaac Lab GPU simulation with reproducible prerequisites, portable configuration, and execution-local resource ownership while retaining ordinary NexuML workflows.

## ADDED Requirements

### Requirement: Isaac uses a reproducible isolated compatibility profile

The integration SHALL identify an exact Isaac Lab 3.0 EA revision and compatible Isaac Sim, Python, PyTorch, CUDA, and resolved dependency versions in a dedicated environment. Its initial supported profile SHALL be Linux x86_64 with Isaac Sim PhysX. Preparation SHALL NOT mutate the ordinary NexuML environment or silently select another simulator release or backend.

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
- **THEN** the three reference definitions and their static contracts remain available without simulator allocation

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
