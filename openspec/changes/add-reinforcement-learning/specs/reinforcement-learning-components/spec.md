## Purpose

Defines the typed interaction, policy, environment, RL-algorithm, and compilation contracts that let NexuML support reinforcement learning now while providing a stable Physical-AI foundation for later imitation, VLA, world-model, simulator, ROS 2, and NexuFL workflows.

## ADDED Requirements

### Requirement: Interaction is a scenario-level concern

`ScenarioSpec` SHALL support an optional `interaction: InteractionSpec | None` field separate from `training`. Existing supervised scenarios SHALL remain valid without specifying interaction.

#### Scenario: Existing supervised scenario is loaded

- **WHEN** a current scenario omits `interaction`
- **THEN** its configuration and training behavior remain unchanged
- **AND** no reinforcement or environment dependency is imported.

#### Scenario: RL scenario is loaded

- **WHEN** `training` is `ReinforcementLearningSpec`
- **THEN** the scenario SHALL contain an `InteractionSpec`
- **AND** the RL algorithm references that interaction rather than owning a second environment definition.

#### Scenario: Future hybrid scenario is authored

- **WHEN** a scenario contains both `data` and `interaction`
- **THEN** the configuration is valid
- **AND** NexuML SHALL NOT assume that interactive and offline data sources are mutually exclusive.

### Requirement: Interaction contract is portable and explicit

NexuML SHALL define a serializable `InteractionContract` containing observation and action field contracts sufficient to compile and validate a policy without retaining a live environment.

Each tensor field contract SHALL support at least:

- shape excluding batch/vectorized environment dimensions;
- dtype;
- modality/semantic category such as vector, image, depth, proprioception, lidar, radar, audio, language/token, or generic;
- optional lower/upper bounds where meaningful;
- optional sampling/control-rate metadata.

The interaction contract SHALL additionally describe action-space semantics (for example continuous, discrete, multi-discrete, or structured), vectorization metadata, and optional control period.

#### Scenario: Multimodal robot contract is resolved

- **WHEN** an environment describes camera, joint-state, and force observations plus bounded joint actions
- **THEN** all fields are represented by stable TensorDict keys and typed metadata
- **AND** pipeline compilation can use the observation fields without connecting to the robot.

#### Scenario: Unsupported field is declared

- **WHEN** a required observation/action leaf cannot be represented by the supported portable contract
- **THEN** resolution fails with a diagnostic naming the field and unsupported type
- **AND** NexuML does not silently substitute a generic tensor shape.

### Requirement: Environment definition separates description from runtime ownership

NexuML SHALL provide an `EnvironmentDefinition` component role with stable kind/name/version identity and two distinct boundaries:

- `describe(...) -> InteractionContract` for portable/static policy I/O description;
- `build(EnvironmentBuildContext) -> EnvironmentRuntime` for mutable interaction runtime creation.

`describe()` SHALL NOT return or retain a live simulator, ROS node, socket, process, or hardware handle. Backends whose third-party API lacks a static schema MAY use a short-lived local descriptor probe that is immediately closed, but description SHALL NOT connect to real hardware or depend on a long-running training runtime.

#### Scenario: Environment definition is persisted

- **WHEN** a scenario containing a registered environment is serialized
- **THEN** its stable component identity and validated parameters are persisted
- **AND** no live environment object or transport handle is serialized.

#### Scenario: Driver resolves a policy contract

- **WHEN** the scenario is compiled before training placement
- **THEN** NexuML obtains an `InteractionContract` from the environment definition
- **AND** does not create the long-lived training environment.

### Requirement: Environment runtime is framework-neutral

The core `EnvironmentRuntime` contract SHALL NOT require subclassing TorchRL `EnvBase`, Gymnasium `Env`, ROS classes, Isaac classes, or CARLA classes.

The runtime SHALL expose equivalent semantics for:

- reset;
- step;
- observation TensorDict;
- action TensorDict;
- reward where the task provides one;
- terminated and truncated signals;
- deterministic close/teardown.

The reinforcement integration SHALL adapt this runtime to TorchRL when constructing collectors.

#### Scenario: Custom simulator runtime is used

- **WHEN** a user implements the NexuML environment runtime protocol without importing TorchRL
- **THEN** the reinforcement adapter can wrap it for TorchRL collection
- **AND** the user's environment implementation remains reusable by future non-RL training modes.

### Requirement: Environment build context models execution-site semantics

`EnvironmentBuildContext` SHALL contain enough execution-local information for simulation and physical systems, including at least:

- purpose (`train` or `eval`);
- seed;
- requested device;
- worker/rank identity where available;
- requested number of vectorized environments.

#### Scenario: Vectorized simulator is built

- **WHEN** a collector requests 128 environments on one worker
- **THEN** the build context communicates the requested vectorization
- **AND** the resulting runtime/adapter reports a contract compatible with that vectorization.

### Requirement: Live interaction runtimes are materialized at the execution site

`NexuSession` SHALL NOT create long-lived environments, ROS nodes, simulator handles, or collector processes before Lightning/Ray placement.

#### Scenario: Direct local collector is used

- **WHEN** the Lightning worker starts fitting
- **THEN** the environment runtime is materialized in that worker
- **AND** is closed during teardown even when training raises an exception.

#### Scenario: Process or Ray collector is used

- **WHEN** collection runs in remote processes/actors
- **THEN** each collector worker materializes its own environment from the immutable definition/build context
- **AND** no pre-created live environment is pickled from the driver.

### Requirement: Action adaptation is a typed deployable component

NexuML SHALL provide an `ActionAdapterDefinition` component role and optional scenario-level `PolicySpec`.

The trainable `scenario.pipeline` remains the neural TensorDict graph. A `CompiledPolicy` SHALL compose:

- the compiled pipeline;
- the action-adapter runtime;
- the portable interaction contract/policy I/O metadata.

The action adapter SHALL be responsible for converting model outputs to valid environment actions, including distribution sampling/mode selection, bounded transforms, or direct action-key routing.

#### Scenario: Continuous PPO policy is compiled

- **WHEN** the neural pipeline emits location and scale parameters
- **THEN** a Gaussian action adapter converts them to valid bounded actions
- **AND** deterministic inference can select the configured mean/mode behavior without constructing the PPO learner.

#### Scenario: Direct-action future VLA policy is compiled

- **WHEN** a model already emits an action/action-chunk tensor
- **THEN** a direct action adapter can expose that output without introducing an RL-specific wrapper.

### Requirement: RL algorithm is a typed component independent of live environments

NexuML SHALL provide an `RLAlgorithmDefinition` component role with stable kind/name/version identity and explicit runtime materialization.

Algorithm construction SHALL receive the compiled policy and `InteractionContract`/action-space information, not ownership of the live environment runtime.

#### Scenario: PPO runtime is materialized

- **WHEN** PPO is selected
- **THEN** it may create critic/value networks, loss modules, value estimators, replay/minibatch state, and target state
- **AND** it does not own or serialize the simulator/ROS runtime.

### Requirement: RL algorithm runtime is checkpointable

The mutable RL algorithm runtime SHALL be registered as part of the Lightning module state. It SHALL either be an `nn.Module` or implement an equivalent explicit checkpoint state protocol accepted by NexuML.

#### Scenario: PPO critic exists

- **WHEN** a PPO run is checkpointed
- **THEN** critic/value parameters required for local resume participate in checkpoint state
- **AND** they are not lost merely because the critic is not part of `scenario.pipeline`.

### Requirement: Existing component registry remains authoritative

Environment, action-adapter, and RL-algorithm identities SHALL use the current component registry/discovery system rather than separate subsystem registries.

#### Scenario: Registry lists Physical-AI components

- **WHEN** component discovery has loaded an installed library
- **THEN** `environment`, `action_adapter`, and `rl_algorithm` entries appear in deterministic common-registry listings
- **AND** conflicts follow the existing stable identity rules.

### Requirement: Pipeline compilation accepts explicit interaction context

NexuML SHALL support compiling a `PipelineSpec` from explicit interaction input metadata independent of `DataSpec`.

`PipelineCompileContext` SHALL include observation input sizes and preserve enough contract metadata for layers that need dtype/modality/class-like metadata later.

#### Scenario: Supervised pipeline compiles

- **WHEN** `compile(scenario)` is called for an existing supervised scenario
- **THEN** NexuML derives the compile context from the existing data configuration
- **AND** behavior remains equivalent to the current compiler.

#### Scenario: Interactive policy compiles

- **WHEN** an interaction contract exposes tensor observation fields
- **THEN** NexuML derives pipeline input keys/shapes from the contract
- **AND** compiles `scenario.pipeline` without constructing a fake dataset.

### Requirement: Deployable pipeline forward contract remains compatible

RL integration SHALL not change the public `CompiledPipeline.forward(x, y) -> (x_out, y_out)` contract.

#### Scenario: Compiled policy performs inference

- **WHEN** a policy receives an observation TensorDict
- **THEN** `CompiledPolicy` invokes the existing compiled pipeline
- **AND** the action adapter produces a valid action TensorDict from pipeline outputs.

### Requirement: Reinforcement dependencies are optional

Importing NexuML and using existing supervised scenarios SHALL NOT require TorchRL, Gymnasium, MuJoCo, ROS 2, Gazebo, Isaac, or CARLA.

#### Scenario: Base install has no reinforcement extra

- **WHEN** a user installs base `nexuml` without `nexuml[reinforcement]`
- **THEN** core and supervised imports work normally
- **AND** attempting to materialize a reinforcement runtime fails with a clear optional-dependency message.

#### Scenario: Heavy simulator is not installed

- **WHEN** a user has `nexuml[reinforcement]` but not a simulator-specific prerequisite
- **THEN** unrelated RL scenarios continue to work
- **AND** resolving/running the heavy simulator scenario reports the missing optional prerequisite clearly.
