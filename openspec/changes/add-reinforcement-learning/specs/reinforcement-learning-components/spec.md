## Purpose

Defines the typed component and compilation contracts that let NexuML materialize environments and RL algorithms without putting live TorchRL/ROS state into portable configuration.

## ADDED Requirements

### Requirement: Environment is a typed component

NexuML SHALL provide an `EnvironmentDefinition` component role with stable kind/name/version identity and explicit runtime materialization.

#### Scenario: Environment definition is persisted

- **WHEN** a scenario containing a registered environment is serialized
- **THEN** its stable component identity and validated parameters are persisted
- **AND** no live `EnvBase`, ROS node, process, socket, or simulator handle is serialized.

#### Scenario: Environment is materialized

- **WHEN** an RL run builds its runtime
- **THEN** the environment definition receives an explicit build context
- **AND** returns the mutable environment runtime used for collection.

### Requirement: RL algorithm is a typed component

NexuML SHALL provide an `RLAlgorithmDefinition` component role with stable kind/name/version identity and explicit runtime materialization.

#### Scenario: Algorithm definition is discovered

- **WHEN** library discovery scans a decorated RL algorithm definition
- **THEN** the common component registry registers it under kind `rl_algorithm`
- **AND** YAML restoration resolves the exact kind/name/version before validating parameters.

#### Scenario: Algorithm is materialized

- **WHEN** an RL scenario runtime is built
- **THEN** the selected algorithm receives the compiled policy and environment context
- **AND** creates mutable loss modules, critics, replay state, target networks, or other algorithm runtime state as required.

### Requirement: Existing component registry remains authoritative

Environment and RL algorithm identities SHALL use the current component registry/discovery system rather than a separate RL registry.

#### Scenario: Registry lists RL components

- **WHEN** component discovery has loaded the installed library
- **THEN** environment and RL algorithm entries appear in deterministic common-registry listings
- **AND** conflicts are diagnosed with the same stable identity rules as existing components.

### Requirement: Pipeline compilation accepts explicit input context

NexuML SHALL support compiling a `PipelineSpec` from explicit input sizes independent of `DataSpec` so environment observations can define policy inputs.

#### Scenario: Supervised pipeline compiles

- **WHEN** `compile(scenario)` is called for an existing supervised scenario
- **THEN** NexuML derives the compile context from the existing data configuration
- **AND** behavior remains equivalent to the current compiler.

#### Scenario: RL policy compiles

- **WHEN** an RL environment exposes tensor observation specs
- **THEN** NexuML derives input TensorDict key shapes from those specs
- **AND** compiles `scenario.pipeline` against them without constructing a fake dataset.

#### Scenario: Unsupported observation leaf is present

- **WHEN** the environment exposes an observation structure that cannot be represented by the supported TensorDict tensor-shape contract
- **THEN** compilation fails with a diagnostic naming the unsupported observation key/type
- **AND** does not silently substitute a default shape.

### Requirement: Deployable pipeline retains its existing forward contract

RL integration SHALL adapt `CompiledPipeline` to TorchRL without changing the public `CompiledPipeline.forward(x, y) -> (x_out, y_out)` contract.

#### Scenario: TorchRL requests policy inference

- **WHEN** a collector passes an observation TensorDict to the RL policy adapter
- **THEN** the adapter invokes the compiled pipeline
- **AND** returns the pipeline's output TensorDict to the TorchRL policy wrapper.

### Requirement: RL dependencies are optional

Importing NexuML and using existing supervised scenarios SHALL NOT require TorchRL, Gymnasium, ROS 2, or `rclpy`.

#### Scenario: Base install has no RL extra

- **WHEN** a user installs base `nexuml` without `nexuml[rl]`
- **THEN** core and supervised imports work normally
- **AND** attempting to materialize an RL environment/algorithm fails with a clear optional-dependency message.
