## Purpose

Defines the first maintained NexuML library vertical slice for reinforcement learning and the extension boundary for robotics/ROS environments.

## ADDED Requirements

### Requirement: Library provides a Gymnasium environment definition

The maintained library SHALL provide a typed environment definition that wraps a Gymnasium-compatible environment through TorchRL.

#### Scenario: Pendulum environment is built

- **WHEN** the definition is configured with `Pendulum-v1`
- **THEN** it materializes a TorchRL environment with observation and action specs
- **AND** exposes TensorDict-compatible reset/step behavior.

### Requirement: Library provides PPO as the reference RL algorithm

The maintained library SHALL provide a typed PPO definition whose runtime uses TorchRL objective/value-estimation primitives rather than a NexuML reimplementation of PPO math.

#### Scenario: PPO runtime is built

- **WHEN** PPO is selected for a continuous-action policy
- **THEN** the runtime creates the required probabilistic policy/value/loss machinery using TorchRL
- **AND** owns its training-only critic/value pipeline separately from the deployable actor pipeline.

#### Scenario: PPO update runs

- **WHEN** a rollout TensorDict is passed to the PPO runtime
- **THEN** advantage and clipped PPO losses are computed
- **AND** the configured actor/critic optimizers are stepped through the Lightning manual-optimization context
- **AND** PPO metrics are returned for NexuML logging.

### Requirement: Library provides a small continuous policy helper

The maintained library SHALL provide a concise policy composition/helper suitable for continuous-action algorithms so users do not need to manually reproduce action distribution parameter plumbing.

#### Scenario: Continuous policy is compiled

- **WHEN** the Pendulum scenario compiles its policy
- **THEN** the pipeline accepts the environment observation key
- **AND** emits explicit location and positive scale parameter keys expected by the PPO runtime.

### Requirement: Library provides an executable Pendulum PPO scenario

The maintained library SHALL provide a discoverable `pendulum-ppo` scenario that demonstrates the complete RL path with a CPU-friendly Gymnasium task.

#### Scenario: User resolves the scenario

- **WHEN** the user resolves `pendulum-ppo`
- **THEN** the resulting config contains a typed Gymnasium environment, typed PPO algorithm, policy pipeline, finite frame budget, logging, and export configuration.

#### Scenario: CI runs the smoke scenario

- **WHEN** the smoke configuration runs with a deliberately small frame budget
- **THEN** it completes collection, at least one optimizer update, policy evaluation, and normal result construction
- **AND** the test does not assert stochastic convergence to a reward threshold.

### Requirement: ROS 2 remains an environment extension

Robotics/ROS integration SHALL be implemented as environment components in libraries or user packages and SHALL NOT become a separate NexuML trainer/backend concept.

#### Scenario: User authors a ROS environment

- **WHEN** a user defines a registered ROS 2 environment component
- **THEN** it may map topics/services/actions into observation, action, reset, reward, and termination semantics
- **AND** the same RL Lightning/TorchRL training path can consume it.

#### Scenario: Base RL dependencies are installed

- **WHEN** a user installs `nexuml[rl]`
- **THEN** ROS 2/`rclpy` is not installed or required by that extra
- **AND** ROS-specific environments may declare their own deployment prerequisites.

### Requirement: Robot safety stays below the learning environment

NexuML SHALL NOT claim ownership of hard-real-time control or hardware safety for ROS-based training.

#### Scenario: Policy controls a physical robot

- **WHEN** an RL environment sends actions to a real ROS 2 system
- **THEN** the environment is expected to use bounded high-level commands through the robot's normal controller stack
- **AND** emergency stops, limits, low-level real-time loops, and safety interlocks remain outside NexuML.
