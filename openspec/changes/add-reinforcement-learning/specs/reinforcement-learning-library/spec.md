## Purpose

Defines the maintained NexuML library vertical slices for reinforcement learning and Physical-AI interaction: Gymnasium/PPO, deployable action adapters, ROS 2 simulation/real-hardware boundaries, and executable scenarios that exercise representative control modalities.

## ADDED Requirements

### Requirement: Library provides a Gymnasium environment definition

The maintained library SHALL provide a typed `GymnasiumEnvironment` definition that describes Gymnasium observation/action spaces as a NexuML `InteractionContract` and builds a framework-neutral NexuML environment runtime.

The reinforcement integration MAY adapt that runtime to TorchRL internally; the public environment definition SHALL not require users to author TorchRL `EnvBase` subclasses.

#### Scenario: Pendulum environment is described

- **WHEN** the definition is configured with `Pendulum-v1`
- **THEN** it exposes vector observations and bounded continuous actions
- **AND** the policy can compile before the long-lived training environment exists.

#### Scenario: Gymnasium exposes no static schema API

- **WHEN** a specific Gymnasium environment requires construction to inspect its spaces
- **THEN** the definition may use a short-lived local descriptor probe
- **AND** that probe is immediately closed and is not reused as the training environment.

### Requirement: Library provides deployable action adapters

The maintained library SHALL provide at least:

- a direct action adapter for pipelines that already emit valid action tensors;
- a bounded Gaussian adapter suitable for continuous PPO policies;
- a categorical/discrete adapter suitable for discrete PPO policies.

Adapters SHALL support deterministic inference where meaningful and SHALL serialize independently of the training algorithm runtime.

#### Scenario: Gaussian policy is exported

- **WHEN** a pipeline emits location/scale parameters
- **THEN** the Gaussian adapter creates valid bounded actions
- **AND** the same adapter can be loaded for deployment without the PPO critic/loss runtime.

### Requirement: Library provides PPO as the reference RL algorithm

The maintained library SHALL provide a typed PPO definition whose runtime uses TorchRL objective/value-estimation primitives rather than a NexuML reimplementation of PPO mathematics.

#### Scenario: PPO runtime is built

- **WHEN** PPO is selected
- **THEN** the runtime creates required loss/value/advantage machinery using TorchRL
- **AND** owns its training-only critic/value pipeline separately from the deployable policy pipeline.

#### Scenario: PPO update runs

- **WHEN** a rollout TensorDict is passed to the PPO runtime
- **THEN** advantages and clipped PPO losses are computed
- **AND** configured actor/critic optimizers are stepped through Lightning manual-optimization APIs
- **AND** PPO metrics are returned for NexuML logging.

### Requirement: Library provides small continuous and discrete policy helpers

The maintained library SHALL provide concise policy compositions/helpers so common examples do not require manual TensorDict distribution plumbing.

#### Scenario: Continuous policy helper is used

- **WHEN** Pendulum compiles
- **THEN** the pipeline emits the parameter keys required by the Gaussian action adapter.

#### Scenario: Discrete policy helper is used

- **WHEN** CartPole compiles
- **THEN** the pipeline emits logits required by the categorical action adapter.

### Requirement: Library provides fast reference RL scenarios

The maintained library SHALL provide discoverable CPU-friendly scenarios:

- `pendulum-ppo` for continuous actions;
- `cartpole-ppo` for discrete actions.

These scenarios SHALL be used for normal CI wiring tests and documentation.

#### Scenario: CI runs fast smoke configurations

- **WHEN** smoke frame budgets are selected
- **THEN** collection, at least one optimizer update, episodic evaluation, export construction, and normal result construction complete
- **AND** tests do not assert stochastic convergence to a reward threshold.

### Requirement: Library provides a visual-observation scenario

The maintained library SHALL provide an executable simulated scenario in which the deployable policy consumes image observations rather than only low-dimensional state.

The implementation MAY use rendered Gymnasium/classic-control observations or another lightweight simulator, but it SHALL exercise the same image TensorDict contract intended for cameras on physical systems.

#### Scenario: Visual policy compiles

- **WHEN** the visual scenario resolves
- **THEN** its interaction contract marks the image field with image modality, shape, and dtype
- **AND** the policy pipeline consumes the image key directly without textualizing the scene.

### Requirement: Library provides an opt-in humanoid scenario

The maintained library SHALL provide an executable humanoid-control scenario using an available simulator such as Gymnasium MuJoCo Humanoid.

This scenario MAY require a simulator-specific optional dependency and SHALL not run in normal CI.

#### Scenario: Humanoid scenario is resolved without MuJoCo

- **WHEN** the optional simulator dependency is missing
- **THEN** discovery/configuration remains usable where possible
- **AND** execution reports the missing simulator prerequisite clearly.

#### Scenario: Humanoid scenario runs

- **WHEN** the simulator dependency is installed
- **THEN** the policy receives high-dimensional proprioceptive state
- **AND** produces bounded continuous control actions through the normal `CompiledPolicy` path.

### Requirement: Library provides an opt-in robot manipulation/control scenario

The maintained library SHALL provide at least one simulated articulated-robot manipulation/control scenario, such as MuJoCo Reacher or a similarly lightweight maintained environment.

The scenario SHALL exercise robot joint/proprioceptive observations and continuous actions separately from the humanoid locomotion example.

#### Scenario: Robot-control scenario runs

- **WHEN** its simulator dependency is available
- **THEN** it trains/evaluates through the same interaction/environment contracts as other scenarios
- **AND** does not require a ROS 2 installation.

### Requirement: Library implements a maintained ROS 2 reference adapter

ROS 2 SHALL be more than a documentation example. The library SHALL contain a maintained ROS 2 reference environment/transport boundary with lazy ROS imports.

The reference adapter SHALL support a bounded high-level joint-control style task and at least:

- joint-state observation subscription;
- high-level command publication or ROS action;
- reset service/action hook for simulation;
- configurable control period and timeout;
- explicit interaction contract;
- deterministic close/shutdown behavior.

Task reward/termination SHALL remain in the task environment/reference scenario rather than pretending ROS transport provides a universal reward function.

#### Scenario: ROS 2 packages are not installed

- **WHEN** a user imports NexuML or uses non-ROS scenarios
- **THEN** `rclpy` is not required
- **AND** ROS-specific execution fails with a focused installation/prerequisite message only when selected.

### Requirement: ROS 2 adapter is testable without ROS middleware

The ROS environment implementation SHALL separate transport from task semantics enough to inject a fake transport in normal CI.

#### Scenario: Fake transport test runs

- **WHEN** normal unit tests execute without a ROS installation
- **THEN** simulated joint-state messages, commands, reset responses, timeouts, and shutdown can be tested
- **AND** the same environment logic used by the real ROS transport is exercised.

### Requirement: ROS 2 adapter can target simulation through the real middleware

An opt-in integration scenario SHALL demonstrate the production ROS 2 transport against a simulator such as Gazebo with `ros2_control`.

#### Scenario: Headless ROS/Gazebo integration runs

- **WHEN** ROS 2/Gazebo prerequisites are installed and the integration test is enabled
- **THEN** NexuML communicates over real ROS 2 topics/services/actions
- **AND** controls the simulated robot through the same high-level adapter intended for later hardware use.

#### Scenario: Physical robot uses the adapter

- **WHEN** the same topic/service contract is provided by real hardware
- **THEN** NexuML does not require a trainer/backend change
- **AND** hard-real-time loops, vendor drivers, safety interlocks, limits, and emergency-stop logic remain below NexuML.

### Requirement: Library provides an autonomous-driving simulation scenario

The maintained library SHALL provide an opt-in self-driving/control scenario using a simulator integration such as CARLA.

The scenario SHALL demonstrate a richer observation contract (for example camera plus ego/vehicle state, with optional additional sensors) and bounded driving actions or trajectory-level actions.

#### Scenario: CARLA is unavailable

- **WHEN** CARLA/system prerequisites are not installed
- **THEN** normal NexuML/reinforcement usage is unaffected
- **AND** the CARLA scenario reports its prerequisite clearly.

#### Scenario: Driving scenario runs

- **WHEN** the simulator is available
- **THEN** the scenario can collect rollouts, train/evaluate a policy, and export the same complete `CompiledPolicy` abstraction
- **AND** no game-state-to-text conversion is required.

### Requirement: Heavy simulator scenarios use explicit test tiers

The repository SHALL distinguish:

- normal unit/CPU CI;
- optional simulator integration tests;
- optional GPU/special-system integration tests;
- real-hardware/manual validation.

Normal CI SHALL not download or boot CARLA, Isaac Sim, Gazebo, or other heavy runtimes.

### Requirement: Reinforcement dependencies are layered

The root package SHALL define:

```toml
[project.optional-dependencies]
reinforcement = [
    "torchrl>=0.14,<0.15",
    "gymnasium>=1,<2",
]
```

with exact compatible bounds confirmed against the lockfile during implementation.

The library package SHALL expose a corresponding `reinforcement` optional dependency for maintained RL examples. Simulator-specific Python dependencies such as MuJoCo SHALL be kept in separate extras where practical.

ROS 2, Gazebo, Isaac, and CARLA system installations SHALL NOT be pulled into the base `reinforcement` extra merely to make the scenario catalog complete.

### Requirement: Physical-AI follow-up modes remain possible

This change SHALL NOT implement imitation learning, VLA training, LeRobot datasets, or world-model training. However, the maintained interaction contracts/policy adapters SHALL be documented as reusable foundations for those modes.

#### Scenario: Future offline robot scenario is added

- **WHEN** a later change adds trajectory/demonstration data
- **THEN** it can use `ScenarioSpec.data` together with the existing `interaction`/policy contract
- **AND** no RL-only environment redesign is required.
