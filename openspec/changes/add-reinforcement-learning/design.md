## Context

NexuML currently has a strong separation between immutable typed definitions and mutable runtimes, but the training path is dataset-centric. `ScenarioSpec.training` is a `TrainingSpec`; `compiler.py` derives policy/model input shapes from `DataSpec`; `NexuLightningModule` consumes batches from `NexuDataModule`; and `NexuSession.run()` implements the supervised fit -> validate -> post-train fit -> test lifecycle.

That is correct for existing supervised/self-supervised workloads but insufficient for Physical AI. Reinforcement learning needs interactive environments, actions, trajectories, rewards, actor/critic state, frame-based stopping, and environment lifecycle management. More importantly, the same physical system will later need other training modes: demonstrations/imitation learning, VLA fine-tuning, world models, offline-to-online learning, and eventually NexuFL fleet/federated adaptation.

The previous proposal already made several good decisions:

- keep Lightning as the single top-level trainer;
- use TorchRL rather than reimplementing PPO/GAE/replay/collectors;
- keep the neural policy pipeline separate from training-only critics;
- add typed environment and RL-algorithm definitions;
- use frames instead of pretending RL has dataset epochs;
- keep ROS 2 and hard-real-time safety outside the base dependency/runtime.

The revision keeps those decisions but changes the abstraction boundary. Core NexuML must not mean "TorchRL environment" when it says environment, and the deployable policy must not mean only "raw neural pipeline" when additional action transformation/distribution logic is required.

## Goals / Non-Goals

**Goals:**

- Keep one public training lifecycle owner: `NexuSession` with `lightning.Trainer`.
- Add typed, serializable RL training without breaking existing supervised scenarios/YAML.
- Introduce the minimum reusable Physical-AI contracts needed by later training modes.
- Keep TensorDict as the common named-tensor representation for multimodal observations/actions.
- Represent observation/action shapes, dtypes, modalities, bounds, vectorization, and control metadata explicitly.
- Separate static environment description from live environment materialization.
- Keep core environment semantics independent of TorchRL, Gymnasium, ROS 2, Gazebo, Isaac, CARLA, and other providers.
- Build stateful environment/collector resources only at their execution site.
- Represent the complete deployable policy as neural pipeline + action adapter + interaction contract.
- Make RL algorithm state checkpointable without polluting normal deployment exports.
- Support direct, process, and later Ray collection with explicit policy-weight synchronization.
- Provide real runnable examples for continuous/discrete/image/humanoid/robot/ROS 2/driving categories at appropriate CI tiers.
- Implement a maintained ROS 2 reference adapter that can be tested with a fake transport and exercised against Gazebo before real hardware.
- Preserve a clean NexuFL boundary: learned policy state is separable from local raw experience/runtime state.

**Non-Goals:**

- Do not replace Lightning with a TorchRL trainer.
- Do not implement a second top-level `RLSession`/`RobotSession` hierarchy.
- Do not reimplement PPO, GAE, replay buffers, or TorchRL collectors in NexuML.
- Do not implement imitation learning, ACT, diffusion policy, VLA, LeRobot, or world-model training in this change.
- Do not implement federated interactive RL in NexuFL in this change.
- Do not claim that the simple humanoid/driving scenarios are state-of-the-art task solutions; they are architecture/integration references.
- Do not put robot drivers, PLC safety, hard-real-time loops, collision safety, emergency stops, or actuator current loops inside NexuML.
- Do not make ROS 2, Gazebo, Isaac, MuJoCo, or CARLA mandatory base dependencies.
- Do not initially support multi-agent or multi-learner distributed RL.
- Do not require every backend to support every vectorization/collector topology on day one; unsupported combinations fail clearly.

## Architectural Overview

```text
ScenarioSpec
│
├── pipeline: PipelineSpec
│      trainable neural TensorDict graph
│
├── policy: PolicySpec | None
│      action adapter / inference semantics
│
├── data: DataSpec
│      existing offline dataset path
│
├── interaction: InteractionSpec | None
│      │
│      ├── environment: EnvironmentDefinition
│      ├── evaluation_environment: EnvironmentDefinition | None
│      └── portable InteractionContract from environment.describe()
│
└── training
       ├── TrainingSpec
       └── ReinforcementLearningSpec
              ├── RLAlgorithmDefinition
              ├── CollectorSpec
              ├── RLEvaluationSpec
              └── finite frame budget

compile
   │
   ├── DataSpec ----------------------> PipelineCompileContext
   │
   └── InteractionContract ----------> PipelineCompileContext
                                          │
                                          v
                                  CompiledPipeline
                                          │
                               + ActionAdapterRuntime
                                          │
                                          v
                                    CompiledPolicy
```

The key intentional split is:

- `pipeline` = trainable neural model graph;
- `policy` = how model outputs become valid actions;
- `interaction` = what observations/actions exist and where they come from;
- `training` = how parameters are learned.

That split is what permits a future imitation/VLA change to reuse the policy and interaction contracts while replacing the training procedure.

## Decisions

### D1 - RL is the first Physical-AI training mode, not the core abstraction

This change remains named `add-reinforcement-learning` and implements PPO first. It does not introduce a generic `PhysicalAITrainer`.

However, new environment/policy I/O contracts SHALL live at a layer that is reusable by future interactive/non-RL modes. In particular, environment ownership moves out of `ReinforcementLearningSpec` into an additive scenario-level `InteractionSpec`.

Conceptually:

```python
class InteractionSpec(SpecModel):
    environment: EnvironmentDefinition
    evaluation_environment: EnvironmentDefinition | None = None


class PolicySpec(SpecModel):
    action_adapter: ActionAdapterDefinition


class ReinforcementLearningSpec(SpecModel):
    mode: Literal["reinforcement"] = "reinforcement"
    algorithm: RLAlgorithmDefinition
    collector: CollectorSpec = Field(default_factory=CollectorSpec)
    evaluation: RLEvaluationSpec = Field(default_factory=RLEvaluationSpec)

    total_frames: int = 100_000
    frames_per_batch: int = 2_048

    accelerator: str = "auto"
    devices: str | int = "auto"
    strategy: str | StrategySpec = "auto"
    precision: str | int = "32-true"


class ScenarioSpec(SpecModel):
    ...
    training: TrainingSpec | ReinforcementLearningSpec = Field(default_factory=TrainingSpec)
    interaction: InteractionSpec | None = None
    policy: PolicySpec | None = None
```

An RL scenario validator requires `interaction` and `policy`. Existing supervised scenarios need neither.

Keeping `data` and `interaction` as separate fields is deliberate. A future workflow may use both:

```text
demonstration dataset -> imitation pretraining
                            |
                            v
                        same policy
                            |
                     online interaction
                            |
                      RL fine-tuning
```

This change does not implement that workflow, but it must not make it structurally impossible.

### D2 - Lightning remains the single top-level trainer

The existing design is retained:

```text
                         NexuSession
                             |
                             v
                       L.Trainer
                             |
             +---------------+---------------+
             |                               |
             v                               v
 NexuLightningModule              NexuRLLightningModule
 dataset learning                 interactive RL learning
```

`NexuSession` remains the public lifecycle owner. It branches internally on training type.

For supervised runs, behavior remains unchanged.

For RL runs:

- `NexuSession` resolves the immutable scenario and interaction contract;
- compiles the pipeline and policy;
- builds the checkpointable RL algorithm module;
- creates `NexuRLLightningModule`;
- does **not** create live environments/collectors;
- `L.Trainer.fit` starts worker-local collection;
- after fit, a fresh episodic evaluation runtime is executed.

There is no second public trainer or session hierarchy.

### D3 - InteractionContract is the portable policy I/O truth

`DataSpec.input_shapes` is too weak for Physical AI. A camera, lidar point cloud, joint-state vector, discrete action, and bounded continuous action cannot be described robustly by shape alone.

Add portable models conceptually equivalent to:

```python
class TensorFieldContract(SpecModel):
    shape: tuple[int, ...]
    dtype: str
    modality: str = "generic"
    low: float | list[float] | None = None
    high: float | list[float] | None = None
    rate_hz: float | None = None
    description: str | None = None


class InteractionContract(SpecModel):
    observations: dict[str, TensorFieldContract]
    actions: dict[str, TensorFieldContract]
    action_space: Literal[
        "continuous",
        "discrete",
        "multi_discrete",
        "structured",
    ]
    num_envs: int = 1
    control_period_s: float | None = None
```

Exact field names may change during implementation, but the semantic contract is fixed.

Examples:

```text
Humanoid:
observations:
  proprioception:
    shape: [376]
    dtype: float32
    modality: proprioception

actions:
  action:
    shape: [17]
    dtype: float32
    modality: joint_command
    low: -1
    high: 1

Robot with camera:
observations:
  front_rgb:
    shape: [3, 224, 224]
    dtype: uint8
    modality: image
  joint_position:
    shape: [7]
    dtype: float32
    modality: proprioception
  force_torque:
    shape: [6]
    dtype: float32
    modality: force
```

Batch/vectorized environment dimensions are not part of individual field `shape`. They are runtime/vectorization metadata.

The initial compiler still primarily consumes shape information, but carrying dtype/modality/bounds now avoids another contract migration later.

### D4 - EnvironmentDefinition has description and materialization boundaries

Add a typed component role:

```python
class EnvironmentDefinition(ComponentDefinition):
    kind = "environment"

    @abstractmethod
    def describe(self) -> InteractionContract:
        ...

    @abstractmethod
    def build(self, context: EnvironmentBuildContext) -> EnvironmentRuntime:
        ...
```

`describe()` returns portable metadata. It must not leave active runtime resources behind.

For providers that offer no static schema (some Gymnasium environments), a definition may create a short-lived local descriptor/probe, inspect observation/action spaces, then immediately close it. That is acceptable because it is not the training runtime and does not connect to real hardware.

ROS/hardware definitions must not connect to a robot merely to compile a scenario. They require enough explicit configuration to describe their contract.

This supports:

```text
driver / config resolution
        |
        +--> environment.describe()
        |        returns portable contract
        |
        +--> compile pipeline/policy
        |
        +--> Lightning/Ray placement
                 |
                 +--> environment.build(...)
                      at execution site
```

### D5 - Core EnvironmentRuntime is framework-neutral

Core NexuML SHALL not require `EnvironmentRuntime` to subclass TorchRL `EnvBase`.

The initial runtime protocol is conceptually:

```python
@dataclass
class InteractionStep:
    observation: TensorDict
    reward: torch.Tensor | None
    terminated: torch.Tensor
    truncated: torch.Tensor
    info: Mapping[str, Any] | None = None


class EnvironmentRuntime(Protocol):
    @property
    def contract(self) -> InteractionContract: ...

    def reset(self, *, seed: int | None = None) -> TensorDict: ...

    def step(self, action: TensorDict) -> InteractionStep: ...

    def close(self) -> None: ...
```

The reinforcement package adds a private adapter:

```text
NexuML EnvironmentRuntime
          |
          v
 _TorchRLEnvironmentAdapter
          |
          v
     TorchRL Collector
```

This makes a ROS 2 environment usable by RL now and by future behavior-cloning evaluation/world-model rollout code without forcing those future systems through a TorchRL inheritance hierarchy.

Native third-party environments can still be efficiently wrapped; the framework-neutral contract is not intended to duplicate every simulator API.

### D6 - EnvironmentBuildContext describes execution-local needs

Conceptual context:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentBuildContext:
    purpose: Literal["train", "eval"]
    seed: int | None = None
    device: str = "cpu"
    worker_rank: int = 0
    num_envs: int = 1
```

Optional additional identifiers may be added if required by Ray/collector workers.

`num_envs` is important from the beginning. Isaac Lab and other robotics simulators gain most of their training throughput from vectorized environments; the NexuML contract must not assume one environment instance means one physical trajectory.

### D7 - Live environments and collectors are worker-local

The original proposal allowed environment/collector objects to be passed into `NexuRLLightningModule.__init__`. That is rejected for the Physical-AI design because:

- ROS nodes/sockets are not safely portable;
- simulators may require process-local GPU/context initialization;
- Ray/DDP may serialize module construction state;
- process/Ray collectors should construct environments where they run.

Driver-side runtime artifacts therefore contain only portable/model state:

```text
driver:
  InteractionContract
  CompiledPipeline
  CompiledPolicy
  RLAlgorithmRuntime
  immutable definitions/config

NOT driver-owned:
  live simulator
  ROS node
  collector worker
  socket
  hardware handle
```

For direct collection, `NexuRLLightningModule.setup("fit")` (or an equivalent worker-local factory hook) builds the environment/collector.

For process/Ray collection, the collector receives a serializable factory containing the immutable `EnvironmentDefinition` + `EnvironmentBuildContext` and constructs environments inside collection workers.

All paths define deterministic teardown/close behavior.

### D8 - Compile pipelines from an explicit context

Retain the previous compile-context refactor, generalized slightly:

```python
@dataclass(frozen=True, slots=True)
class PipelineCompileContext:
    input_sizes: Mapping[str, tuple[int, ...]]
    input_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    num_classes: int | None = None
    skip_stages: tuple[str, ...] = ()
```

Existing:

```python
compile(scenario)
    -> compile_context_from_data(scenario.data)
    -> compile_pipeline(...)
```

Interactive:

```python
contract = scenario.interaction.environment.describe()
context = compile_context_from_interaction(contract)
pipeline = compile_pipeline(
    scenario.pipeline,
    context=context,
    resolved_config=ResolvedConfig.from_scenario(scenario),
    training=scenario.training,
)
```

No fake dataset is constructed for RL.

`CompiledPipeline` may retain current supervised loss/metric/optimizer fields for compatibility in this change; they are not used by RL.

### D9 - The deployable policy is pipeline + action adapter + contract

This corrects the most important ambiguity in the original proposal.

The original design said:

1. `scenario.pipeline` emits Gaussian parameters such as `action_loc` / `action_scale`.
2. TorchRL's probabilistic actor converts those into actions.
3. only `scenario.pipeline` is exported.

Those statements are incompatible: the raw pipeline is not a complete policy if deployment needs extra distribution/bounds logic.

Add an action-adapter component role:

```python
class ActionAdapterDefinition(ComponentDefinition):
    kind = "action_adapter"

    @abstractmethod
    def build(self, contract: InteractionContract) -> ActionAdapterRuntime:
        ...
```

And a runtime wrapper:

```python
class CompiledPolicy(nn.Module):
    def __init__(
        self,
        pipeline: CompiledPipeline,
        action_adapter: ActionAdapterRuntime,
        contract: InteractionContract,
    ):
        ...

    def forward(
        self,
        observation: TensorDict,
        *,
        deterministic: bool = True,
    ) -> TensorDict:
        x, _ = self.pipeline(observation, None)
        return self.action_adapter(x, deterministic=deterministic)
```

Initial adapters:

- `DirectActionAdapter`: route already-produced action keys;
- `GaussianActionAdapter`: create bounded continuous actions from distribution parameters;
- `CategoricalActionAdapter`: select/sample discrete actions from logits.

The adapters should preferably use base PyTorch semantics so exported policies do not require TorchRL merely for inference. The TorchRL integration wraps/adapts them for collector/log-prob needs.

`scenario.pipeline` remains the primary trainable neural identity. `CompiledPolicy` is the executable deployment identity.

### D10 - RL algorithms receive policy + contract, not live environments

Add:

```python
@dataclass(frozen=True, slots=True)
class RLAlgorithmBuildContext:
    policy: CompiledPolicy
    contract: InteractionContract


class RLAlgorithmDefinition(ComponentDefinition):
    kind = "rl_algorithm"

    @abstractmethod
    def build(self, context: RLAlgorithmBuildContext) -> RLAlgorithmRuntime:
        ...
```

The algorithm does not own simulator/ROS lifecycle.

This separation matters for remote collection:

```text
learner:
  CompiledPolicy
  critic
  losses
  optimizers
  algorithm counters

collectors:
  policy copy/reference
  environments
  trajectories

driver:
  immutable definitions + orchestration
```

### D11 - Algorithm runtime must be checkpoint-visible

A critic hidden in a plain Python object attached indirectly to the Lightning module may not be represented correctly in state/checkpoint logic.

Require `RLAlgorithmRuntime` to be an `nn.Module` or expose an explicit equivalent checkpoint protocol. The first implementation should use `nn.Module`:

```python
class RLAlgorithmRuntime(nn.Module):
    def configure_optimizers(self, policy: CompiledPolicy): ...
    def update(self, rollout, *, lightning_module, optimizers): ...
```

`NexuRLLightningModule` registers it as a child module:

```python
self.policy = policy
self.algorithm = algorithm_runtime
```

PPO critic/value modules are children of `self.algorithm` and therefore visible to checkpoint/state_dict machinery.

Replay persistence is algorithm/config specific. Normal policy exports never include replay.

### D12 - Collector batches are the Lightning training iterable

Retain the original strong decision: collection is not hidden in epoch hooks.

Conceptually:

```python
class RolloutDataset(IterableDataset):
    def __init__(self, collector):
        self.collector = collector

    def __iter__(self):
        yield from self.collector
```

The RL module can expose `train_dataloader()` after worker-local setup:

```python
def train_dataloader(self):
    return DataLoader(
        RolloutDataset(self._collector),
        batch_size=None,
        num_workers=0,
    )
```

One Lightning `training_step` corresponds to one collector rollout batch. PPO may run many minibatch update epochs within that step.

`total_frames` is the actual training budget. Lightning uses one finite logical fit epoch internally; users do not configure RL in synthetic epochs.

### D13 - Policy synchronization is explicit for copied-policy collectors

Direct collection may share the same policy object as the learner. Process/Ray collectors generally own copies.

After optimizer updates, copied-policy collectors must be refreshed. Add a semantic setting:

```python
class CollectorSpec(SpecModel):
    backend: Literal["direct", "process", "ray"] = "direct"
    num_collectors: int = 1
    num_envs_per_collector: int = 1
    sync: bool = True

    env_device: str = "cpu"
    storing_device: str = "cpu"
    policy_device: str = "auto"

    policy_sync_interval_batches: int = 1
```

After configured update batches, the runtime uses the supported installed TorchRL collector API to update worker policy weights.

Do not hard-code an unstable TorchRL method name into the public NexuML spec; the internal adapter maps semantic synchronization to the installed API.

A test must simulate separate learner/collector policy copies and prove rollout policy parameters change after synchronization.

### D14 - Learner placement and collection topology are separate

Retain this distinction:

- `scenario.execution`: where the Lightning learner runs;
- `training.collector`: how/where environment interaction runs.

Examples:

```text
local learner + direct env
local learner + process env workers
local learner + Ray env workers
Ray learner + direct env in each learner worker
```

Unvalidated nested Ray compositions must fail early.

The first acceptance path is direct collection. Process collection is enabled after testing. Ray collection remains a staged validation path, but its config and policy-sync semantics are designed now.

### D15 - PPO remains the reference algorithm

The library PPO runtime uses TorchRL primitives such as:

- GAE;
- clipped PPO loss;
- TensorDict-compatible value functions;
- replay/minibatch primitives where useful.

NexuML wires them into Lightning and its own policy/environment contracts; it does not reimplement PPO mathematics.

Example:

```python
@rl_algorithm("PPO")
class PPO(RLAlgorithmDefinition):
    critic: PipelineSpec = Field(default_factory=default_value_pipeline)
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 10
    minibatch_size: int = 256
```

Algorithm-specific settings remain on algorithm definitions, not the general `ReinforcementLearningSpec`.

The first PPO implementation must support:

- bounded continuous Gaussian actions;
- discrete categorical actions.

That gives the scenario suite meaningful action-space coverage immediately.

### D16 - RL evaluation is a fresh policy/environment operation

Add:

```python
class RLEvaluationSpec(SpecModel):
    episodes: int = 5
    deterministic: bool = True
    seed: int | None = None
    max_steps_per_episode: int | None = None
```

`InteractionSpec` may provide:

```python
evaluation_environment: EnvironmentDefinition | None = None
```

After fit:

1. choose evaluation definition (override or training definition);
2. build a fresh environment with `purpose="eval"`;
3. execute `CompiledPolicy` directly;
4. use the action adapter's deterministic/stochastic mode;
5. report at least mean/std return and mean length;
6. close the environment.

Training environment state is never silently reused.

The existing dataset-oriented `EvalAlgorithmDefinition` system is not repurposed for RL environment loops.

### D17 - Session/runtime result types evolve additively

Current `RuntimeArtifacts` assumes a data module and current `TrainResult` assumes `NexuLightningModule`.

Internally, either use a small union of supervised/RL artifacts or make data module optional. The public goal is:

- existing supervised callers see the same objects/behavior;
- RL does not invent a fake `NexuDataModule`;
- `TrainResult.pipeline` remains the neural pipeline for compatibility;
- add `TrainResult.policy: CompiledPolicy | None` so interactive callers can deploy the complete policy.

Conceptual:

```python
@dataclass
class TrainResult:
    pipeline: CompiledPipeline
    policy: CompiledPolicy | None
    lightning_module: L.LightningModule
    trainer: L.Trainer
    validation_results: list[dict[str, float]] = ...
    test_results: list[dict[str, float]] = ...
    eval_algorithm_results: dict[str, float] = ...
    interaction_results: dict[str, float] = ...
```

Exact field placement may follow existing repository conventions; do not add an unrelated result model.

### D18 - ROS 2 is a maintained interaction adapter, not a trainer backend

The original proposal only documented how users could create a ROS environment. This revision requires a maintained reference implementation.

Architecture:

```text
                         NexuML
                            |
                    Ros2 task environment
                            |
                   Ros2Transport protocol
                      /             \
                     /               \
          FakeRos2Transport      RclpyTransport
                 |                    |
             normal CI           real ROS graph
                                      |
                                ros2_control
                                 /        \
                                /          \
                           Gazebo         robot
```

The maintained reference environment should cover a bounded joint-control/reach task. It needs:

- explicit observation/action contract;
- joint-state subscription;
- high-level command publication or action;
- reset hook for simulation;
- control period;
- timeout behavior;
- close/shutdown;
- task-specific reward/termination in the task environment.

It must **not** pretend that generic ROS transport defines a universal reward.

The transport boundary makes normal CI possible without ROS:

```python
class Ros2Transport(Protocol):
    def receive_joint_state(...): ...
    def send_joint_command(...): ...
    def reset(...): ...
    def close(...): ...
```

`FakeRos2Transport` drives deterministic tests.

`RclpyTransport` is lazily imported and used only when ROS is installed.

An opt-in integration test/scenario launches headless Gazebo + `ros2_control` and runs the production `rclpy` path. This provides a genuine simulation of the real adapter rather than a completely different simulator-only code path.

Real hardware later exposes the same high-level ROS contract. Robot safety and hard-real-time loops remain outside NexuML.

### D19 - Library scenarios form a capability matrix

A single Pendulum example is insufficient to prove the architecture.

Use tiers:

| Scenario | Purpose | Dependency/Test tier |
| --- | --- | --- |
| `pendulum-ppo` | continuous actions | fast CPU CI |
| `cartpole-ppo` | discrete actions | fast CPU CI |
| `visual-cartpole-ppo` | image -> action | lightweight/optional rendering CI |
| `mujoco-humanoid-ppo` | high-dimensional humanoid control | optional MuJoCo |
| `mujoco-reacher-ppo` | articulated robot control | optional MuJoCo |
| `ros2-gazebo-joint-reach-ppo` | real ROS adapter against simulation | optional ROS/Gazebo |
| `carla-lane-following-ppo` | camera + ego state + driving control | optional CARLA |

Exact scenario names/environment versions may adjust to available installed packages, but the capability coverage is required.

The CARLA scenario is an integration/reference task, not a claim that simple PPO lane following is a production autonomous-driving architecture.

Likewise, MuJoCo is chosen as a practical first humanoid/manipulation reference. Isaac Lab should later plug into the same `EnvironmentDefinition/InteractionContract` without changing core architecture.

### D20 - Optional dependencies stay layered

Root:

```toml
[project.optional-dependencies]
reinforcement = [
    "torchrl>=0.14,<0.15",
    "gymnasium>=1,<2",
]

all = [
    "nexuml[library,audio,data,tracking,tuning,export,ray,s3,reinforcement]",
]
```

Exact bounds are confirmed against `uv.lock` and supported PyTorch/Python versions during implementation.

Library gets a corresponding `reinforcement` extra for examples and may add separate simulator extras such as a MuJoCo-oriented extra.

Do **not** put these into base `reinforcement` merely for catalog completeness:

- `rclpy` / ROS 2 system installation;
- Gazebo;
- Isaac Sim / Isaac Lab;
- CARLA.

Those are documented integration prerequisites with lazy imports and scenario-specific diagnostics.

### D21 - Export complete policy, not training runtime

Normal RL/interactive export contains:

```text
CompiledPolicy
├── CompiledPipeline weights
├── ActionAdapter definition/state
└── Interaction/Policy I/O contract
```

It excludes:

```text
critic / Q network
target network
optimizer
replay buffer
collected trajectories
collector
simulator state
ROS node/socket
hardware handles
```

A resumable Lightning checkpoint may contain training-only algorithm/optimizer state.

Metadata records:

- `training_mode="reinforcement"`;
- algorithm stable identity/version;
- environment stable identity/version;
- action-adapter stable identity/version;
- interaction-contract hash;
- resolved scenario/config provenance.

This distinction is essential for later NexuFL:

```text
future NexuFL client
      |
      +--> load complete policy/config
      +--> build local environment
      +--> local interaction/training
      +--> select learned policy parameter update
      +--> keep raw trajectories/environment state local
```

### D22 - Future imitation/VLA/world-model work reuses the seams

This change deliberately does not add generic training abstractions before there are real implementations. It does, however, establish the following reusable pieces:

```text
InteractionContract
EnvironmentDefinition
EnvironmentRuntime
PolicySpec
ActionAdapterDefinition
CompiledPolicy
PipelineCompileContext
```

Future imitation learning can use:

```text
trajectory dataset -> CompiledPolicy
                  + optional InteractionSpec for evaluation/fine-tuning
```

Future VLA can use:

```text
camera + language + proprioception
              |
        CompiledPipeline
              |
      DirectActionAdapter
              |
         action chunk
```

Future world models can use the same observation/action contract and environment rollout boundary.

No generic `ImitationLearningSpec`/`VLASpec`/`WorldModelSpec` is added now. Those should be designed from real implementation requirements.

## Example Scenario Shapes

### Continuous control

```python
@scenario("pendulum-ppo")
def pendulum_ppo(...) -> ScenarioSpec:
    return ScenarioSpec(
        name="pendulum_ppo",
        interaction=InteractionSpec(
            environment=GymnasiumEnvironment(env_name="Pendulum-v1"),
        ),
        pipeline=continuous_policy_mlp(
            observation_key="observation",
            loc_key="action_loc",
            scale_key="action_scale",
            hidden_dims=[64, 64],
        ),
        policy=PolicySpec(
            action_adapter=GaussianActionAdapter(
                loc_key="action_loc",
                scale_key="action_scale",
                action_key="action",
            )
        ),
        training=ReinforcementLearningSpec(
            algorithm=PPO(...),
            total_frames=50_000,
            frames_per_batch=1_024,
        ),
        ...
    )
```

### Discrete control

```python
@scenario("cartpole-ppo")
def cartpole_ppo(...) -> ScenarioSpec:
    return ScenarioSpec(
        interaction=InteractionSpec(
            environment=GymnasiumEnvironment(env_name="CartPole-v1"),
        ),
        pipeline=discrete_policy_mlp(
            observation_key="observation",
            logits_key="action_logits",
        ),
        policy=PolicySpec(
            action_adapter=CategoricalActionAdapter(
                logits_key="action_logits",
                action_key="action",
            )
        ),
        training=ReinforcementLearningSpec(
            algorithm=PPO(...),
            ...
        ),
    )
```

### ROS 2 simulation

```python
@scenario("ros2-gazebo-joint-reach-ppo")
def ros2_joint_reach(...) -> ScenarioSpec:
    contract = InteractionContract(
        observations={
            "joint_position": ...,
            "joint_velocity": ...,
            "target_position": ...,
        },
        actions={
            "joint_command": ...,
        },
        action_space="continuous",
        control_period_s=0.05,
    )

    return ScenarioSpec(
        interaction=InteractionSpec(
            environment=Ros2JointReachEnvironment(
                contract=contract,
                joint_state_topic="/joint_states",
                command_topic="/joint_trajectory_controller/joint_trajectory",
                reset_service="/reset_simulation",
            )
        ),
        ...
    )
```

The same environment definition can later target a real ROS graph when topics/services have the same high-level contract.

## Expected Implementation Layout

```text
src/nexuml/
├── core/
│   ├── components.py          # + env/action-adapter/RL definitions + contexts
│   ├── discovery.py           # + decorators
│   ├── registry.py            # + component kinds
│   ├── compiler.py            # + PipelineCompileContext seam
│   ├── policy.py              # CompiledPolicy
│   └── types.py               # interaction/policy/RL specs
│
├── interaction/
│   ├── __init__.py
│   └── runtime.py             # framework-neutral runtime contracts
│
├── reinforcement/
│   ├── __init__.py
│   ├── torchrl_env.py         # EnvironmentRuntime -> TorchRL adapter
│   ├── collector.py           # collector factory, rollout iterable, policy sync
│   └── runtime.py             # shared RL runtime protocols/helpers
│
└── training/
    ├── lightning.py           # existing supervised behavior + session branching
    └── reinforcement.py       # NexuRLLightningModule

library/src/nexuml_library/
├── action_adapters/
│   ├── direct.py
│   ├── gaussian.py
│   └── categorical.py
├── environments/
│   ├── gymnasium.py
│   ├── ros2/
│   │   ├── transport.py
│   │   ├── rclpy_transport.py
│   │   ├── fake_transport.py
│   │   └── joint_reach.py
│   └── carla.py               # optional
├── reinforcement/
│   ├── algorithms/
│   │   └── ppo.py
│   └── policies/
│       ├── continuous.py
│       └── discrete.py
└── scenarios/
    └── reinforcement/
        ├── pendulum_ppo.py
        ├── cartpole_ppo.py
        ├── visual_cartpole_ppo.py
        ├── mujoco_humanoid_ppo.py
        ├── mujoco_reacher_ppo.py
        ├── ros2_gazebo_joint_reach_ppo.py
        └── carla_lane_following_ppo.py
```

Exact modules may be merged where the repository's style prefers fewer files; the ownership boundaries are more important than filenames.

## Testing Strategy

### Normal CPU CI

1. Existing supervised suite unchanged.
2. Registry/serialization tests for environment/action-adapter/RL-algorithm definitions.
3. `InteractionContract` validation/round-trip tests including image/vector/continuous/discrete contracts.
4. Compiler-context tests from both `DataSpec` and `InteractionContract`.
5. `CompiledPolicy` direct/Gaussian/categorical inference tests.
6. Optional-dependency import tests.
7. Tiny Pendulum PPO smoke.
8. Tiny CartPole PPO smoke.
9. Copied-policy collector synchronization test.
10. Checkpoint test proving PPO critic state is restored.
11. Export/load test proving complete policy inference without environment/PPO learner.
12. Fake ROS transport/task tests.
13. Heavy scenario config/contract resolution tests without starting their simulators.

### Optional simulator CI/manual jobs

- visual rendering integration if system graphics dependencies require it;
- MuJoCo humanoid/reacher reset/step/small rollout;
- ROS 2 + headless Gazebo + `ros2_control` production transport integration;
- CARLA server/client reset/step/small rollout.

### GPU/special-system optional jobs

Later Isaac Lab/Isaac Sim integrations, when added, belong here.

### Real hardware

Manual only. No CI should assume access to a robot/vehicle.

Do not add stochastic "must solve environment" assertions. Smoke tests verify architecture/wiring, while longer benchmark scenarios can document expected learning curves separately.

## Risks / Trade-offs

- **[Interaction contract grows too generic]** -> keep first fields concrete and tensor-oriented; add new modalities/structure only when real scenarios require them.
- **[Environment description cannot be fully static]** -> allow short-lived local descriptor probes where unavoidable, but forbid persistent/hardware connections during description.
- **[Framework-neutral runtime duplicates Gym/TorchRL APIs]** -> keep it intentionally tiny: reset/step/close + TensorDict semantics. Provider-specific behavior stays in adapters.
- **[Action adapter duplicates RL policy wrappers]** -> accept the small wrapper because it fixes deployment correctness and makes policy inference algorithm-independent.
- **[Gaussian training/deployment semantics diverge]** -> one adapter definition/runtime is the source of truth for bounds and deterministic behavior; TorchRL integration must respect it.
- **[Lightning fit is dataset-oriented]** -> finite iterable rollout loader, one logical fit epoch, frames remain the public budget.
- **[Algorithm critic is lost on resume]** -> register checkpointable algorithm runtime as Lightning child state.
- **[Remote collectors use stale policy]** -> explicit sync cadence and tests with copied policies.
- **[Simulator/ROS objects break serialization]** -> build only at execution site; serialize immutable definitions/contracts.
- **[Vectorized env batch dimensions leak into model shapes]** -> interaction field shapes exclude vectorized batch dimensions; runtime vectorization is metadata.
- **[Heavy examples make installation/CI brittle]** -> layered extras and explicit test tiers; normal CI never boots heavy simulators.
- **[ROS fake tests drift from real transport]** -> shared task/environment logic + opt-in real ROS/Gazebo integration test.
- **[Simple CARLA PPO is misconstrued as autonomous-driving architecture]** -> document it as an integration/control scenario only.
- **[Future imitation/VLA needs a different trainer]** -> only policy/interaction contracts are generalized now; training-spec design waits for actual implementation requirements.
- **[NexuFL later needs different exchange state]** -> keep neural pipeline parameters identifiable independently of action adapter and local environment experience.

## Migration Plan

1. Add interaction/field/action/policy types and new component roles with zero supervised behavior change.
2. Add `PipelineCompileContext` and `CompiledPolicy`; prove existing `compile(scenario)` remains equivalent.
3. Add reinforcement optional dependencies and TorchRL environment/collector adapter.
4. Add worker-local RL Lightning runtime and checkpoint-visible algorithm state.
5. Implement direct/Gaussian/categorical action adapters and PPO.
6. Add Pendulum/CartPole fast scenarios and export/resume/policy-sync tests.
7. Add lightweight visual scenario.
8. Add optional MuJoCo humanoid/reacher scenarios.
9. Implement fake-testable ROS 2 reference adapter and ROS joint-control scenario.
10. Add opt-in ROS 2 + Gazebo/`ros2_control` integration.
11. Add opt-in CARLA reference scenario.
12. Validate process collection, then Ray collection separately; reject unsupported nested placement until tested.
13. Document future imitation/VLA/world-model usage of the same contracts.
14. Run `openspec validate add-reinforcement-learning --strict` plus repository lint/type/docs/test gates.

Rollback remains additive: remove the new interaction/RL component kinds/specs/modules and optional extras. Existing `TrainingSpec`, supervised scenarios, and their artifacts remain the baseline contract.
