## Context

NexuML currently has a strong separation between immutable typed definitions and mutable runtimes, but the training path is dataset-centric. `ScenarioSpec.training` is a `TrainingSpec`; `compiler.py` reads `scenario.data` for input shapes and `scenario.training` for loss/metric/optimizer state; `NexuLightningModule` consumes batches from `NexuDataModule`; and `NexuSession.run()` performs the standard fit -> validate -> post-train fit -> test lifecycle.

That is appropriate for the existing supervised/self-supervised workloads but not for online RL. The old PRISMA repository solved this by keeping Lightning as the outer trainer and embedding TorchRL collection, replay, actor/critic loss modules, and manual optimizer steps inside a `LightningModule`. That basic decision is retained, but the old implementation must not be copied literally: it used epoch hooks as the RL clock, kept raw optimizers outside Lightning, and mixed environment collection with lifecycle callbacks.

Current TorchRL exposes `Collector` as the preferred construction API across direct, process, and Ray collection; PPO uses standard building blocks such as `ClipPPOLoss` and `GAE`; replay buffers are first-class TensorDict stores. NexuML should compose those primitives rather than create its own RL framework.

## Goals / Non-Goals

**Goals:**

- Keep one NexuML training engine: `lightning.Trainer`.
- Add a typed, serializable RL scenario mode without bloating the existing `TrainingSpec`.
- Reuse `CompiledPipeline` as the deployable actor/policy model.
- Reuse TensorDict end-to-end between NexuML and TorchRL.
- Make environments and algorithms discoverable typed components with no live runtime state in serialized config.
- Support a small, real PPO + Gymnasium scenario end-to-end first.
- Make the design naturally extensible to SAC, TD3, DQN, simulators, ROS 2, and later NexuFL federated policy training.
- Keep the delta small enough that current supervised scenarios and exports remain stable.

**Non-Goals:**

- Do not replace Lightning with TorchRL trainers.
- Do not implement a generic RL framework, replay buffer, collector, PPO math, or advantage estimator in NexuML.
- Do not make ROS 2 a required dependency of NexuML or `nexuml-library`.
- Do not put hard-real-time control, robot safety, actuator current loops, collision protection, or emergency-stop logic inside NexuML.
- Do not implement federated RL in NexuFL in this change.
- Do not initially support multi-learner distributed RL or nested Ray Train + Ray Collector execution.
- Do not ship PPO, SAC, TD3, DQN, and multi-agent algorithms all at once; PPO is the first reference implementation.
- Do not create `RLSession`, `RobotSession`, or a parallel trainer hierarchy.

## Decisions

### D1 - Lightning remains the single top-level trainer

RL SHALL be represented by a second Lightning module, not a second trainer/runtime framework:

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
 dataset learning                    TorchRL learning
```

The existing `NexuSession` remains the public lifecycle owner. Its runtime construction branches on the training spec type:

```python
if isinstance(scenario.training, ReinforcementLearningSpec):
    module = NexuRLLightningModule(...)
else:
    module = NexuLightningModule(...)
```

`NexuRLLightningModule` uses:

```python
class NexuRLLightningModule(L.LightningModule):
    def __init__(self, policy, algorithm, environment, collector):
        super().__init__()
        self.automatic_optimization = False
        self.policy = policy
        self.algorithm = algorithm
        self.environment = environment
        self.collector = collector

    def configure_optimizers(self):
        return self.algorithm.configure_optimizers(self.policy)

    def training_step(self, rollout, batch_idx):
        optimizers = self.optimizers()
        metrics = self.algorithm.update(
            rollout=rollout,
            policy=self.policy,
            optimizers=optimizers,
            lightning_module=self,
        )
        self.log_dict(metrics)
```

The exact runtime protocol may use a dataclass/context instead of this positional signature, but the ownership is fixed:

- Lightning owns device placement, precision, strategy, logging, checkpoint hooks, backward calls, and optimizer wrappers.
- TorchRL owns environment, collection, replay, policy distributions, value estimation, RL losses, and target updates.
- NexuML owns typed configuration, materialization, orchestration, export, and stable policy identity.

Alternative considered: introduce `TorchRLTrainingRuntime` parallel to Lightning. Rejected because it duplicates accelerator/checkpoint/logging/execution behavior and weakens the current one-trainer mental model.

### D2 - Preserve `TrainingSpec`; add a union rather than a rename

Do not rename the existing public `TrainingSpec` to `SupervisedTrainingSpec`. Add:

```python
class ReinforcementLearningSpec(SpecModel):
    mode: Literal["reinforcement"] = "reinforcement"

    environment: EnvironmentDefinition
    algorithm: RLAlgorithmDefinition
    collector: CollectorSpec = Field(default_factory=CollectorSpec)

    total_frames: int = 100_000
    frames_per_batch: int = 2_048
    evaluation_episodes: int = 5

    accelerator: str = "auto"
    devices: str | int = "auto"
    strategy: str | StrategySpec = "auto"
    precision: str = "32-true"
```

and:

```python
class ScenarioSpec(SpecModel):
    ...
    training: TrainingSpec | ReinforcementLearningSpec = Field(
        default_factory=TrainingSpec
    )
```

The union should be validated through the existing strict Pydantic models. Existing Python scenarios and persisted supervised YAML remain valid because the existing `TrainingSpec` shape is unchanged. The RL spec has required `environment` and `algorithm` fields and a literal `mode`, so it does not ambiguously validate as the existing training type.

Do not add RL-only keys such as `gamma`, `clip_epsilon`, replay size, or exploration noise directly to `ReinforcementLearningSpec`. Those belong to the concrete algorithm definition.

### D3 - Environment and algorithm are normal typed NexuML components

Extend `components.py`:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentBuildContext:
    seed: int | None = None
    device: str = "cpu"


class EnvironmentDefinition(ComponentDefinition):
    kind = "environment"

    @abstractmethod
    def build(self, context: EnvironmentBuildContext) -> Any:
        """Build a TorchRL EnvBase runtime."""


@dataclass(frozen=True, slots=True)
class RLAlgorithmBuildContext:
    policy: CompiledPipeline
    environment: Any


class RLAlgorithmDefinition(ComponentDefinition):
    kind = "rl_algorithm"

    @abstractmethod
    def build(self, context: RLAlgorithmBuildContext) -> Any:
        """Build a mutable algorithm runtime."""
```

The concrete return types should be imported lazily or guarded under `TYPE_CHECKING` so importing `nexuml` does not require the `rl` extra.

Add matching decorators:

```python
@environment("Gymnasium")
class GymnasiumEnvironment(EnvironmentDefinition):
    ...


@rl_algorithm("PPO")
class PPO(RLAlgorithmDefinition):
    ...
```

and extend `ComponentRegistry.scan()` with exactly the new kinds. Do not create a second registry.

Definitions remain frozen, portable, and runtime-free. Live `EnvBase`, collector, replay buffer, loss modules, target networks, and optimizers exist only after `build()`.

### D4 - Compile pipelines from an explicit input context

RL exposes observation shapes through the environment, while current `compile(scenario)` reads them from `DataSpec`. Avoid faking a dataset for RL.

Factor shape/class/stage-skip inputs into a small context:

```python
@dataclass(frozen=True, slots=True)
class PipelineCompileContext:
    input_sizes: Mapping[str, tuple[int, ...]]
    num_classes: int | None = None
    skip_stages: tuple[str, ...] = ()
```

Refactor the current compiler into an internal/public seam such as:

```python
def compile_pipeline(
    pipeline_spec: PipelineSpec,
    *,
    context: PipelineCompileContext,
    resolved_config: ResolvedConfig,
    training: TrainingSpec | ReinforcementLearningSpec,
) -> CompiledPipeline:
    ...


def compile(scenario: ScenarioSpec) -> CompiledPipeline:
    context = compile_context_from_data(scenario.data)
    return compile_pipeline(
        scenario.pipeline,
        context=context,
        resolved_config=ResolvedConfig.from_scenario(scenario),
        training=scenario.training,
    )
```

The RL runtime performs:

```python
environment = scenario.training.environment.build(...)
context = compile_context_from_environment(environment)
policy = compile_pipeline(
    scenario.pipeline,
    context=context,
    resolved_config=ResolvedConfig.from_scenario(scenario),
    training=scenario.training,
)
```

`compile_context_from_environment()` maps TorchRL observation specs to NexuML TensorDict keys and shapes. It fails loudly for unsupported non-tensor observation leaves instead of guessing.

For compatibility, `CompiledPipeline.loss_keys`, `metric_keys`, and optimizer/scheduler specs may remain for the supervised path in this change. For RL they are empty/absent and SHALL NOT be used. A broader removal of training semantics from `CompiledPipeline` is a separate refactor.

### D5 - The main pipeline is always the deployable policy

`scenario.pipeline` has one stable meaning in RL: the model that maps observation TensorDict entries to policy parameters/actions and is the default deployable/federated artifact.

Training-only models belong to the concrete algorithm definition/runtime. Example:

```python
@rl_algorithm("PPO")
class PPO(RLAlgorithmDefinition):
    clip_epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    entropy_coef: float = 0.01
    critic: PipelineSpec = Field(default_factory=default_value_pipeline)
    actor_lr: float = 3e-4
    critic_lr: float = 1e-3
    update_epochs: int = 10
    minibatch_size: int = 256
```

The PPO runtime compiles the critic from the same environment-derived `PipelineCompileContext`, wraps actor and critic for TorchRL, constructs `GAE` and `ClipPPOLoss`, and owns any temporary replay/minibatch machinery.

This avoids a generic `auxiliaries: dict[str, PipelineSpec]` abstraction before multiple real algorithms prove it necessary.

### D6 - Add a thin TensorDict policy adapter instead of changing CompiledPipeline

`CompiledPipeline.forward()` returns `(x, y)`, while TorchRL policy modules conventionally consume/return a TensorDict. Add one private adapter in the RL implementation:

```python
class _TorchRLPolicyAdapter(nn.Module):
    def __init__(self, pipeline: CompiledPipeline):
        super().__init__()
        self.pipeline = pipeline

    def forward(self, tensordict):
        x_out, _ = self.pipeline(tensordict, None)
        return x_out
```

Do not modify `CompiledPipeline.forward()` just to mimic a TorchRL API. The adapter keeps the current NexuML contract stable.

For the initial continuous-action PPO example, the policy pipeline SHALL emit distribution parameters under explicit keys, for example `action_loc` and `action_scale`. The PPO runtime wraps those with TorchRL's probabilistic actor using the environment action spec and produces the final `action` and sample log-probability keys.

A library policy head may encapsulate the positive-scale transform so scenario authors do not hand-code it repeatedly.

### D7 - Collector batches are the Lightning training dataloader

Do not collect in `on_train_epoch_start()` as PRISMA did. Add a tiny iterable bridge around the TorchRL collector:

```python
class RolloutDataset(torch.utils.data.IterableDataset):
    def __init__(self, collector):
        self.collector = collector

    def __iter__(self):
        yield from self.collector


def make_rollout_loader(collector):
    return DataLoader(
        RolloutDataset(collector),
        batch_size=None,
        num_workers=0,
    )
```

One Lightning `training_step` therefore corresponds to one collected rollout batch. The algorithm runtime may perform multiple PPO minibatch/update epochs inside that step using manual optimization.

Initial RL runs SHALL be finite: `total_frames > 0`. TorchRL's collector exhausts after that frame budget. The Lightning trainer uses one logical fit epoch over the finite iterable and does not reinterpret `max_epochs` as an RL training budget.

Metrics should use frame/rollout semantics:

- `train/reward_mean`
- `train/episode_return_mean` when complete episodes are present
- `train/frames`
- `train/loss_actor`
- `train/loss_critic`
- PPO-specific metrics such as entropy/clip fraction when emitted by TorchRL.

### D8 - Lightning owns optimizers even with manual optimization

Unlike PRISMA's old `configure_optimizers() -> None`, RL algorithms SHALL return real optimizers through the Lightning module's `configure_optimizers()`.

The algorithm runtime may declare optimizer factories in its typed definition, but the final optimizer objects are returned to Lightning and retrieved with `self.optimizers()`. Updates use:

```python
optimizer.zero_grad()
self.manual_backward(loss)
self.clip_gradients(optimizer, ...)
optimizer.step()
```

Do not call raw `torch.optim.Optimizer.step()` objects that Lightning never received. This keeps precision plugins, accelerator integration, profiling, and checkpoint state coherent.

### D9 - PPO is the reference algorithm; NexuML does not reimplement PPO

The library PPO runtime uses TorchRL primitives such as:

- `ProbabilisticActor`
- `ValueOperator` or a TensorDict-compatible critic adapter
- `GAE`
- `ClipPPOLoss`
- TorchRL `ReplayBuffer`/`TensorDictReplayBuffer` only where useful for minibatching

NexuML code only wires those components and translates their result TensorDict into Lightning logging/optimization.

Algorithm-specific output loss names remain internal to the PPO runtime. The general NexuML RL module delegates rather than containing `if algorithm == "ppo"` branches.

### D10 - Use TorchRL Collector as the collection topology seam

Define a small serializable collector spec:

```python
class CollectorSpec(SpecModel):
    backend: Literal["direct", "process", "ray"] = "direct"
    num_collectors: int = 1
    sync: bool = True
    env_device: str = "cpu"
    storing_device: str = "cpu"
    policy_device: str = "auto"
```

Materialization maps it to the current TorchRL `Collector` front door:

```python
Collector(
    create_env_fn=environment_factory,
    policy=policy,
    frames_per_batch=training.frames_per_batch,
    total_frames=training.total_frames,
    num_collectors=...,
    sync=...,
    backend="ray" if requested else None,
    ...
)
```

The first acceptance path is `backend="direct"`. Process collection may be enabled once smoke-tested. Ray collection is an explicit later validation path.

`scenario.execution` and RL collector backend are independent axes:

- `execution` answers where the Lightning learner runs.
- `collector` answers where/how environments collect experience.

Until nested placement is validated, reject `execution.kind == "ray"` together with `collector.backend == "ray"` with a clear error rather than accidentally starting Ray inside Ray Train workers.

### D11 - Gymnasium + Pendulum is the first library vertical slice

Add:

```text
library/src/nexuml_library/
├── environments/
│   ├── __init__.py
│   └── gymnasium.py
├── reinforcement/
│   ├── __init__.py
│   ├── algorithms/
│   │   ├── __init__.py
│   │   └── ppo.py
│   └── layers/
│       ├── __init__.py
│       └── gaussian_policy_head.py
└── scenarios/
    └── reinforcement/
        ├── __init__.py
        └── pendulum_ppo.py
```

Example environment definition:

```python
@environment("Gymnasium")
class GymnasiumEnvironment(EnvironmentDefinition):
    env_name: str
    device: str = "cpu"

    def build(self, context: EnvironmentBuildContext):
        from torchrl.envs.libs.gym import GymEnv

        return GymEnv(
            self.env_name,
            device=self.device if self.device != "auto" else context.device,
        )
```

Example scenario shape:

```python
@scenario("pendulum-ppo")
def pendulum_ppo(
    total_frames: int = 50_000,
    frames_per_batch: int = 1_024,
) -> ScenarioSpec:
    return ScenarioSpec(
        name="pendulum_ppo",
        pipeline=continuous_policy_mlp(
            observation_key="observation",
            loc_key="action_loc",
            scale_key="action_scale",
            hidden_dims=[64, 64],
        ),
        training=ReinforcementLearningSpec(
            environment=GymnasiumEnvironment(env_name="Pendulum-v1"),
            algorithm=PPO(
                critic=value_mlp(hidden_dims=[64, 64]),
                clip_epsilon=0.2,
                gamma=0.99,
                gae_lambda=0.95,
                actor_lr=3e-4,
                critic_lr=1e-3,
                update_epochs=10,
                minibatch_size=256,
            ),
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            evaluation_episodes=5,
        ),
        logging=default_logging(...),
        exports=default_exports(),
    )
```

Exact TorchRL constructor keyword names must follow the installed version; the public NexuML definition fields above are the stable contract and should not expose every TorchRL constructor knob.

### D12 - ROS 2 is an environment implementation boundary, not a trainer backend

A ROS-based user/library environment implements the same `EnvironmentDefinition -> EnvBase` contract. NexuML core does not know about topics, services, actions, Gazebo, or `ros2_control`.

A custom environment can look like:

```python
@environment("Ros2JointControl")
class Ros2JointControl(EnvironmentDefinition):
    node_name: str = "nexuml_rl"
    joint_state_topic: str = "/joint_states"
    command_topic: str = "/joint_trajectory_controller/joint_trajectory"
    reset_service: str | None = "/simulation/reset"
    control_period_s: float = 0.05

    def build(self, context: EnvironmentBuildContext):
        from my_robot_library.ros_env import Ros2JointControlEnv

        return Ros2JointControlEnv(
            node_name=self.node_name,
            joint_state_topic=self.joint_state_topic,
            command_topic=self.command_topic,
            reset_service=self.reset_service,
            control_period_s=self.control_period_s,
        )
```

That runtime owns observation/action specs, reset/step synchronization, timeouts, and task-specific reward/termination semantics.

For real hardware, the learned policy should normally issue bounded high-level position/velocity/trajectory commands through ROS 2/`ros2_control`; hard-real-time low-level control and safety interlocks stay outside NexuML.

Do not add `rclpy` to the base or `rl` extra. A future ROS-specific package/library may provide installation guidance for supported ROS distributions.

### D13 - RL evaluation uses fresh environments and episode metrics

RL evaluation SHALL not reuse the training environment state blindly. The runtime builds a separate evaluation environment from the same immutable definition (or an explicit evaluation environment field added later when a real use case requires it).

After training, run `evaluation_episodes` deterministic policy rollouts and report at least:

- mean episode return;
- standard deviation of episode return;
- mean episode length.

Map these into the existing `TrainResult` result dictionaries so CLI/logging callers do not need a second result type.

The existing post-train `EvalAlgorithmDefinition` system remains dataset/output evaluation and is not overloaded to implement the RL environment loop in this change.

### D14 - Export the policy as the portable model; keep RL training state separate

The ordinary NexuML train package SHALL continue to package `scenario.pipeline` / `CompiledPipeline` as the primary model artifact.

For an RL run:

- policy/actor parameters are part of the exported model state;
- critic/Q/target networks are training-only unless a future explicit export mode requests them;
- replay buffer and collected trajectories are never included in the normal train package;
- live environments/ROS objects are never serialized into the model artifact;
- metadata records `training_mode="reinforcement"`, the algorithm identity/version, and environment identity/version/config hash;
- a Lightning checkpoint may contain enough RL module state for local resume, but that checkpoint is not the deployable policy contract.

This is the seam needed for later NexuFL:

```text
NexuFL round
    |
    +--> load federated policy weights
    +--> local environment interaction
    +--> local PPO/SAC/etc. updates
    +--> return selected policy update + metrics

local by default:
    trajectories, observations, replay, ROS state, environment,
    optimizer state, critic/target state
```

This change does not implement that NexuFL client procedure; it ensures NexuML does not make it impossible.

### D15 - Optional dependencies stay layered

Add:

```toml
[project.optional-dependencies]
rl = [
    "torchrl>=0.14,<0.15",
    "gymnasium>=1",
]

all = [
    "nexuml[library,audio,data,tracking,tuning,export,ray,s3,rl]",
]
```

The exact lower bounds should be confirmed against the lockfile during implementation. Do not add ROS Python packages to this extra.

Core modules must use lazy imports so `import nexuml` and existing supervised scenarios work without `nexuml[rl]`.

## Implementation Sketch

Expected focused core delta:

```text
src/nexuml/
├── core/
│   ├── components.py        # + env/RL definitions and build contexts
│   ├── discovery.py         # + @environment/@rl_algorithm
│   ├── registry.py          # + two component kinds
│   ├── compiler.py          # + PipelineCompileContext seam
│   └── types.py             # + ReinforcementLearningSpec/CollectorSpec
│
├── reinforcement/
│   ├── __init__.py
│   ├── adapter.py           # CompiledPipeline -> TorchRL TensorDict adapter
│   ├── collector.py         # Collector materialization + RolloutDataset
│   └── runtime.py           # small shared runtime protocols/helpers
│
└── training/
    ├── lightning.py         # existing supervised behavior + session branching
    └── reinforcement.py     # NexuRLLightningModule
```

Avoid creating generic manager/factory/service layers beyond these seams.

## Testing Strategy

Keep tests focused:

1. Registry test: environment and RL algorithm definitions discover, serialize, and restore by stable identity.
2. Config test: current supervised `ScenarioSpec` YAML round-trips unchanged; RL scenario YAML round-trips with typed definitions.
3. Compiler-context test: one pipeline compiles from dataset shapes and from environment observation shapes.
4. Lightning ownership test: PPO optimizers are returned by `configure_optimizers()`; a tiny manual-optimization step changes actor parameters.
5. End-to-end smoke: `pendulum-ppo` runs a very small finite frame budget on CPU, logs reward/loss metrics, evaluates episodes, and returns a normal `TrainResult`.
6. Export test: RL-trained policy exports/loads and runs inference from an observation TensorDict without requiring a live environment.
7. Regression: existing supervised session/export/Ray tests remain unchanged and pass.
8. Optional import test: `import nexuml` and supervised training do not require TorchRL/Gymnasium.
9. ROS extension example test: if a sample helper is shipped, test its configuration/transport boundary with a fake runtime rather than requiring ROS in normal CI.

Do not add long stochastic convergence tests. The smoke test verifies wiring, not that PPO solves Pendulum in CI.

## Risks / Trade-offs

- [Lightning's fit loop is dataset-oriented] -> expose TorchRL collector batches through an `IterableDataset` and keep one finite logical fit epoch.
- [Manual optimization can bypass Lightning features] -> return all optimizers through `configure_optimizers()` and use Lightning optimizer wrappers/manual backward.
- [TorchRL API evolves quickly] -> expose a small NexuML semantic config and keep TorchRL constructor details inside library runtimes.
- [Actor/critic ownership becomes ambiguous] -> declare `scenario.pipeline` as the only default deployable policy; algorithm-owned networks are training-only.
- [Compiler currently assumes `DataSpec`] -> add one explicit `PipelineCompileContext` rather than fake data sources for environments.
- [Nested Ray can allocate recursively] -> reject learner-on-Ray plus Ray-collector composition until explicitly validated.
- [ROS environments are stateful and hardware-sensitive] -> keep ROS out of core dependencies and require runtime construction from typed definitions.
- [RL checkpoints contain more state than deployable models] -> distinguish resumable Lightning checkpoint from portable policy export.

## Migration Plan

1. Add the new typed component roles/specs and compile context with no behavior change to supervised scenarios.
2. Add the RL Lightning module and direct TorchRL collector bridge behind the `rl` optional extra.
3. Implement Gymnasium + PPO library components and the Pendulum smoke scenario.
4. Add RL evaluation and export metadata; verify the exported policy remains loadable independently of its environment.
5. Add docs and a ROS 2 custom-environment example.
6. Only after the local reference path is stable, validate process collection and then Ray collection as separate follow-ups.
7. Later, add a NexuFL interactive client procedure that consumes the same policy artifact and performs local RL without transmitting experience data.

Rollback is straightforward: remove the new RL component kinds/specs/modules and optional extra. Existing `TrainingSpec`, supervised scenarios, and their artifacts remain the baseline contract.
