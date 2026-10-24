## Purpose

Defines reinforcement-learning scenarios that use Lightning as NexuML's single top-level training engine, TorchRL for reinforcement primitives, and the scenario-level interaction contract for Physical-AI environment/policy integration.

## ADDED Requirements

### Requirement: Scenario supports reinforcement-learning training

`ScenarioSpec.training` SHALL accept a typed `ReinforcementLearningSpec` in addition to the existing `TrainingSpec`. Existing supervised scenarios SHALL remain valid without source changes.

`ReinforcementLearningSpec` SHALL configure training semantics rather than own the environment. At minimum it SHALL contain:

- one typed RL algorithm definition;
- collector configuration;
- finite frame budget;
- frames per collector batch;
- RL evaluation configuration;
- accelerator/devices/strategy/precision settings compatible with the existing trainer surface.

The scenario-level `interaction` SHALL identify the environment and policy/action-adapter configuration.

#### Scenario: Existing supervised scenario is loaded

- **WHEN** a scenario supplies the existing `TrainingSpec`
- **THEN** NexuML follows the current dataset/Lightning training path
- **AND** no RL dependency or interaction environment is required.

#### Scenario: RL scenario is loaded

- **WHEN** a scenario supplies `ReinforcementLearningSpec`
- **THEN** the scenario must also contain `InteractionSpec`
- **AND** NexuML selects the RL Lightning module without creating a second trainer type.

### Requirement: Lightning remains the training engine

All reinforcement-learning runs SHALL execute under `lightning.Trainer`. The RL module SHALL use manual optimization where required by actor/critic algorithms but SHALL register optimizers with Lightning.

#### Scenario: RL training starts

- **WHEN** `NexuSession.fit()` is called for an RL scenario
- **THEN** the existing session constructs an `L.Trainer`
- **AND** trains a `NexuRLLightningModule`
- **AND** does not introduce a parallel TorchRL trainer/session API.

#### Scenario: Algorithm uses multiple optimizers

- **WHEN** the RL algorithm requires actor and critic optimizers
- **THEN** the RL Lightning module returns them from `configure_optimizers()`
- **AND** uses Lightning-managed optimizer wrappers/manual backward APIs for updates.

### Requirement: Driver-side runtime construction remains free of live environments

Before Lightning/Ray placement, NexuML SHALL compile the interaction contract, neural pipeline, action adapter, and algorithm module state, but SHALL NOT create the long-lived training environment or collector.

#### Scenario: Session setup runs on a driver

- **WHEN** `NexuSession.setup()` executes before distributed placement
- **THEN** it may call the immutable environment definition's description boundary
- **AND** it does not retain a simulator, ROS node, socket, process collector, or hardware handle.

### Requirement: Collector/environment lifecycle is worker-local

The RL Lightning runtime SHALL materialize direct environment/collector resources in the learner worker or pass immutable environment factories/definitions to process/Ray collector workers for materialization there.

All owned live resources SHALL have deterministic teardown.

#### Scenario: Fit starts with direct collection

- **WHEN** the Lightning worker enters fit setup
- **THEN** it builds the environment and collector from immutable definitions
- **AND** exposes rollout batches through the module's training dataloader/iterable bridge.

#### Scenario: Fit ends or fails

- **WHEN** training completes, is cancelled, or raises
- **THEN** collector and environment close/teardown hooks are invoked
- **AND** ROS/simulator resources are not intentionally left running by NexuML.

### Requirement: RL uses frame-based collection semantics

RL training SHALL use environment frames and collector batches as its training budget. NexuML SHALL NOT expose a converted synthetic epoch count as the user-visible RL budget.

#### Scenario: Finite RL run executes

- **WHEN** `total_frames=10000` and `frames_per_batch=1000`
- **THEN** collection terminates at the configured frame budget subject only to backend batch-boundary semantics
- **AND** each collector batch is presented to the RL training step
- **AND** the run does not require a dataset-sized epoch definition.

### Requirement: Collection is not hidden in epoch callbacks

Experience collection SHALL be represented by an iterable rollout source owned by the RL runtime rather than side effects in `on_train_epoch_start()`.

#### Scenario: Trainer requests the next RL batch

- **WHEN** Lightning advances the RL training iterable
- **THEN** NexuML obtains the next TensorDict rollout from the collector
- **AND** the resulting rollout is the input to `training_step`.

### Requirement: Algorithm runtime owns RL update semantics

The generic RL Lightning module SHALL delegate algorithm-specific loss, replay, target, advantage, entropy, exploration, and update behavior to the selected algorithm runtime.

#### Scenario: PPO update is executed

- **WHEN** a PPO rollout reaches `training_step`
- **THEN** the PPO runtime computes advantages/losses and performs its configured minibatch updates
- **AND** the generic RL Lightning module contains no PPO-specific conditional branch.

### Requirement: Train-only algorithm modules participate in checkpoint state

Training-only actor-critic auxiliaries needed for resume SHALL be registered under the RL Lightning module and restored by the normal checkpoint path.

#### Scenario: PPO checkpoint is restored

- **WHEN** local training resumes from a compatible checkpoint
- **THEN** actor, critic/value network, optimizer state, counters, and explicitly checkpointable algorithm state are restored
- **AND** live environment/collector resources are rebuilt rather than deserialized.

#### Scenario: Replay buffer is not configured for checkpoint persistence

- **WHEN** a run resumes and replay persistence is disabled
- **THEN** the replay buffer may restart empty according to the algorithm's documented semantics
- **AND** this does not alter the portable policy artifact contract.

### Requirement: Collector policy weights are synchronized after learning updates

For collector topologies in which rollout workers do not share the learner's live policy module, NexuML SHALL explicitly synchronize updated deployable policy weights according to collector configuration.

`CollectorSpec` SHALL provide a semantic synchronization setting such as `policy_sync_interval_batches` with a safe default.

#### Scenario: Direct collector shares policy parameters

- **WHEN** the direct collector uses the learner policy by reference
- **THEN** synchronization may be a no-op.

#### Scenario: Process or Ray collectors own policy copies

- **WHEN** the learner completes the configured number of update batches
- **THEN** NexuML invokes the supported TorchRL collector weight-update mechanism
- **AND** subsequent rollouts use refreshed policy weights.

### Requirement: Collection placement and learner placement are separate axes

`scenario.execution` SHALL continue to describe learner placement. `CollectorSpec` SHALL describe rollout collection topology.

#### Scenario: Local learner with process collectors

- **WHEN** learner execution is local and collector backend is process
- **THEN** the learner remains local while environment workers run in collector processes.

#### Scenario: Unsupported nested Ray composition is requested

- **WHEN** learner execution and collector configuration would create an unvalidated nested Ray topology
- **THEN** configuration fails before allocation with a clear diagnostic
- **AND** NexuML does not recursively start Ray runtimes by accident.

### Requirement: Reinforcement evaluation is explicit and episodic

`ReinforcementLearningSpec` SHALL contain an `RLEvaluationSpec` with at least:

- episode count;
- deterministic/stochastic policy mode;
- seed;
- optional max steps per episode.

`InteractionSpec` SHALL support an optional evaluation-environment override; otherwise the training environment definition is reused to create a fresh evaluation runtime.

#### Scenario: PPO run completes

- **WHEN** training finishes and evaluation episodes are positive
- **THEN** NexuML creates a fresh evaluation environment at the evaluation execution site
- **AND** reports mean episode return, return standard deviation, mean episode length, and termination/truncation counts or rates where available.

#### Scenario: Domain-randomized training uses fixed evaluation

- **WHEN** `interaction.evaluation_environment` is configured
- **THEN** evaluation uses that definition rather than reusing the training environment definition
- **AND** the deployable policy is unchanged.

#### Scenario: Deterministic evaluation is configured

- **WHEN** evaluation requests deterministic policy behavior
- **THEN** `CompiledPolicy` invokes the action adapter's deterministic mode
- **AND** no PPO learner object is required to choose the deterministic action.

### Requirement: RL results reuse the existing public result surface

Episode-level RL metrics SHALL be returned through `TrainResult`/existing logging surfaces rather than introducing an unrelated public result hierarchy.

#### Scenario: RL run returns

- **WHEN** `NexuSession.run()` finishes an RL scenario
- **THEN** callers can access the compiled/deployable policy and trainer
- **AND** RL evaluation metrics are available in the normal result dictionaries/metadata.

### Requirement: Supervised lifecycle remains unchanged

The existing supervised fit -> validate -> post-train fit -> test lifecycle SHALL remain unchanged.

RL runs SHALL use a mode-appropriate lifecycle of fit -> fresh episodic evaluation and SHALL NOT invoke dataset validation/post-train-fit/test phases merely to preserve superficial symmetry.

#### Scenario: Current supervised regression test runs

- **WHEN** an existing supervised scenario is executed
- **THEN** it follows the same lifecycle as before this change
- **AND** no RL branch changes its datamodule, validation, post-train-fit, or test behavior.
