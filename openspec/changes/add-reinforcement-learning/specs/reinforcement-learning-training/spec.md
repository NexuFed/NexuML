## Purpose

Defines reinforcement-learning scenarios that use TorchRL for interactive data collection and RL objectives while preserving Lightning as NexuML's single top-level training engine.

## ADDED Requirements

### Requirement: Scenario supports reinforcement-learning training

`ScenarioSpec.training` SHALL accept a typed reinforcement-learning specification in addition to the existing `TrainingSpec`. Existing supervised scenarios SHALL remain valid without source changes.

#### Scenario: Existing supervised scenario is loaded

- **WHEN** a scenario supplies the existing `TrainingSpec`
- **THEN** NexuML follows the current dataset/Lightning training path
- **AND** no RL dependency or environment is required.

#### Scenario: RL scenario is loaded

- **WHEN** a scenario supplies `ReinforcementLearningSpec`
- **THEN** the spec identifies one typed environment, one typed RL algorithm, collector configuration, and a finite frame budget
- **AND** NexuML selects the RL Lightning module without creating a second trainer type.

### Requirement: Lightning remains the training engine

All reinforcement-learning runs SHALL execute under `lightning.Trainer`. The RL module SHALL use manual optimization for actor/critic updates but SHALL register optimizers with Lightning.

#### Scenario: RL training starts

- **WHEN** `NexuSession.fit()` is called for an RL scenario
- **THEN** the existing session constructs an `L.Trainer`
- **AND** trains a `NexuRLLightningModule`
- **AND** does not invoke a TorchRL trainer as the top-level engine.

#### Scenario: Algorithm uses multiple optimizers

- **WHEN** the RL algorithm requires actor and critic optimizers
- **THEN** the RL Lightning module returns them from `configure_optimizers()`
- **AND** retrieves Lightning-managed optimizer wrappers during `training_step`
- **AND** performs backward/clip/step through Lightning manual-optimization APIs.

### Requirement: RL uses frame-based collection semantics

RL training SHALL use environment frames and collector batches as its training budget. NexuML SHALL NOT expose a converted synthetic epoch count as the RL budget.

#### Scenario: Finite RL run executes

- **WHEN** `total_frames=10000` and `frames_per_batch=1000`
- **THEN** the collector terminates after the configured frame budget
- **AND** each yielded collector batch is presented to the Lightning training step
- **AND** the run does not require a dataset-sized epoch definition.

### Requirement: Collection is not hidden in epoch hooks

Experience collection SHALL be represented by an iterable rollout input to the Lightning fit loop rather than by side effects in `on_train_epoch_start()`.

#### Scenario: Trainer requests the next RL batch

- **WHEN** Lightning advances the RL training dataloader
- **THEN** NexuML obtains the next TensorDict rollout from the TorchRL collector
- **AND** the resulting rollout is the input to `training_step`.

### Requirement: Algorithm runtime owns RL update semantics

The general RL Lightning module SHALL delegate algorithm-specific loss, replay, target, advantage, and update behavior to the selected algorithm runtime.

#### Scenario: PPO update is executed

- **WHEN** a PPO rollout reaches `training_step`
- **THEN** the PPO runtime computes advantage/loss and performs its configured minibatch updates
- **AND** the generic RL Lightning module contains no PPO-specific conditional branch.

### Requirement: RL evaluation uses episodes

After RL fitting, NexuML SHALL support deterministic episode evaluation in a fresh environment runtime and SHALL expose episode-level results through the existing training result surface.

#### Scenario: PPO run completes

- **WHEN** training finishes and `evaluation_episodes` is positive
- **THEN** NexuML evaluates the policy in a fresh environment
- **AND** reports mean episode return, return standard deviation, and mean episode length
- **AND** returns those metrics through the existing `TrainResult` structure.
