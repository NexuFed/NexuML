# Spec Delta

## Purpose

Let users compare policy behavior during finite-frame training through reproducible episodic evaluations without perturbing learning or changing the default session lifecycle.

## ADDED Requirements

### Requirement: Evaluation scheduling uses ordinary typed configuration

`training.evaluation` SHALL retain its episode count, policy mode, seed, and optional episode-step limit, and add `every_n_frames=None` and `at_start=False`. An interval, when supplied, SHALL be a positive integer. The configuration SHALL round-trip through YAML and checkpoints and support ordinary nested overrides.

#### Scenario: Existing configuration omits scheduling
- **WHEN** an existing RL scenario omits initial/periodic/media evaluation options
- **THEN** plain fit does not gain unconditional evaluation and the ordinary run still performs final evaluation
- **AND** existing scalar metric/result keys remain available

#### Scenario: Invalid schedule is supplied
- **WHEN** an interval is zero/negative or an existing episode/step-limit constraint is violated
- **THEN** configuration fails with a field-specific diagnostic before a long run begins

#### Scenario: Evaluation is disabled
- **WHEN** the evaluation episode count is zero and recording is disabled
- **THEN** no evaluation runtime, episode loop, or recording encoder is used
- **AND** evaluation returns an empty scalar result

### Requirement: Periodic evaluation is driven by collected frames

Enabled periodic evaluation SHALL run after a completed rollout update and its due policy synchronization, before the next synchronous rollout request. Crossing an interval SHALL trigger at most one evaluation of the current policy per rollout batch. Event steps SHALL be actual collected-frame counts, not epochs or optimizer steps.

#### Scenario: A batch crosses an interval
- **WHEN** a completed update advances collected frames from 16 to 32 with an interval of 20
- **THEN** one evaluation occurs at frame 32
- **AND** it is not skipped because 32 is not an exact multiple of 20

#### Scenario: One batch crosses several intervals
- **WHEN** one rollout advances past several evaluation thresholds
- **THEN** the current policy is evaluated once at the actual frame count
- **AND** missed thresholds are not replayed or assigned fabricated intermediate weights

### Requirement: Initial evaluation respects restored progress

When `at_start=True`, evaluation SHALL run before the first learner update only if restored collected frames are zero. Resume SHALL derive future cadence from restored frame progress, without replaying historical evaluations or relabeling resumed weights as an untrained baseline.

#### Scenario: A fresh run requests a baseline
- **WHEN** training starts with zero collected frames and initial evaluation enabled
- **THEN** the unupdated policy is evaluated and reported at frame zero before the first optimizer update

#### Scenario: A partially trained run resumes
- **WHEN** a checkpoint restores frame 32 with an interval of 20 and initial evaluation enabled
- **THEN** no frame-zero or frame-20 baseline is emitted
- **AND** the next interval is 40, reported at the actual batch boundary that crosses it

### Requirement: Final evaluation is emitted once per final policy state

With initial/periodic/media evaluation enabled and episodes positive, final evaluation SHALL complete while configured loggers are active. The normal run SHALL reuse results already corresponding to the final policy/configuration/frame state instead of duplicating final episodes, metrics, or recordings.

#### Scenario: Final frames were just evaluated periodically
- **WHEN** the last periodic evaluation already measured the final unchanged policy at the final frame count
- **THEN** final result construction reuses that evaluation
- **AND** it does not emit duplicate media or scalar points for that evaluation

#### Scenario: Final frames are between intervals
- **WHEN** training completes after its most recent evaluation but before the next interval
- **THEN** one fresh final evaluation measures the final policy at its actual frame count
- **AND** final metrics are available through the existing result dictionary

### Requirement: Evaluation owns a separate compatible environment

Every evaluation SHALL use a fresh, single-environment runtime from the configured evaluation override or training definition. It SHALL validate required observation/action semantics before rollout and reject incompatible or unsupported batched evaluation. It SHALL NOT reset, step, or render the training collector's environment.

#### Scenario: Training uses vectorized collection
- **WHEN** periodic evaluation runs beside supported vectorized training
- **THEN** evaluation executes complete episodes in its own single environment
- **AND** training runtime state, collected-frame budgets, and synchronization ordering are unchanged

#### Scenario: Evaluation configuration is incompatible
- **WHEN** evaluation exposes incompatible required fields, shapes, dtypes, action categories/bounds, or an unsupported environment batch
- **THEN** evaluation fails with a focused contract diagnostic instead of silently combining trajectories or reinterpreting controls
- **AND** any owned evaluation runtime is closed

### Requirement: Evaluation preserves learner state and random streams

Evaluation SHALL use the complete policy in inference mode without constructing or updating a learner. It SHALL preserve parameters, gradients, module training flags, and Python/NumPy/PyTorch random streams, including relevant accelerator state. Explicit evaluation seeds SHALL provide repeatable resets and stochastic action sampling independently of learner RNG use.

#### Scenario: Fixed seeds compare policy stages
- **WHEN** an evaluation seed is configured, or falls back to the configured training seed
- **THEN** evaluation episode resets use that seed plus the episode index at every evaluation point
- **AND** the same unchanged policy and stochastic-mode configuration produce reproducible evaluations under the supported deterministic CPU setup

#### Scenario: Training continues after evaluation
- **WHEN** deterministic or stochastic evaluation completes
- **THEN** subsequent learner random draws and all prior module training flags match their pre-evaluation state
- **AND** learner parameters and gradients were not changed by evaluation

### Requirement: Metrics cover complete episodes independently of recording

Evaluation SHALL report `eval/rl/return_mean`, `return_std`, `length_mean`, `terminated`, and `truncated` over all configured episodes. Environment terminal/truncated outcomes and evaluator step limits SHALL remain explicit. Reaching a recording frame cap SHALL NOT stop episode evaluation or exclude later episodes from metrics.

#### Scenario: Capture ends before evaluation
- **WHEN** the first episode exceeds the recording cap and additional episodes are configured
- **THEN** the evaluator continues the full episode/horizon and subsequent episodes for scalar metrics
- **AND** only recording stops at the cap

#### Scenario: The evaluator horizon ends an episode
- **WHEN** the configured step limit is reached without an environment terminal outcome
- **THEN** the episode contributes its complete observed return/length and a truncation
- **AND** an environment terminal outcome at the same step is not converted into a horizon-only truncation

### Requirement: Failure and cancellation restore owned state

Evaluation SHALL close owned resources and restore policy modes and RNG state on success, policy/environment/media failure, or cancellation. Cleanup SHALL preserve the primary failure rather than hiding it. Explicit recording failures SHALL NOT be reported as successful scalar-only evaluations.

#### Scenario: Rendering or policy execution fails
- **WHEN** an exception occurs after evaluation resources or state guards are acquired
- **THEN** the owned runtime closes and policy/RNG state is restored
- **AND** the original failure remains visible, with any cleanup failure reported without replacing it

#### Scenario: Evaluation is cancelled
- **WHEN** evaluation is interrupted before its episodes or encoding finish
- **THEN** owned evaluation resources are released and learner state is restored
- **AND** incomplete media is not presented as a completed recording
