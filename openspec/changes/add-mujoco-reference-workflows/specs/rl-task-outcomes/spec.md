# Spec Delta

## Purpose

Report task success independently of return and task-specific distance metrics while preserving strict outcome validation, episode isolation and existing scalar evaluation behavior.

## ADDED Requirements

### Requirement: Success does not require a fabricated distance metric

Evaluation SHALL accept validated task success without joint error. Joint error SHALL be optional additional task data, not a prerequisite for success reporting. When both are supplied, their existing metric names and meanings SHALL remain unchanged; joint error without task success SHALL remain invalid.

#### Scenario: A manipulation task supplies success only
- **WHEN** every transition supplies one valid task-success value and no joint error
- **THEN** evaluation reports `eval/rl/task_success_rate` separately from return
- **AND** it emits no fabricated `eval/rl/final_joint_error_rad_mean`

#### Scenario: A joint-reaching task supplies both outcomes
- **WHEN** every transition supplies valid success and nonnegative finite joint error
- **THEN** success rate and final-joint-error mean retain their existing names and interpretation
- **AND** a task that requires both outcomes continues supplying and validating both at its provider boundary

### Requirement: Native success is validated before conversion

The environment boundary SHALL accept a native scalar boolean or finite numeric zero/one success flag, reject ambiguous/nonfinite/non-scalar values, and expose boolean tensor outcomes with the environment's batch/device semantics. Unrelated native information SHALL NOT become policy input or serialized policy state.

#### Scenario: A native benchmark provides numeric success
- **WHEN** a single environment provides numeric success equal to zero or one
- **THEN** the outcome becomes a boolean tensor without changing the native reward or action

#### Scenario: A native success flag is malformed
- **WHEN** a selected outcome is a string, non-scalar value, nonfinite number or numeric value other than zero/one
- **THEN** execution fails clearly rather than applying Python truthiness or replacing the value with failure

#### Scenario: Several environments provide outcomes
- **WHEN** supported collection batches multiple environment transitions
- **THEN** outcome rows retain their environment identity and device
- **AND** missing outcomes in only some rows fail rather than receiving invented defaults

### Requirement: Outcome presence and types remain stable

For one evaluation run, the supported outcome signature SHALL be consistently absent, success-only, or success plus joint error across all steps and episodes. Success SHALL be a scalar boolean tensor; supplied joint error SHALL be a scalar floating tensor with a finite nonnegative value. Malformed or disappearing outcomes SHALL fail visibly.

#### Scenario: A distance disappears after evaluation begins
- **WHEN** a run initially supplies success plus joint error and later omits joint error
- **THEN** evaluation fails instead of reusing stale error or downgrading to success-only reporting

#### Scenario: A supplied outcome is invalid
- **WHEN** success has the wrong dtype/shape, joint error is invalid, or joint error appears without success
- **THEN** evaluation raises a focused outcome error and closes its owned runtime

### Requirement: Success is accumulated per episode only

Episode success SHALL be the logical OR of valid success flags from that episode, counted at most once. Final joint error SHALL use that episode's final measured error when provided. Outcome accumulators SHALL reset between episodes and evaluation runs, without changing return/length/end statistics.

#### Scenario: A success signal is transient
- **WHEN** the first episode succeeds at an intermediate step and the next episode never succeeds
- **THEN** the two-episode success rate is one half regardless of the first episode's final success flag
- **AND** distance aggregation, when enabled, uses each episode's own final error

### Requirement: Tasks without outcomes retain their behavior

Environments that supply neither supported outcome SHALL retain their existing evaluation metrics, RNG/mode isolation, scheduled evaluation, native logging/media and cleanup behavior. Task results SHALL use the existing evaluation result surface rather than a second metric service or public result hierarchy.

#### Scenario: An existing classic-control or MuJoCo task has no success signal
- **WHEN** its scalar or recorded evaluation runs
- **THEN** existing return, length and termination/truncation results remain unchanged with no invented success metric
- **AND** existing logging, scheduling and primary-failure preservation remain intact
