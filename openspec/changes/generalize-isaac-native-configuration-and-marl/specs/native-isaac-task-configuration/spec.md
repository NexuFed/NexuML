# Spec Delta

## Purpose

Configure trusted native Isaac tasks through portable NexuML definitions while preserving one execution owner and explicit physical contracts.

## ADDED Requirements

### Requirement: NC1 Typed customization is portable and trusted

Native task customization SHALL use importable typed functions with JSON-safe arguments, receiving the resolved native config and optionally replacing it. Static discovery, description and YAML restoration SHALL NOT execute factories, import native runtimes or allocate CUDA. Execution SHALL require trusted operator-installed components; remote requests SHALL NOT select executable targets.

#### Scenario: Configuration survives a static round trip
- **WHEN** a built-in or trusted custom definition is serialized/restored with native imports and CUDA allocation blocked
- **THEN** its function identity, arguments, explicit contract and bindings survive without executing the function or creating native resources

#### Scenario: Invalid or unauthorized function selection
- **WHEN** customization uses a lambda, closure, incompatible signature, nonportable object or remote executable target
- **THEN** validation rejects it before native execution instead of treating YAML as an authorization mechanism

### Requirement: NC2 Registered tasks and native presets retain their identity

The integration SHALL resolve built-in and installed custom task registrations and native presets through supported Isaac mechanisms. It SHALL validate actual workflow, task, tensors, controls, sensors, count and device against explicit contracts/bindings. Unknown or incompatible selections SHALL fail without silent defaults. General configuration SHALL NOT alter noise, history or replay distributions implicitly.

#### Scenario: A registered custom task replaces its resolved config
- **WHEN** a trusted typed function returns a modified native configuration with matching explicit tensor/control semantics
- **THEN** the actual registered task uses those values and records resolved task/preset/config identities through normal CLI execution

#### Scenario: Customization conflicts with execution ownership
- **WHEN** a function changes protected device/count, disables required final observations, allocates its own runtime or selects an unknown preset
- **THEN** the job rejects the conflict before native stepping and releases any owned partial resources

### Requirement: NC3 The whole job owns native lifetime

Normal train/evaluate commands and an explicit whole-job Python scope SHALL own native initialization and teardown. Effective resume/evaluation definitions SHALL determine ownership. Checkpoint/export/provenance and logger finalization SHALL precede native termination; failures and cancellation statuses SHALL survive cleanup. Same-job scopes SHALL join safely; conflicting runtimes SHALL reject.

#### Scenario: Training fails or is cancelled
- **WHEN** an owned native job exits through failure or cancellation
- **THEN** output/logger cleanup runs before native fast-close, owned resources stop, and the original failure or signal-derived status is retained

#### Scenario: Replay or completed-budget resume is selected
- **WHEN** evaluation uses an explicit evaluation environment or resume has no remaining collection budget
- **THEN** replay scopes that selected environment in a fresh process, while completed-budget resume allocates no simulator
- **AND** unsupported conflicting in-fit evaluation rejects rather than creating a second native context

### Requirement: NC4 Native persistence has one current schema

Repository-owned unmerged Physical AI definitions, scenarios and fixtures SHALL use one current versioned native configuration schema. Superseded implementation-specific branches SHALL be removed without legacy readers, converters or parallel native representations. Historical evidence/user data SHALL remain untouched. Unrelated published general, supervised and export contracts SHALL remain supported.

#### Scenario: Current artifacts are persisted and resumed
- **WHEN** this implementation saves YAML, checkpoints and complete policies
- **THEN** current-schema restore/resume/replay preserves configuration and meaning without a second runtime representation or hypothetical historical-checkpoint migration path

#### Scenario: An ordinary supervised scenario runs
- **WHEN** an existing non-robot scenario compiles, trains and exports after the change
- **THEN** it retains its published interfaces and export behavior without simulation, robot contracts or MARL prerequisites

### Requirement: NC5 Native reference compatibility requires execution proof

Native configuration/lifecycle acceptance SHALL include existing manager/direct Cartpole, Franka, SO-101 and one genuinely registered custom task. Each claimed supported path SHALL demonstrate actual native config/transition correctness, NexuML optimizer updates, cleanup and complete export/replay. Registration, serialization, mock tests and upstream-runner success SHALL NOT establish NexuML native acceptance.

#### Scenario: A custom task only passes a mocked adapter test
- **WHEN** no real registered native job has produced verified transitions and NexuML updates
- **THEN** its native acceptance remains open even if discovery and YAML tests pass

### Requirement: NC6 Native qualification respects the pinned resource boundary

This change SHALL use the approved worktree environment and pinned Lab/Sim/TorchRL/PyTorch-CUDA selections. Native work SHALL monitor free space and stop owned work at 4 GiB free, using bounded resources. It SHALL NOT alter `/env`, drivers or containers, install replacements, download large pretrained models, expose public listeners, use unauthorized paid/cloud compute, actuate hardware or interfere with other jobs/worktrees.

#### Scenario: Required native prerequisites are absent
- **WHEN** disk falls below the minimum or graphics/assets/runtime prerequisites are unavailable
- **THEN** dependent execution stops with the exact blocker and leaves acceptance unverified instead of provisioning infrastructure, disturbing other jobs or claiming a skipped gate passed
