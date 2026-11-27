# Spec Delta

## Purpose

Provide reproducible, audited discrete and bounded-continuous RL references through NexuML's ordinary workflow, with honest evidence and unchanged supervised and deployment boundaries.

## ADDED Requirements

### Requirement: Foundation verification records current evidence

The project SHALL publish a baseline and gap report before implementing new evaluation behavior. The report SHALL distinguish current execution evidence, historical results, missing prerequisites, and deferred capabilities. Unrelated blocked integrations SHALL NOT be represented as completed.

#### Scenario: Audit establishes the baseline
- **WHEN** Phase 0 verification runs in the target worktree
- **THEN** `reports/physical-ai/phase-0-audit.md` records the commit, import origins, Python/platform, installed dependencies/extras, exact commands, durations, pass/fail/skip counts, and unresolved failures
- **AND** it covers contract/configuration round trips, data/interaction coexistence, policy inference/export, checkpoints, collection, evaluation, and existing regression/static/documentation gates

#### Scenario: Capability coverage is classified
- **WHEN** the audit assesses Gymnasium, MuJoCo, ROS/Gazebo, ordinary and demonstration datasets, IL, VLA, Isaac, benchmarks, export, and evaluation/media
- **THEN** each assessment uses present-and-tested, present-but-not-system-tested, missing, or deferred, with supporting evidence
- **AND** temporal/multimodal gaps are recorded without implementing later roadmap phases
- **AND** the original live ROS/Gazebo and CARLA tasks remain open until separately proved

### Requirement: Existing references use the ordinary user workflow

The existing `cartpole-ppo` and `pendulum-ppo` references SHALL support discovery, YAML resolution/restoration, ordinary CLI overrides, training, evaluation, checkpointing, export, and reload without user source edits. They SHALL configure native TensorBoard logging by default without enabling rendering, periodic evaluation, or MLflow by default.

#### Scenario: Each reference executes a CPU smoke
- **WHEN** either reference runs with a tiny finite frame budget and short evaluation episodes through the normal CLI/session path
- **THEN** rollout collection and at least one real optimizer update complete, actor parameters change, and losses and episode metrics are finite
- **AND** resolved YAML and overrides preserve the scenario's discrete or bounded-continuous action semantics
- **AND** ordinary CI does not require a stochastic reward-convergence threshold

#### Scenario: Users select their native loggers
- **WHEN** a reference's ordinary logging configuration selects MLflow-only, TensorBoard-only, both, or disables logging
- **THEN** that configuration controls the run's logging without a special RL-only logging path
- **AND** media remains opt-in regardless of the selected logger

### Requirement: Reference transitions and budgets remain explicit

Both references SHALL preserve finite collected-frame budgets, valid action bounds/categories, seeded CPU reproducibility, reset semantics, and separate termination/truncation signals. Backend batch-boundary rounding SHALL be documented rather than concealed.

#### Scenario: Discrete and continuous episodes end
- **WHEN** CartPole terminates or Pendulum reaches its time limit
- **THEN** evaluation reports the corresponding terminal or truncated outcome and valid episode length
- **AND** subsequent episodes begin from an explicit reset

#### Scenario: Collection uses multiple environments
- **WHEN** a supported direct/process topology collects vectorized rollouts
- **THEN** batch dimensions remain distinct from individual observation/action shapes
- **AND** the run stops at the finite budget subject to documented batch rounding

### Requirement: Resume and deployment remain separate verified paths

Reference verification SHALL prove interrupted training resume and complete-policy inference round trips. Resume SHALL restore learner progress and training state while rebuilding live resources. Portable inference SHALL NOT require a learner, collector, simulator, recording encoder, or training-only state.

#### Scenario: Partial training resumes
- **WHEN** either reference resumes a checkpoint taken before its frame budget completes
- **THEN** policy, critic, optimizer state, and frame/update counters are restored and the remaining budget advances
- **AND** environments/collectors are rebuilt rather than deserialized as live objects

#### Scenario: An exported policy is loaded for inference
- **WHEN** the policy package is reloaded in a clean process with learning, simulation, and recording dependencies unavailable
- **THEN** deterministic actions match the pre-export actions for identical observations
- **AND** the ordinary inference artifact excludes critic/optimizer/rollout/environment state

### Requirement: Supported collector ownership remains verified

The reference workflow SHALL retain direct and synchronous-process collection, copied-policy synchronization, explicit device/topology restrictions, and cleanup on completion, failure, and interruption. It SHALL NOT enable new distributed or asynchronous topologies as part of verification.

#### Scenario: Process workers own policy copies
- **WHEN** a supported copied-policy collector reaches its synchronization boundary after learning
- **THEN** subsequent collection uses the refreshed policy
- **AND** collection does not eagerly request a synchronous rollout before the previous update/synchronization completes

#### Scenario: A run fails or is cancelled
- **WHEN** learner execution raises or is interrupted
- **THEN** owned collectors and environments close without leaving owned worker processes running
- **AND** unsupported device/topology combinations still fail clearly

### Requirement: Verification stays scoped and reproducible

The workflow SHALL preserve supervised regressions, optional dependency boundaries, and existing test tiers. Documentation SHALL provide exact installation, train, resume, export, evaluation/replay, expected-output, and limitation guidance. Heavy integrations SHALL remain explicit opt-ins.

#### Scenario: Ordinary CPU verification runs
- **WHEN** the fast CPU suite runs without rendering, GPU, datasets, middleware, or external tracking services
- **THEN** structural and scalar-evaluation coverage remains runnable using synthetic inputs or local fakes
- **AND** optional skips are listed separately rather than counted as capability proofs

#### Scenario: Completion is reported
- **WHEN** Phases 0–1 are declared verified
- **THEN** the report records focused/regression tests, lock/lint/format/type/docs/OpenSpec/package gates, and opt-in rendering/tracking evidence
- **AND** command examples identify actual checkpoint outputs instead of assuming a fixed version directory
- **AND** unresolved unrelated system prerequisites remain explicitly deferred
