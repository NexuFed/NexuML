# Spec Delta

## Purpose

Allow users to evaluate and record trusted exported policy packages without starting training, using NexuML's existing configuration, inference, and experiment logging conventions.

## ADDED Requirements

### Requirement: Exported policies have an inference-only evaluation command

`nexuml evaluate <policy-package>` SHALL load a complete exported policy and its embedded resolved scenario through the existing inference-package path and run the shared episodic evaluator. It SHALL NOT construct a training session, learner, optimizer, or collector, or initiate training or scenario exports.

#### Scenario: A complete policy package is evaluated
- **WHEN** users evaluate a trusted CartPole or Pendulum policy package
- **THEN** fresh compatible evaluation episodes execute using that policy's action semantics
- **AND** scalar results are reported without constructing a PPO learner or loading optimizer/critic state

#### Scenario: A non-policy or incompatible artifact is supplied
- **WHEN** the input is not a complete policy package, lacks required evaluation configuration, or describes incompatible evaluation I/O
- **THEN** the command exits nonzero with a focused diagnostic
- **AND** it does not silently train, invent observation/action semantics, or replace the packaged policy

### Requirement: Replay uses ordinary evaluation and logging overrides

The command SHALL accept repeatable `-O`/`--override key.path=value` using existing coercion, validation, ordering, and override-summary conventions. Evaluation seeds/counts/horizons, media format/limits, and native logging SHALL be configurable without a second configuration syntax. Overrides SHALL NOT silently replace packaged executable policy state.

#### Scenario: Users request a seeded MP4 replay
- **WHEN** `nexuml evaluate <policy-package> -O training.evaluation.seed=7 -O training.evaluation.video.enabled=true -O training.evaluation.video.format=mp4` runs with the required prerequisites
- **THEN** the shared evaluator executes the requested seeded evaluation and publishes MP4/JSON through the selected native logging backends
- **AND** overrides are shown with key, old value, and new value

#### Scenario: An invalid override is supplied
- **WHEN** an override path/value is invalid or attempts an unsupported replacement of packaged policy state
- **THEN** the command identifies the problem and exits nonzero before evaluation

#### Scenario: Replay is scalar-only
- **WHEN** recording is disabled and the command evaluates a compatible environment
- **THEN** scalar evaluation does not require or import GIF/MP4 encoders

### Requirement: Replay owns a separate traceable evaluation run

Standalone evaluation SHALL use native logger construction/finalization to create an evaluation run with source-package provenance. It SHALL use an explicitly recorded source training-frame count when available, otherwise a documented standalone step of zero. Optimizer global steps SHALL NOT be fabricated into training-frame counts.

#### Scenario: Source training frames are unavailable
- **WHEN** a package contains optimizer-step provenance but no explicit collected-training-frame value
- **THEN** the standalone evaluation uses event step zero and records the source-package identity and absence of a training-frame value
- **AND** it does not label that evaluation as an untrained policy baseline

#### Scenario: Native logging selects MLflow-only replay
- **WHEN** the command's native logging configuration enables MLflow but not TensorBoard
- **THEN** selected media/JSON and scalar metrics are saved in its native evaluation MLflow run with source-package provenance
- **AND** no TensorBoard writer or additional media-specific tracking run is required

### Requirement: Replay preserves trusted-package and cleanup boundaries

Documentation SHALL distinguish trusted executable policy packages from passive data, policy replay from checkpoint resume, and encoder/simulator prerequisites from inference requirements. The command SHALL close owned evaluation/encoding resources and finalize configured loggers on success, failure, or cancellation without hiding the primary error.

#### Scenario: Replay fails during evaluation or encoding
- **WHEN** policy execution, environment stepping, media encoding, or artifact delivery fails
- **THEN** the command exits with a visible failure and releases owned resources
- **AND** no incomplete media is presented as successfully exported

#### Scenario: A user wants to continue training
- **WHEN** users consult the replay/checkpoint instructions
- **THEN** they are directed to the existing `train --trainer-checkpoint` path for resume and the evaluation command for inference-only replay
- **AND** installation and trust limitations are explicit
