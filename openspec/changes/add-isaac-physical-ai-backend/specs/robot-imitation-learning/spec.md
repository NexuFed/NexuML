# Spec Delta

## Purpose

Train and evaluate maintained imitation policies and distilled students through NexuML's existing data, model, session and artifact lifecycle rather than a separate robotics trainer.

## ADDED Requirements

### Requirement: Offline imitation uses the existing model lifecycle

Behavior cloning and ACT SHALL train from validated episodes through ordinary NexuML configuration, data, optimizer, checkpoint and logging conventions. Pure offline training SHALL require neither simulation nor physical hardware. External model implementations SHALL be reused rather than copied wholesale.

#### Scenario: A tiny approved demonstration dataset is used
- **WHEN** an ACT training reference runs offline
- **THEN** a real loss and parameter update occur, with a resumable checkpoint and no ROS or simulator startup

### Requirement: Sequence losses respect task boundaries

Imitation learning SHALL preserve observation history, episode-disjoint sampling, future action chunks and validity masks. Dataset-derived normalization SHALL be fitted only on the permitted training split and retained for inference with the same action semantics.

#### Scenario: A batch contains padded action targets
- **WHEN** its loss is computed
- **THEN** invalid padded positions contribute no loss and no target comes from another episode

### Requirement: Chunked policies complete closed-loop round trips

An ACT reference SHALL demonstrate offline training, intermediate resume, complete export, clean inference and closed-loop evaluation in a matching Isaac task. Chunk buffering, execution stride and episode resets SHALL retain the trained observation/action meanings.

#### Scenario: The policy is replayed after export
- **WHEN** the same recorded input sequence and reset schedule are supplied
- **THEN** decoded actions match the reference sequence and subsequent simulator evaluation reports task success separately from return

### Requirement: Vision students replace unavailable teacher inputs

Teacher/student distillation SHALL train a student using real-hardware-available observations and teacher supervision from supported simulated tasks. Teacher weights, dataset identity, labels, trainable scope and privileged inputs SHALL be recorded; a student SHALL NOT receive hidden teacher-only observations during inference.

#### Scenario: The SO-101 state teacher supervises wrist-camera control
- **WHEN** the student is exported and evaluated with object ground truth blocked
- **THEN** it acts from the declared camera and joint inputs and reports held-out performance relative to the teacher

### Requirement: Corrective demonstrations support reproducible adaptation

Imitation updates SHALL accept authorized operator corrections with segment provenance and a declared trainable scope. Updated policies SHALL be compared on frozen old and new tasks before any promotion; naïve fine-tuning and a replay or regularization baseline SHALL be distinguished.

#### Scenario: New demonstrations improve one task but erase another
- **WHEN** candidate and reference policies are evaluated
- **THEN** the report exposes forgetting and does not promote solely because the new-task training loss improved

### Requirement: Demonstration and learned-success evidence remain separate

Synthetic generation, scripted control and upstream trained policies SHALL be attributed separately from NexuML learning. Functional smoke tests SHALL NOT imply convergence; an independent multi-seed quality experiment SHALL report sample counts, task success and uncertainty.

#### Scenario: A scripted trajectory succeeds before ACT training
- **WHEN** the integration report is assembled
- **THEN** that result validates task/data feasibility but does not satisfy learned-policy quality acceptance

### Requirement: Diffusion Policy has a complete comparable reference

A maintained Diffusion Policy SHALL train, resume, export, reload and complete closed-loop Isaac evaluation through NexuML. Its actual loss, sampler, normalization, denoising and chunk settings SHALL be retained. Comparison with ACT SHALL use matching approved data/splits, embodiment and I/O contracts. Upstream trainer execution or checkpoint download alone SHALL NOT satisfy this requirement.

#### Scenario: The Diffusion reference completes its lifecycle
- **WHEN** the approved demonstration task is trained and resumed through NexuML
- **THEN** actual trainable-parameter changes, restored optimizer/state and a complete portable inference package are verified
- **AND** fresh Isaac evaluation reports task success and held-out results alongside ACT under recorded comparable conditions

#### Scenario: Seeded diffusion inference is replayed after export
- **WHEN** the same observation sequence, reset schedule and sampling seed are supplied
- **THEN** the recorded inference scheduler, denoising parameters, processors and chunk execution reproduce reference actions within declared numerical tolerances
- **AND** an unseeded stochastic difference is not silently described as deterministic parity
