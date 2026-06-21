# Spec: experiment-program

## Purpose

TBD — defines constraints and validation rules for the DCASE 2026 Task 2 submission experiment program.

## Requirements

### Requirement: Experiment program scope

The experiment program SHALL be limited to the submission-critical path for
DCASE 2026 Task 2: pruned detector sweep, view sweep, config freeze by
official dev Ω, eval-set fitting, export dry run, and packaging.

#### Scenario: Sweep runs on cached embeddings

- **WHEN** a detector sweep run executes for a (backbone, view) pair
- **THEN** embeddings SHALL be read from the feature cache and the run SHALL
  NOT recompute backbone forward passes

### Requirement: Submission compliance

Every submitted system SHALL use only resources on the official allowed
external resources list (locked 2026-06-01).

#### Scenario: Dev-only oracle excluded

- **WHEN** the submission configs are frozen
- **THEN** no frozen config SHALL reference
  MIT/ast-finetuned-audioset-10-10-0.4593

### Requirement: Eval export validation

The exporter SHALL validate that each (machine, section) score file contains
exactly the expected number of rows (200 unlabeled clips per section on the
2026 eval set) and SHALL fail the export on mismatch.

#### Scenario: Row-count mismatch

- **WHEN** an export produces a CSV whose row count differs from the
  expected clip count
- **THEN** the export SHALL abort with an error naming the machine and
  section

### Requirement: Decision thresholds

Each submitted system SHALL produce decision_result CSVs using a gamma
distribution fitted on the training anomaly scores, with the 90th percentile
as the decision threshold.

#### Scenario: Threshold derivation

- **WHEN** anomaly scores for a machine's training data are available
- **THEN** the decision threshold SHALL be the 90th percentile of the fitted
  gamma distribution
