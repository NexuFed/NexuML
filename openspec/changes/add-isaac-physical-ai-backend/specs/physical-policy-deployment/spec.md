# Spec Delta

## Purpose

Move complete policies from simulation to explicitly approved runtimes and supervised hardware trials, then improve them locally through validated offline experience and controlled release decisions.

## ADDED Requirements

### Requirement: Deployment verifies complete reference parity first

Every deployment target SHALL reproduce reference preprocessing, state updates and decoded actions on representative recorded sequences before closed-loop use. Device, memory, sensor availability, calibration and rate requirements SHALL be checked; successful conversion or model loading alone SHALL NOT establish deployability.

#### Scenario: The target changes normalization or joint order
- **WHEN** parity is checked before enabling commands
- **THEN** deployment is blocked with the semantic mismatch identified

### Requirement: Sim-to-real is an operator-supervised arm workflow

The first physical proof SHALL use one approved low-cost arm with verified firmware, calibration, controller interface, protective workspace, independent stop mechanism and human intervention. Hardware enablement SHALL require explicit operator action and SHALL NOT run through normal CI or unattended exploration.

#### Scenario: The hardware enable gate is absent
- **WHEN** a simulation-validated policy is loaded on a connected robot workstation
- **THEN** inference or shadow evaluation can run without enabling motion and no physical command is published automatically

### Requirement: Physical trials have measurable go and no-go conditions

Approved trials SHALL begin at low speed in a bounded workspace and record success, tracking, interventions, delays and safety-trigger behavior. Sensor/controller failures SHALL use independently validated stop/fallback behavior. A model score SHALL NOT override an operator stop or external safety control.

#### Scenario: The operator intervenes during a trial
- **WHEN** physical execution is stopped
- **THEN** policy commands cease through the validated controller boundary and the intervention and associated policy revision are recorded

### Requirement: The sim-real gap is measured on matched cases

Simulation and supervised physical trials SHALL use traceable matched tasks and record differences in sensing, calibration, contacts, dynamics and delays. System-identification or randomization updates SHALL retain the source traces and parameter provenance without claiming simulation robustness guarantees physical safety.

#### Scenario: A controller responds differently on hardware
- **WHEN** matched simulation and recorded physical responses are compared
- **THEN** the discrepancy and affected parameters are reported before further policy promotion

### Requirement: Local adaptation has precise inputs and outputs

An adaptation operation SHALL accept an approved policy identity, authorized local dataset reference, trainable scope, learning budget and frozen evaluation suite. It SHALL produce a candidate artifact and reviewable old/new-task evidence with deterministic failure/result handling and no automatic upload or fleet update.

#### Scenario: An approved correction dataset is selected
- **WHEN** offline retraining completes
- **THEN** the candidate records its parent, consumed data, trainable parameters, compute budget and evaluation decision

### Requirement: Continual improvement measures selection and forgetting

The local loop SHALL support validated corrective experience, failure/novelty selection and a replay or regularization comparison when testing adaptation. Equal labeling/compute budgets and disjoint old/new evaluation splits SHALL distinguish genuine improvement from leakage or additional data consumption.

#### Scenario: Targeted failure selection is compared with random selection
- **WHEN** both candidates are trained under the same budget
- **THEN** new-task benefit and retained old-task performance are reported for both rather than presuming targeted selection wins

### Requirement: Promotion and rollback remain human controlled

A candidate SHALL require an explicit pass/fail/insufficient-evidence report and operator approval before another physical trial. A known reference artifact and manual rollback procedure SHALL remain available; policies SHALL NOT hot-swap during an active physical episode.

#### Scenario: A candidate fails an old-task regression
- **WHEN** promotion is considered
- **THEN** the previous approved policy is retained and the failed candidate remains traceable without replacing the deployed version

### Requirement: Full completion cannot borrow simulation evidence for hardware

The full program SHALL retain a separate manual hardware evidence gate. Simulation-only milestone completion SHALL be useful and reportable but SHALL NOT mark physical execution, field-data ingestion or supervised redeployment as complete.

#### Scenario: No approved robot is available
- **WHEN** program status is reported
- **THEN** the hardware milestone stays blocked with exact prerequisites while completed software evidence remains intact
