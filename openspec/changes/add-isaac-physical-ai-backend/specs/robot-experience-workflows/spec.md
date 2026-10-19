# Spec Delta

## Purpose

Make real, simulated and synthetic robot experience usable by existing NexuML data and learning workflows with validated timing, episode boundaries and traceable provenance.

## ADDED Requirements

### Requirement: Maintained formats are consumed without a new storage engine

The backend SHALL ingest declared versions of LeRobot datasets, ROS MCAP recordings and Isaac demonstration exports through maintained readers or explicit converters. Conversion SHALL preserve observations, actions, episode identity, timestamps and relevant metadata without replacing general NexuML dataset support.

#### Scenario: A supported Mimic HDF5 recording is converted
- **WHEN** it is mapped to the selected LeRobot episode schema
- **THEN** episode counts, ordering, action units, normalization inputs and sampled frame/action parity are validated
- **AND** unsupported schema versions fail clearly

### Requirement: Experience carries physical and production provenance

Experience SHALL identify simulator or hardware origin, task and embodiment, policy/controller revision, calibration, processors, outcome and intervention segments. License, permitted use and retention references SHALL be retained without collecting unauthorized personal identifiers or embedding raw recordings in policy packages.

#### Scenario: A corrective segment follows an autonomous attempt
- **WHEN** the episode is ingested
- **THEN** policy-generated and operator-corrected actions retain their producer and timing boundaries

### Requirement: Multi-rate data is validated before training

Dataset validation SHALL detect missing or stale frames, nonmonotonic timestamps, calibration changes, inconsistent rates, invalid action bounds, duplicates and broken episode fragments. Alignment, resampling, dropped-sample and validity-mask policies SHALL be recorded rather than silently repairing data.

#### Scenario: Camera frames lag the declared control timeline
- **WHEN** the recorded lag exceeds the dataset contract
- **THEN** validation rejects or explicitly quarantines those samples and reports their counts before learning

### Requirement: Demonstration capture reuses upstream operator tools

SO-101 leader/follower capture and supported Isaac input devices SHALL use maintained teleoperation, retargeting and recording tools. Simulation-only capture SHALL be distinct from physical actuation. XR or specialized devices SHALL require explicit prerequisites and SHALL NOT become mandatory for offline ingestion.

#### Scenario: No XR device is available
- **WHEN** approved prerecorded demonstrations are selected
- **THEN** ingestion, validation and offline training remain possible without pretending live XR capture was verified

### Requirement: Synthetic generation preserves annotation and split identity

Mimic and supported synthetic sensor generation SHALL record source demonstrations, annotations, generator/configuration, asset versions and seeds. Derived episodes SHALL remain grouped with their sources for leakage-aware splits; generation success SHALL be distinguished from trained-policy success.

#### Scenario: Several generated episodes derive from one demonstration
- **WHEN** train and held-out sets are created
- **THEN** the configured source-group split prevents those derivatives from leaking across evaluation boundaries

### Requirement: Training records what experience was actually consumed

Every demonstration or replay-backed training run SHALL record dataset/content identities, split, quality exclusions, selected episode/window counts and sampling configuration. Local replay, failure selection and unlabeled-stream use SHALL remain reproducible without automatic external uploads.

#### Scenario: Failure cases are prioritized for adaptation
- **WHEN** a fixed labeling and compute budget is used
- **THEN** selected episode identities are retained and compared against a matched random-selection baseline

### Requirement: Non-episodic industrial streams remain supported

Industrial time-series and image streams SHALL retain their existing window/data contracts without invented robot actions or episode boundaries. Read-only machine ingestion SHALL validate units, time bases and missing data independently of ROS or Isaac installation.

#### Scenario: A machine anomaly dataset has no control actions
- **WHEN** it is loaded and used for prediction training
- **THEN** the ordinary dataset workflow operates without requiring a robot policy or simulator
