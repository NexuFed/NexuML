# Spec: scenario-learning-docs

## Purpose

Learning documentation covering the NexuML scenario concept, ScenarioSpec anatomy, and the distinction between registered scenarios and local scenario files.

## Requirements

### Requirement: Scenario Concept Page

The documentation SHALL include a beginner-oriented scenario concept page under the learning path.

#### Scenario: Beginner asks what a scenario is

- **WHEN** a beginner opens the scenario learning page
- **THEN** the page explains that a scenario declares data, pipeline, training, evaluation, logging, callbacks, tuning, checkpoint, and exports through `ScenarioSpec`.

#### Scenario: Beginner sees the lifecycle

- **WHEN** a beginner reads the scenario learning page
- **THEN** the page shows how a scenario becomes runnable through resolve, build, train, evaluate, and export.

### Requirement: ScenarioSpec Anatomy

The scenario learning page SHALL explain the main `ScenarioSpec` fields using implementation-current names from `src/nexuml/core/types.py`.

#### Scenario: User reads minimal scenario example

- **WHEN** the page shows a minimal scenario example
- **THEN** it uses current types and preferred concepts such as `ScenarioSpec`, `DataSpec`, `PipelineSpec`, `LayerSpec`, `TrainingSpec`, and `EvaluationSpec`.

#### Scenario: User reads layer explanation

- **WHEN** the page explains a `LayerSpec`
- **THEN** it explains that `type_key` resolves to a registered layer and `keys_in` / `keys_out` define TensorDict key contracts.

### Requirement: Registered Scenario Versus Scenario File

The documentation SHALL explain the difference between a registered scenario and a trusted Python scenario file.

#### Scenario: User wants reusable CLI scenario

- **WHEN** a user wants a scenario available by name to `nexuml resolve` and `nexuml train`
- **THEN** the docs route them to registered scenarios and `@scenario`.

#### Scenario: User wants local experiment file

- **WHEN** a user wants to run a local trusted Python file
- **THEN** the docs route them to `--scenario-file` and explain the required `scenario() -> ScenarioSpec` or implementation-current loading pattern.

### Requirement: Scenario Docs Link To Tasks And Reference

The scenario learning page SHALL link to task guides and generated reference instead of duplicating all details.

#### Scenario: User needs to implement a scenario

- **WHEN** a user finishes the scenario learning page
- **THEN** the docs link to define-scenario, trusted scenario-file, custom-scenario, decorators, and generated API reference pages.
