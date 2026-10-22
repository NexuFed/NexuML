# Spec Delta

## MODIFIED Requirements

### Requirement: Scenario Concept Page
The documentation SHALL explain scenarios in an accessible section of the canonical architecture path, with one linked create/run procedure. It SHALL NOT require a separate duplicated scenario learning article before users can run an experiment.

#### Scenario: Beginner asks what a scenario is
- **WHEN** a beginner opens the scenario explanation
- **THEN** it explains that a Python recipe composes data, pipeline, training, evaluation, logging, callbacks, tuning, checkpoint policy, exports, and execution through `ScenarioSpec`

#### Scenario: Beginner sees the lifecycle
- **WHEN** a beginner reads the scenario explanation
- **THEN** it shows how scenario composition leads to resolved configuration, a built pipeline, training/evaluation, and export
- **AND** it identifies the actual CLI commands without suggesting a nonexistent standalone evaluation command

### Requirement: ScenarioSpec Anatomy
The canonical scenario explanation SHALL use implementation-current fields from `ScenarioSpec` and distinguish component configuration from pipeline/data/evaluation routing. Exact field contracts SHALL be linked to reference rather than duplicated into multiple conceptual pages.

#### Scenario: User reads minimal scenario example
- **WHEN** the explanation shows a minimal scenario example
- **THEN** it uses current types and concepts such as `ScenarioSpec`, `DataSpec`, `PipelineSpec`, `LayerSpec`, `TrainingSpec`, and `EvaluationSpec`
- **AND** the example is consistent with the linked runnable create/run guide

#### Scenario: User reads layer explanation
- **WHEN** the explanation introduces a `LayerSpec`
- **THEN** it explains that `component` contains a typed layer definition and `keys_in` / `keys_out` define TensorDict key contracts

### Requirement: Registered Scenario Versus Scenario File
The canonical create/run guide SHALL distinguish registered recipes, resolved YAML, and trusted Python scenario files, including which commands accept each source. It SHALL preserve the trusted-code warning and provenance instructions beside file-execution commands.

#### Scenario: User wants reusable CLI scenario
- **WHEN** a user wants a scenario available by name to `nexuml resolve` and `nexuml train`
- **THEN** the guide explains registered recipes, `@scenario`, and the required installed-library or local-root discovery path

#### Scenario: User wants local experiment file
- **WHEN** a user wants to run a local trusted Python file
- **THEN** the guide explains `--scenario-file` and the implementation-current `scenario() -> ScenarioSpec` loading contract
- **AND** it states that the file executes trusted Python and explains optional provenance capture

#### Scenario: User wants persisted configuration
- **WHEN** a user chooses resolved YAML
- **THEN** the guide explains how to resolve, inspect, build, and train that configuration
- **AND** it does not imply that every CLI command accepts YAML or trusted files interchangeably

### Requirement: Scenario Docs Link To Tasks And Reference
The scenario explanation SHALL link to canonical create/run, trusted-file, library/extension, and generated API reference destinations rather than duplicate their full procedures or route readers through obsolete placeholder articles.

#### Scenario: User needs to implement a scenario
- **WHEN** a user finishes the scenario explanation
- **THEN** they can reach the current create/run procedure, trusted-file guidance, library extension instructions, decorator reference, and generated API
- **AND** older scenario-learning and walkthrough URLs map to the appropriate canonical destination
