# Spec Delta

## RENAMED Requirements

- FROM: `### Requirement: CIFAR ResNet First Success`
- TO: `### Requirement: Synthetic CPU First Success`

## MODIFIED Requirements

### Requirement: Synthetic CPU First Success
The documentation SHALL use the existing `synthetic-linear-ae-reconstruction` recipe for the canonical first successful run, explicitly selecting one epoch on CPU with locally generated data. The default walkthrough SHALL not require a GPU, dataset download, source checkout, or optional training/data backend.

#### Scenario: Beginner follows first-run tutorial
- **WHEN** a beginner follows the first-run tutorial after normal-user installation with the base library
- **THEN** its commands discover, resolve, build, and run the same synthetic recipe for one CPU epoch
- **AND** the page explains how to inspect the resulting metrics and outputs before changing the scenario

#### Scenario: Beginner lists scenarios
- **WHEN** a beginner runs `nexuml registry list scenarios`
- **THEN** the tutorial identifies `synthetic-linear-ae-reconstruction` as the first-run recipe available with the installed base library

#### Scenario: User wants a realistic vision dataset example
- **WHEN** a reader chooses the optional CIFAR example
- **THEN** it remains reachable with its additional dataset/setup prerequisites stated
- **AND** those prerequisites do not block the default synthetic first run

### Requirement: First-Run Tutorial Explains Each Command
The first-run tutorial SHALL explain each command's purpose, expected artifacts, and introduced concept using verified implementation-current syntax. It SHALL distinguish the registered recipe key, resolved configuration path, and actual training outputs.

#### Scenario: User resolves the scenario
- **WHEN** the tutorial resolves the synthetic recipe to an explicit configuration destination
- **THEN** it explains that the Python recipe produces validated, reproducible configuration
- **AND** the stated YAML path agrees with the command and actual output rather than an assumed recipe-name transformation

#### Scenario: User builds the pipeline
- **WHEN** the tutorial builds that resolved YAML
- **THEN** it explains validation, pipeline/layer inspection, available shape/parameter information, and diagram output only if enabled by that scenario

#### Scenario: User trains the model
- **WHEN** the tutorial runs the scenario or resolved configuration with the documented CPU and one-epoch settings
- **THEN** it explains that Lightning performs training and evaluation
- **AND** it shows how to recognize completion and locate actual logs/artifacts without inventing checkpoints or assuming optional logging was enabled

#### Scenario: User exports the model
- **WHEN** the tutorial introduces model export
- **THEN** it explains the portable package and links to the canonical export/reload guide
- **AND** any demonstrated trained-model export uses an actual completed training artifact rather than silently exporting newly initialized weights

### Requirement: Beginner Inconsistencies Removed
Beginner-facing tutorials SHALL NOT introduce one scenario and run commands for a different scenario unless the switch is explicitly explained.

#### Scenario: Getting-started page references scenario name
- **WHEN** a page identifies a first-run recipe and configuration destination
- **THEN** subsequent commands consistently use that recipe and destination
- **AND** any comparison with a different recipe is explicitly labeled

## ADDED Requirements

### Requirement: First-run commands are executable acceptance evidence
The documented first-run sequence SHALL have an automated CPU smoke check in a normal-user package environment. The check SHALL validate completion and stated outputs without adding ML dependencies to the separate static documentation build.

#### Scenario: Clean user environment runs the first example
- **WHEN** CI installs the published-equivalent framework and base-library artifacts and executes the documented sequence without GPU access
- **THEN** discovery, resolution, build, and one-epoch training complete successfully
- **AND** expected configuration/output assertions agree with the guide

#### Scenario: User chooses Studio instead of the CLI
- **WHEN** a user follows the linked Studio first-run procedure
- **THEN** it uses the same synthetic example and explains the same pipeline/training outcome without requiring CLI use first
