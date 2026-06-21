# Spec: first-run-tutorial

## Purpose

The first-run tutorial experience, ensuring beginners have a consistent, command-by-command walkthrough using the canonical `cifar-resnet` scenario.

## Requirements

### Requirement: CIFAR ResNet First Success

The documentation SHALL make `cifar-resnet` the canonical first successful training example.

#### Scenario: Beginner follows first-run tutorial

- **WHEN** a beginner follows the first-run tutorial
- **THEN** every command uses `cifar-resnet` consistently unless explicitly comparing with another scenario.

#### Scenario: Beginner lists scenarios

- **WHEN** a beginner runs `nexuml registry list scenarios`
- **THEN** the tutorial explains that `cifar-resnet` should appear after the base library is installed.

### Requirement: First-Run Tutorial Explains Each Command

The first-run tutorial SHALL explain each command's purpose, expected artifacts, and introduced concept.

#### Scenario: User resolves the scenario

- **WHEN** the tutorial runs `nexuml resolve cifar-resnet`
- **THEN** it explains that the scenario is compiled to a reproducible config YAML and states the expected config path or tells the implementer to verify the current path.

#### Scenario: User builds the pipeline

- **WHEN** the tutorial runs `nexuml build configs/cifar-resnet.yaml`
- **THEN** it explains validation, layer inspection, tensor shape inspection, parameter counts, and diagram output if configured.

#### Scenario: User trains the model

- **WHEN** the tutorial runs `nexuml train cifar-resnet --max-epochs=2`
- **THEN** it explains that PyTorch Lightning runs the training loop and identifies expected checkpoint/log output locations.

#### Scenario: User exports the model

- **WHEN** the tutorial runs `nexuml export cifar-resnet`
- **THEN** it explains the portable export package and links to the full export/reload guide.

### Requirement: Beginner Inconsistencies Removed

Beginner-facing tutorials SHALL NOT introduce one scenario and run commands for a different scenario.

#### Scenario: Getting-started page references scenario name

- **WHEN** a page tells the user to use `cifar-resnet`
- **THEN** all subsequent commands in that tutorial use `cifar-resnet` unless the page explicitly explains a switch.
