# Spec: documentation-journey

## Purpose

The overall documentation navigation and structural design, ensuring a beginner-friendly journey with clear diataxis roles and a progressive learning path.

## Requirements

### Requirement: Beginner Journey Navigation

The documentation SHALL expose a beginner-friendly top-level navigation that guides readers through Home, Start here, Learn NexuML, How-to guides, Concepts, Reference, and Development.

#### Scenario: New reader opens the docs

- **WHEN** a new reader opens the NexuML documentation
- **THEN** the primary navigation presents a clear path from installation and first training through learning concepts, task guides, and reference material.

#### Scenario: Existing feature docs remain discoverable

- **WHEN** a reader needs an advanced feature such as tuning, tracking, dataset export, model export, backends, deployment, or autoresearch
- **THEN** the feature remains discoverable under grouped how-to, concept, or reference sections.

### Requirement: Diataxis Roles Preserved

The documentation SHALL keep page purposes distinct according to tutorial, how-to, explanation, and reference roles.

#### Scenario: Reader follows a tutorial

- **WHEN** a reader opens a tutorial page
- **THEN** the page guides the reader through a complete learning experience with a specific outcome and minimal decision-making.

#### Scenario: Reader opens a how-to guide

- **WHEN** a reader opens a how-to guide
- **THEN** the page focuses on completing a specific task and links to concepts/reference instead of mixing exhaustive explanation inline.

#### Scenario: Reader opens a reference page

- **WHEN** a reader opens a reference page
- **THEN** the page provides compact facts, generated API information, CLI information, registry information, or configuration details without becoming a tutorial.

### Requirement: Lightning-Inspired Learning Progression

The documentation SHALL provide a progressive "Learn NexuML" path for ML researchers familiar with PyTorch or PyTorch Lightning.

#### Scenario: Lightning user evaluates NexuML

- **WHEN** a user familiar with PyTorch Lightning reads the learning path
- **THEN** the docs explain how NexuML relates to Lightning and what NexuML adds on top of Lightning training.

#### Scenario: Beginner learns the lifecycle

- **WHEN** a beginner reads the mental model page
- **THEN** the docs explain the lifecycle from `ScenarioSpec` to resolved config, compiled pipeline, Lightning training, evaluation, and export.
