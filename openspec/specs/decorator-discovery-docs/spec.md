# Spec: decorator-discovery-docs

## Purpose

Documentation covering NexuML decorators and component discovery, providing a learning page that users encounter before extension how-to guides.

## Requirements

### Requirement: Decorator Learning Page

The documentation SHALL include a learning page that explains NexuML decorators and discovery before users reach extension how-to guides.

#### Scenario: User asks how NexuML finds components

- **WHEN** a user opens the decorators and discovery learning page
- **THEN** the page explains that NexuML discovers decorated components from installed packages, entry-point libraries, and local library roots.

### Requirement: Scenario Decorator Explained

The documentation SHALL explain `@scenario`.

#### Scenario: User registers a scenario

- **WHEN** the docs introduce `@scenario("key")`
- **THEN** they explain that the decorated function becomes available by key to CLI commands such as resolve and train.

### Requirement: Layer Decorator Explained

The documentation SHALL explain `@layer`.

#### Scenario: User adds a custom layer

- **WHEN** the docs introduce `@layer("key")`
- **THEN** they explain that the decorated `PipelineLayer` becomes available through `LayerSpec(type_key="key")`.

### Requirement: Data Source Decorator Explained

The documentation SHALL explain `@data_source`.

#### Scenario: User adds a custom dataset source

- **WHEN** the docs introduce `@data_source("key")`
- **THEN** they explain that the decorated source becomes available through `DataSpec(source_type="key")`.

### Requirement: Evaluation Algorithm Decorator Explained

The documentation SHALL explain `@eval_algorithm`.

#### Scenario: User adds evaluation logic

- **WHEN** the docs introduce `@eval_algorithm("key")`
- **THEN** they explain that the decorated evaluation algorithm becomes available through `EvalAlgorithmSpec(type="key")` or the implementation-current evaluation spec field.

### Requirement: Registry Verification Commands

The decorator learning page SHALL show how to verify discovery through registry commands.

#### Scenario: User verifies registered components

- **WHEN** a user has installed or added a library
- **THEN** the docs show verification commands for scenarios, layers, data sources, and evaluation algorithms using `nexuml registry list`.

### Requirement: Decorator Docs Link To Reference And How-To Pages

The decorator learning page SHALL link to detailed reference and task pages.

#### Scenario: User needs exact decorator API

- **WHEN** a user needs exact syntax and compact reference
- **THEN** the learning page links to `docs/reference/decorators.md`.

#### Scenario: User wants to build an extension

- **WHEN** a user wants to add a layer, data source, scenario, or library
- **THEN** the learning page links to the corresponding how-to guide.
