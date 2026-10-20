# Spec: reference-docs

## Purpose

Compact reference entry points for configuration/integrations, authentic generated CLI contracts, and both static public Python API trees with searchable symbols and revision-correct source links.

## Requirements

### Requirement: Reference Landing Page

Reference SHALL provide compact entry points for generated CLI commands, configuration/integrations, and both public Python API trees. Decorator, registry, backend, environment, and tuning contracts SHALL remain reachable through those entry points and contextual task links without becoming competing beginner paths.

#### Scenario: User needs command details

- **WHEN** a user opens Reference looking for command syntax
- **THEN** it routes them directly to the generated CLI reference, including nested command groups

#### Scenario: User needs Python API details

- **WHEN** a user opens Reference looking for classes or functions
- **THEN** it exposes both generated package trees and their symbol/source links

#### Scenario: User needs registry details

- **WHEN** a user opens Reference looking for discovered components
- **THEN** it routes them to the current registry contract and generated decorator signatures
- **AND** task procedures remain in the canonical library/extension guide

#### Scenario: User needs specialized configuration details
- **WHEN** a user needs scenario fields, backend selection, root resolution, or a tuning-file contract
- **THEN** the relevant exact reference remains reachable and searchable without duplicating it in every task guide

### Requirement: Generated API Reference Preserved

The documentation SHALL preserve generated source-derived API coverage for `nexuml` and `nexuml_library` within the Starlight site, including public module navigation, symbols, and revision-correct implementation links.

#### Scenario: Docs build runs

- **WHEN** the site is built
- **THEN** public API pages for `src/nexuml` and `library/src/nexuml_library` are generated and linked under Reference
- **AND** they are searchable alongside authored documentation

### Requirement: Generated CLI Reference Preserved

The documentation SHALL preserve a CLI reference generated from the actual implementation, replacing the old generator without maintaining a second handwritten command/option definition.

#### Scenario: User opens CLI reference

- **WHEN** a user opens the CLI reference page
- **THEN** its command groups, arguments, options, defaults, and help text reflect the current Typer/Click app
- **AND** the generated result is checked against actual root and subcommand help
