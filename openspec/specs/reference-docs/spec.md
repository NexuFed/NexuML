# Spec: reference-docs

## Purpose

The reference documentation section, including a landing page and preservation of generated CLI and API reference pages.

## Requirements

### Requirement: Reference Landing Page

The documentation SHALL include a reference landing page that routes users to CLI, API, decorators, registry, backends, and environment reference pages.

#### Scenario: User needs command details

- **WHEN** a user opens the reference landing page looking for command syntax
- **THEN** the page routes them to the generated CLI reference.

#### Scenario: User needs Python API details

- **WHEN** a user opens the reference landing page looking for classes or functions
- **THEN** the page routes them to the generated API reference.

#### Scenario: User needs registry details

- **WHEN** a user opens the reference landing page looking for discovered components
- **THEN** the page routes them to registry and decorator reference pages.

### Requirement: Generated API Reference Preserved

The documentation SHALL preserve the generated API reference for `nexuml` and `nexuml_library`.

#### Scenario: Docs build runs

- **WHEN** the docs site is built
- **THEN** generated API pages for `src/nexuml` and `library/src/nexuml_library` are still generated and linked under Reference.

### Requirement: Generated CLI Reference Preserved

The documentation SHALL preserve the generated CLI reference.

#### Scenario: User opens CLI reference

- **WHEN** a user opens the CLI reference page
- **THEN** the page is still generated from the current CLI implementation using the existing docs tooling.
