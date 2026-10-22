# Spec Delta

## MODIFIED Requirements

### Requirement: Decorator Learning Page
The documentation SHALL explain decorators and discovery within the canonical component/library extension guide, with a brief architectural explanation where useful. Users SHALL not have to read a separate prerequisite learning article before adding a component.

#### Scenario: User asks how NexuML finds components
- **WHEN** a user opens the library or component extension guide
- **THEN** it explains discovery from the base library, installed entry-point libraries, and configured local roots
- **AND** it distinguishes direct Python definitions from registry identities used for discovery and persistence

### Requirement: Registry Verification Commands
The canonical extension guide SHALL show how to verify discovery through implementation-current registry commands and diagnose discovery errors. Consolidation SHALL preserve commands for all documented component roles.

#### Scenario: User verifies registered components
- **WHEN** a user has installed or added a library
- **THEN** the guide shows `nexuml registry list` commands for scenarios, layers, data sources, and evaluation algorithms
- **AND** it identifies the supported verbose diagnostics option

### Requirement: Decorator Docs Link To Reference And How-To Pages
The canonical extension guide SHALL link directly to exact generated decorator/API reference and the relevant component task sections. Its explanation SHALL not repeat complete role-specific implementation procedures.

#### Scenario: User needs exact decorator API
- **WHEN** a user needs exact decorator syntax and signature details
- **THEN** the guide links to the generated public decorator reference
- **AND** the former decorator-reference URL remains available or maps to that destination

#### Scenario: User wants to build an extension
- **WHEN** a user wants to add a layer, data source, evaluation algorithm, scenario, or library
- **THEN** the guide links to the corresponding canonical task section
- **AND** existing typed-definition/runtime, direct PyTorch module, loader configuration, and portability guidance remains covered
