## MODIFIED Requirements

### Requirement: Graph-first authoring and expert disclosure

Studio SHALL group installed components by kind and module provenance, support drag/drop insertion and accessible alternatives, and edit existing routing when a connection is added, reconnected or removed. Visual display names SHALL be layout-only metadata. Native component I/O descriptions SHALL expose evaluator parameter/default inputs and declared dataset labels separately from parameter schemas; unknown runtime-only contracts SHALL remain explicitly unresolved.

#### Scenario: User authors on the canvas
- **WHEN** a user drops a component into a stage, connects an initially empty input, renames a node and removes a connection
- **THEN** insertion and routing edit the ordinary ordered configuration, the display name remains layout-only, connection removal is undoable, and node placement never silently reorders execution

#### Scenario: User configures nested typed settings
- **WHEN** a schema describes nested objects, references, unions, nulls, arrays or mappings
- **THEN** Studio exposes corresponding typed controls with installed defaults/options and constraints, preserves unrelated values, and reserves raw JSON/YAML for explicit Expert editing or genuinely untyped values

#### Scenario: User follows a validation problem
- **WHEN** a user activates a field-addressable diagnostic
- **THEN** Studio opens the correct inspector or settings section, reveals the field and focuses it; a runtime error without a field location is not falsely assigned to a draft field

#### Scenario: User connects a dataset label
- **WHEN** the selected source declares fixed label outputs through its native contract
- **THEN** the Data node exposes the same label ports without reading custom JSON Schema keywords or constructing the dataset
- **AND** labels already declared by the scenario remain represented

#### Scenario: User edits parameter-backed evaluator inputs
- **WHEN** a native contract describes scalar, optional or collection key parameters
- **THEN** ports and accessible connection selectors edit those existing parameters with the correct domain and value shape
- **AND** no component-specific frontend mapping or inferred `_key` naming convention is required

#### Scenario: User inspects an inherited evaluator default
- **WHEN** an omitted or cleared context route resolves to an evaluator fallback
- **THEN** Studio displays the same effective key used by runtime construction
- **AND** it does not hide an inherited connection when clearing it would restore that fallback
- **AND** explicit overrides remain editable and component-parameter defaults are not confused with inherited context defaults

#### Scenario: Connection candidates have different domains or producers
- **WHEN** a user adds, reconnects or selects a route after a key overwrite or order edit
- **THEN** candidates retain matching-domain and applicable-preceding-producer rules
- **AND** alias, label, metadata and evaluator routing continue to edit their existing configuration locations

#### Scenario: Runtime-only I/O is not declared
- **WHEN** an otherwise usable component lacks a static contract or has dynamic outputs
- **THEN** Studio retains schema-backed and raw configuration editing while reporting unresolved I/O
- **AND** it does not invent edges, treat unknown outputs as known-empty, automatically build the component or claim readiness from missing declarations

#### Scenario: User browses component groups
- **WHEN** installed built-in or contributed definitions are listed
- **THEN** grouping follows their kind and module provenance without schema-specific category overrides
- **AND** component search, insertion and accessible navigation remain available even when a former custom subgroup disappears

#### Scenario: User edits a native fixed-choice field
- **WHEN** a definition schema exposes its native enumeration
- **THEN** the existing generic dropdown offers its supported values without custom schema extras
- **AND** an invalid value entered through raw configuration produces a field-addressable validation error without discarding the user's buffer
