# Studio Workbench

## Purpose

Provide an accessible, corporate-design local interface for configuring and observing existing NexuML scenarios, using a node graph without introducing new ML domain concepts.

## ADDED Requirements

### Requirement: NexuML-native navigation and concepts

Studio SHALL present existing scenario sources, component/library discovery, pipeline configuration, training, execution results, and artifacts. A selected working directory SHALL be filesystem context, not a new required Studio project model. Studio SHALL NOT require accounts or introduce its own library registry, execution providers, or experiment configuration format.

#### Scenario: User opens an existing working directory
- **WHEN** a user starts Studio against a working directory and NexuML installation
- **THEN** the user can select a discovered scenario or an existing configuration without creating a Studio project or logging in

#### Scenario: User manages a library source
- **WHEN** the user lists, adds, or removes a local library source
- **THEN** Studio displays the results of existing NexuML library operations rather than a separate frontend catalog

### Requirement: Ordered graph projection

Studio SHALL show data sources, pipeline stages/layers, objectives, and evaluation relationships in a node graph that projects the existing NexuML configuration. Connections SHALL represent existing key routing; stage and layer execution order SHALL remain explicit and preserved. Graph grouping SHALL be visual only unless the existing configuration already defines the represented behavior.

#### Scenario: User moves a node
- **WHEN** a user drags a node to another canvas position
- **THEN** its visual position changes without changing stage or layer execution order

#### Scenario: User changes a connection or execution order
- **WHEN** a user changes routing or explicitly reorders a layer
- **THEN** Studio edits the corresponding existing configuration fields, preserves supported routing mappings, and requires valid upstream availability before reporting the configuration as ready

#### Scenario: A tensor key is written more than once
- **WHEN** the existing pipeline overwrites a tensor key in ordered execution
- **THEN** graph dependencies reflect the preceding applicable producer at each consumer rather than connecting every consumer to an arbitrary same-named output

#### Scenario: User views evaluation configuration
- **WHEN** evaluation is displayed on the graph
- **THEN** its settings remain evaluation configuration rather than being converted into invented pipeline layers

### Requirement: Schema-backed configuration inspectors

Studio SHALL configure components from schemas provided by the selected NexuML installation, display required fields before advanced settings, and provide field-addressable validation feedback. Generic factory arguments and complex values without suitable controls SHALL remain editable through an explicit structured-value or YAML fallback; Studio SHALL NOT invent constructor schemas or defaults.

#### Scenario: User selects a typed component
- **WHEN** the user selects a discovered component
- **THEN** the inspector uses its actual schema, defaults, descriptions, and validation constraints

#### Scenario: Component has arbitrary factory arguments
- **WHEN** no typed constructor schema is available for a generic factory
- **THEN** Studio identifies the limitation and provides explicit arguments editing without fabricating dedicated properties

### Requirement: Configuration editing fidelity and recovery

Graph, inspector, and YAML views SHALL describe one semantic draft. Valid unrendered fields SHALL remain preserved. Invalid or unrecognized input SHALL remain available for correction without replacing it with an empty/default graph or overwriting the last valid saved configuration. Editor layout metadata SHALL remain separate from executable configuration.

#### Scenario: User edits a simple property in a complex valid scenario
- **WHEN** a valid scenario contains supported NexuML settings not represented by visual controls
- **THEN** editing another property preserves those unrendered settings on save

#### Scenario: YAML becomes invalid
- **WHEN** the user enters invalid YAML or a definition unknown to the selected installation
- **THEN** Studio retains the input, explains the error, and prevents graph changes from silently replacing it

#### Scenario: External edit conflicts with save
- **WHEN** the API reports that the source file changed externally
- **THEN** Studio offers reload or explicit user-directed conflict resolution and does not silently overwrite either version

#### Scenario: User saves and reopens a configuration
- **WHEN** the configuration and matching layout metadata are saved and reopened
- **THEN** graph positions are restored while the executable configuration remains usable by the existing CLI without Studio

### Requirement: Guided checks and training configuration

Studio SHALL separate pipeline composition, training settings, and execution observation. It SHALL offer discovered scenarios as starting points, expose the existing data/training/execution settings, distinguish field validation from executable build checks, and identify stale or unknown shape information. Training readiness SHALL reflect backend diagnostics, not merely a lack of frontend errors.

#### Scenario: User changes an upstream component
- **WHEN** a configuration change affects previously checked shapes
- **THEN** Studio marks the build result and affected shape information stale until a new explicit build check succeeds

#### Scenario: User selects an unavailable backend
- **WHEN** the selected installation reports missing dependencies or an unsupported scenario/backend combination
- **THEN** Studio shows the specific diagnostic without automatically changing execution settings

### Requirement: Execution and results workflow

Studio SHALL launch an existing NexuML training operation from a reviewed configuration, display its actual progress/logs/results, reconnect without relaunching it, and expose actual checkpoint/export artifacts. Editing the draft SHALL NOT change the active invocation. Unsupported cancellation, checkpoint resume, export, and progress capabilities SHALL be clearly unavailable.

#### Scenario: User launches and edits a scenario
- **WHEN** the user starts training and then edits the draft
- **THEN** the results view remains associated with the frozen launch configuration rather than the edited draft

#### Scenario: Connection is lost
- **WHEN** the execution WebSocket disconnects
- **THEN** Studio shows a connection-stale state and attempts authenticated reconnection without marking training failed or submitting it again

#### Scenario: User inspects completed execution
- **WHEN** an observed invocation completes
- **THEN** Studio shows its actual metrics, evaluation results, available artifacts, and launch configuration

### Requirement: Approved corporate appearance

Studio SHALL use `#101010` as its base background and `#188FD5` with `#000000` text for primary actions. Typography SHALL follow the supplied Montserrat headline and Geist body/label design; spacing, radii, tonal surfaces, and status colors SHALL follow the supplied corporate design. Normal text SHALL meet at least 4.5:1 contrast, and status meaning SHALL NOT rely on color alone.

#### Scenario: Primary action is rendered
- **WHEN** the primary training action is displayed
- **THEN** its default fill is `#188FD5`, its text is black, and its surrounding interface uses the approved dark corporate palette

#### Scenario: Diagnostics change state
- **WHEN** a node or execution changes from valid to warning/error
- **THEN** the state includes a readable label or icon as well as color

### Requirement: Accessible and responsive editing

Studio SHALL provide visible keyboard focus, labeled controls, alternatives to drag-only node addition/routing/reordering, undo/redo for draft edits, and reduced-motion behavior. The editor SHALL support collapsible/resizable side panels at desktop sizes and non-overlapping single-panel navigation at narrow sizes; execution observation SHALL remain usable on mobile.

#### Scenario: User edits without dragging
- **WHEN** a user operates the editor with a keyboard or single-pointer controls
- **THEN** the user can add/select/configure/connect/reorder components through accessible controls and an ordered outline

#### Scenario: User undoes a configuration edit
- **WHEN** the user undoes or redoes a routing or property change
- **THEN** the semantic draft and its visual projection remain synchronized

#### Scenario: User opens Studio on a narrow screen
- **WHEN** the viewport cannot accommodate the desktop three-pane editor
- **THEN** navigation exposes the canvas, outline, and inspector separately without obstructing controls or causing document-level horizontal overflow
