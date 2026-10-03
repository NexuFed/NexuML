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
- **WHEN** a user drags a stage container or repositions a layer within its owning stage
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
- **THEN** graph positions, container sizes and display names are restored while semantic stage membership/order remains in the executable configuration usable by the existing CLI without Studio

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

Studio SHALL provide visible keyboard focus, labeled controls, keyboard and single-pointer alternatives to drag-only node addition/routing/reordering/transfers/movement/resizing, undo/redo for draft and layout edits, and reduced-motion behavior. The editor SHALL support collapsible/resizable side panels at desktop sizes and non-overlapping single-panel navigation at narrow sizes; execution observation SHALL remain usable on mobile.

#### Scenario: User edits without dragging
- **WHEN** a user operates the editor with a keyboard or single-pointer controls
- **THEN** the user can add/select/configure/connect/reorder/transfer components and move/resize stage containers through accessible controls and an ordered Structure outline

#### Scenario: User undoes a configuration edit
- **WHEN** the user undoes or redoes a routing or property change
- **THEN** the semantic draft and its visual projection remain synchronized

#### Scenario: User opens Studio on a narrow screen
- **WHEN** the viewport cannot accommodate the desktop three-pane editor
- **THEN** navigation exposes the canvas, outline, and inspector separately without obstructing controls or causing document-level horizontal overflow

### Requirement: Graph-first authoring and expert disclosure

Studio SHALL group installed components by kind and module provenance, support drag/drop insertion and accessible alternatives, and edit existing routing when a connection is added, reconnected or removed. Visual display names SHALL be layout-only metadata. Known definition-schema routing metadata SHALL expose evaluator parameter/default inputs and declared dataset labels; unknown runtime-only contracts SHALL remain explicitly unresolved.

#### Scenario: User authors on the canvas
- **WHEN** a user drops a component into a stage, connects an initially empty input, renames a node and removes a connection
- **THEN** insertion and routing edit the ordinary ordered configuration, the display name remains layout-only, connection removal is undoable, and node placement never silently reorders execution

#### Scenario: User configures nested typed settings
- **WHEN** a schema describes nested objects, references, unions, nulls, arrays or mappings
- **THEN** Studio exposes corresponding typed controls with installed defaults/options and constraints, preserves unrelated values, and reserves raw JSON/YAML for explicit Expert editing or genuinely untyped values

#### Scenario: User follows a validation problem
- **WHEN** a user activates a field-addressable diagnostic
- **THEN** Studio opens the correct inspector or settings section, reveals the field and focuses it; a runtime error without a field location is not falsely assigned to a draft field

### Requirement: Stage-container authoring

Studio SHALL represent each configured stage as a movable, resizable canvas container whose layers remain visually associated with that stage. Stage headers SHALL identify execution order and layer count; layer cards SHALL identify their stage/layer execution position independently of free canvas placement. The hierarchy SHALL remain stage-to-layers, not unordered groups or nested executable stages. Stage movement, resizing and within-stage layer placement SHALL be layout-only and SHALL NOT change routing, executable configuration, semantic build validity or an active frozen invocation.

#### Scenario: User moves a stage with layers
- **WHEN** a user drags a stage header to another canvas position
- **THEN** the stage and all its layers move together while stage/layer membership, execution order and routing remain unchanged

#### Scenario: User resizes a populated stage
- **WHEN** a user selects a stage and resizes its container
- **THEN** resize controls are visible and the stage retains enough space for its header and children rather than clipping or concealing those layers

#### Scenario: User freely arranges layers within a stage
- **WHEN** a user changes the spatial arrangement of layers inside their owning stage
- **THEN** their execution-position badges and ordered-list positions remain unchanged, making layout distinct from execution order

#### Scenario: User adds a stage on the canvas
- **WHEN** a user drops a Stage item from Structure onto the canvas or uses its non-drag addition control
- **THEN** Studio obtains a unique configuration stage name, states the execution insertion point after the selected stage or last when none is selected, and creates an empty container with Drop a layer here and Add layer affordances without deriving order from its canvas position

#### Scenario: User inserts a new layer
- **WHEN** a user drops an installed layer into a stage or uses Add layer
- **THEN** Studio identifies the destination and insertion slot, adds the ordinary layer configuration there, and requires an explicit destination for a drop outside every stage rather than silently choosing an unrelated stage

### Requirement: Previewed layer transfers

Studio SHALL support moving an existing layer between stages using drag/drop and equivalent non-drag controls. A transfer SHALL preview the destination, execution insertion position and any skipped-stage status before commitment. Successful transfers SHALL preserve the layer's settings, key-routing fields, stable editor identity and selection while updating ordinary stage membership/order and contained placement together. Cancellation and invalid drops SHALL preserve the original draft and SHALL NOT leave floating executable layers.

#### Scenario: User transfers a layer into another stage
- **WHEN** a user drags an existing layer over another stage
- **THEN** Studio highlights a single destination and shows the proposed before/after position or explicit append position; only a valid completed drop commits that membership/order edit

#### Scenario: User drops a layer outside all stages or cancels
- **WHEN** a user releases an existing layer outside every stage or cancels a transfer
- **THEN** Studio restores its original contained placement and semantic membership without creating an undo entry, and explains an invalid drop

#### Scenario: User moves a layer into a skipped stage
- **WHEN** a proposed destination is skipped by the existing NexuML configuration
- **THEN** the transfer preview explicitly warns that the layer will not execute there, and a committed transfer retains the visible skipped-stage state without silently changing that setting

#### Scenario: User undoes a layer transfer
- **WHEN** a user undoes or redoes a completed transfer, including one involving identical layer configurations
- **THEN** both stage lists, execution positions, visual placement, stable editor identity and selection are restored consistently as one edit

### Requirement: Direct ordered-structure editing

Studio SHALL provide a sortable Structure tree for stage/layer ordering and transfers, plus a labeled canvas execution-order strip for stage reordering. Drag operations SHALL expose insertion markers and equivalent keyboard/single-pointer actions. Canvas layout coordinates SHALL NOT determine execution order. Explicit sequence/membership edits SHALL update the ordinary graph projection and diagnostics, preserve key-routing fields and unrendered settings, and mark previous build evidence stale without automatically sorting dependencies, rewriting routes, compiling or launching work.

#### Scenario: User reorders stages from the canvas or Structure tree
- **WHEN** a user moves a stage using an execution-order handle in the canvas strip or Structure tree
- **THEN** the indicated stage sequence and all execution-position badges update in the shared draft without treating that action as a free-layout move or changing within-stage layer order

#### Scenario: User reorders or transfers layers in Structure
- **WHEN** a user drops a layer row at an explicit insertion marker within or between stages
- **THEN** the ordinary layer sequence/membership changes at that slot, matching the canvas hierarchy and preserving the layer's editor identity and settings

#### Scenario: A sequence edit invalidates upstream availability
- **WHEN** a committed sequence/membership edit places a declared input before its producer or changes the applicable preceding writer of a repeated key
- **THEN** Studio updates the projected producer relationships, explains invalid declared dependencies and marks earlier build evidence stale without silently repairing order/routes; runtime-only contracts remain unresolved until explicitly checked

#### Scenario: User completes a layout or order gesture and saves
- **WHEN** a user completes movement, resizing or a sequence edit and saves/reopens the configuration
- **THEN** each completed gesture is one undoable edit, layout metadata remains separate from ordinary ordered YAML, and both canvas and Structure views restore consistently; older matching layout metadata without container sizes uses defaults and stale metadata retains the existing fallback behavior

### Requirement: Native progress and terminal updates

Studio SHALL show actual phase and available epoch/batch counters from native callbacks, use indeterminate activity when totals are unavailable, and render supported terminal overwrite/cursor controls in place without treating terminal text as training telemetry. Raw output SHALL remain available unchanged.

#### Scenario: Preparation emits a terminal download bar
- **WHEN** a download emits carriage-return or supported ANSI progress updates
- **THEN** the displayed terminal updates the existing line instead of accumulating bars, completed messages remain readable and Studio does not invent a download percentage

#### Scenario: Training progresses and observation reconnects
- **WHEN** native callbacks report epoch/batch counters and the browser disconnects then reconnects
- **THEN** Studio displays replayed actual progress without resubmitting training or guessing counters from terminal output
