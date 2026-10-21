# studio-workbench Specification

## Purpose

Provide an accessible, corporate-design local interface for configuring and observing existing NexuML scenarios, using a node graph without introducing new ML domain concepts.

## Requirements

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

### Requirement: Task-focused workbench navigation

Studio SHALL present Pipeline, Training and Runs as primary destinations. Metrics, Logs, Artifacts and Configuration SHALL belong to the selected observed operation within Runs. Navigating between these views SHALL retain the current draft, selection and selected operation without saving, launching or replacing configuration.

#### Scenario: User moves between editing and observation
- **WHEN** a user with an unsaved draft and a selected run switches through Pipeline, Training and Runs
- **THEN** draft edits and selections remain intact, results retain their operation identity, and no operation is submitted by navigation

#### Scenario: User opens Runs without a draft
- **WHEN** Studio has no loaded scenario and the user opens Runs
- **THEN** retained training, build and export observations remain selectable with their kind and status, or an actionable empty state appears

### Requirement: Contextual source and file actions

Studio SHALL offer Choose a recipe and Open YAML at entry. After loading a scenario, source switching and file actions SHALL be available through the scenario menu rather than a permanent setup row. Save state SHALL distinguish an unsaved recipe/draft from a saved file, and Save as SHALL allow an explicit destination while preserving existing conflict and sidecar behavior.

#### Scenario: User opens and saves a recipe
- **WHEN** a user explicitly opens an installed recipe and invokes Save before it has a file destination
- **THEN** Studio asks for a destination, writes the ordinary configuration and separate layout through existing save behavior, and reports Saved only after successful persistence

#### Scenario: User switches sources with unsaved changes
- **WHEN** a user opens another recipe or YAML file while the current draft or YAML buffer is unsaved
- **THEN** Studio obtains confirmation before replacement and cancellation retains the current document and buffers

### Requirement: Alternate draft YAML editing

Studio SHALL present YAML as an alternate editing view of the same draft, with explicit Apply and Discard actions. Switching views SHALL preserve unapplied or invalid buffers. While YAML is unapplied, visual semantic edits, saving and launch SHALL remain blocked with a visible path back to the buffer; observation of existing runs SHALL remain available.

#### Scenario: User leaves an invalid YAML buffer
- **WHEN** a user enters invalid YAML, changes view and returns to YAML
- **THEN** the exact buffer remains available, the last valid draft is retained, and applying unrelated visual changes cannot overwrite the buffer

#### Scenario: User applies corrected YAML
- **WHEN** a user successfully validates and applies corrected YAML
- **THEN** visual editing resumes against that same semantic draft with undo support and preserved valid unrendered fields

### Requirement: Contextual component discovery and insertion

Studio SHALL separate Structure and Components into tabs of the left panel, with Structure initially shown for a loaded scenario. Components SHALL provide installed-catalog search and categories. Add layer SHALL open Components with its destination and insertion slot visible. An unspecified destination SHALL be resolved from explicit selection or requested, never silently assigned to the first stage.

#### Scenario: User adds from a selected stage
- **WHEN** a user activates Add layer for a stage
- **THEN** Components opens with search focused, that stage and the insertion slot are identified, and insertion preserves existing before/after, skipped-stage and undo behavior

#### Scenario: User browses without a stage selection
- **WHEN** a user selects a layer component without a chosen destination or selected stage/layer
- **THEN** Studio requests a destination before inserting, and changing panel tabs retains the component search and current draft

### Requirement: Readable ordered canvas workspace

Studio SHALL allocate the remaining desktop workspace to the canvas and side panels, with compact execution-order controls and collapsible diagnostics. Initial editing SHALL use a readable node scale; Fit view SHALL remain an explicit overview action. Compact cards SHALL retain order, connected ports, missing required inputs and skipped state while optional routing controls remain discoverable by selection or focus.

#### Scenario: User opens and navigates a pipeline
- **WHEN** a user opens a pipeline wider than the available canvas and selects a layer from Structure
- **THEN** Studio provides a readable focused view without forcing every node into view, changing positions or changing execution order, and Fit view remains available

#### Scenario: User connects or inspects a compact node
- **WHEN** optional unconnected ports are collapsed
- **THEN** connected edges and required missing inputs remain visible, selection or keyboard-accessible routing controls reveal addition options, and changing disclosure does not change routes

#### Scenario: User reopens an existing layout
- **WHEN** a configuration has a matching saved layout sidecar
- **THEN** existing positions and container sizes are preserved, and compact defaults or arrangement occur only for new layouts or an explicit Arrange action

### Requirement: Focused component inspection

Studio SHALL lead the inspector with the selected component and Settings and Routing tabs. Required/common settings SHALL precede Advanced controls. Routing SHALL retain aliases, labels, metadata and non-drag editing. Raw editing SHALL be available at the component or settings-section boundary without repeated raw controls for every collection. Invalid field buffers SHALL survive navigation between tabs and sections.

#### Scenario: User inspects settings and routing
- **WHEN** a user switches between a layer's Settings and Routing tabs
- **THEN** both edit the same component, all supported routes remain accessible, and unapplied raw input remains available without being silently applied or discarded

#### Scenario: User follows an issue in a hidden inspector section
- **WHEN** a user activates a field-addressable issue for a hidden routing or advanced property
- **THEN** Studio selects the object, reveals its tab/section and focuses the field with the corresponding error text

### Requirement: Task-oriented training forms

Studio SHALL organize Training into Basics, Optimizer and schedule, Data loading, Execution, and Checkpoints and logging. Common settings SHALL use direct controls and readable labels, preserving supported unions, explicit values, inheritance and custom factories. Each setting SHALL remain bound to its original configuration field, with remaining valid settings reachable through relevant Advanced controls.

#### Scenario: User configures fixed or automatic batch size
- **WHEN** the installed schema supports fixed and automatic training batch size
- **THEN** Studio presents those meaningful modes, accepts a precise fixed number, exposes the actual automatic-mode fields and preserves their types rather than asking for an Integer or Object mode

#### Scenario: Loader configuration overrides training batch size
- **WHEN** an explicit loader batch-size override exists
- **THEN** Basics identifies that override and provides a path to edit it; choosing Use training batch size in Data loading restores inheritance rather than copying the current training value

#### Scenario: User edits a custom or alternative value
- **WHEN** a configuration contains a valid string/integer alternative, automatic setting, custom factory or untyped argument
- **THEN** Studio retains and exposes that value, preserves its serialized type and unrelated fields, and does not invent constructor options or defaults

#### Scenario: User navigates training sections
- **WHEN** a user changes sections with edits or retained invalid input
- **THEN** that state remains available, and selecting a validation issue opens and focuses the appropriate section

### Requirement: Compact actionable check status

Studio SHALL expose Check fields and Build model together with their different execution effects stated. A compact Problems surface SHALL show issue count and current/stale build state, expanding on request or a failed explicit check/build/apply. Issues SHALL retain their source, field navigation and accessible announcement. A successful field check SHALL NOT imply a successful build.

#### Scenario: User checks fields then changes the model
- **WHEN** a user performs field validation, explicitly builds the model, then makes a semantic edit
- **THEN** Studio distinguishes the field and build results, marks previous build evidence as needing rechecking, and does not compile or launch automatically

#### Scenario: A failure occurs with Problems collapsed
- **WHEN** an explicit check, build or apply fails while diagnostic details are collapsed
- **THEN** the failure is announced and its details become visible, with field-addressable issues focusable and unaddressed failures linked to their actual source

### Requirement: Decision-focused run review

Studio SHALL group Run review around destination, supported resources, observed capacity and the launch summary. Backend-provided choices and generic contributed-backend fields SHALL remain usable. Review and confirmation SHALL stay distinct, with the next action visually emphasized. Cancel SHALL discard staged edits and changed selections SHALL invalidate earlier review.

#### Scenario: User revises a reviewed destination
- **WHEN** a user changes backend, target, template or resource settings after review
- **THEN** prior review becomes invalid, confirmation requires renewed review, and cancel leaves the original draft intact

#### Scenario: Capacity is incomplete
- **WHEN** the selected backend reports unknown or partial capacity
- **THEN** Studio shows that limitation with source and age, keeps known blockers prominent, and follows backend preflight rather than claiming readiness or treating unknown as zero

### Requirement: Run-centered observation and outputs

Studio SHALL keep status, progress, errors and operation identity visible across the selected run's tabs. Configuration SHALL show the frozen launch settings, distinct from draft YAML, with the existing explicit Open as draft action. Tab changes SHALL preserve observation without relaunch. Stop, resume, inspection, artifact and export controls SHALL reflect actual backend capabilities.

#### Scenario: User edits the draft during a run
- **WHEN** a user changes the draft and returns to the active operation
- **THEN** charts, logs, artifacts and Configuration still describe its frozen source, and Studio identifies a draft difference only when it can compare the corresponding configurations

#### Scenario: User navigates or reconnects to observation
- **WHEN** a user switches run tabs or reloads and reselects an existing operation
- **THEN** observation remains available or is replayed for that identity, stale/gap states remain visible, and no new workload is submitted

#### Scenario: Backend supports only native status and logs
- **WHEN** a selected operation lacks scalar telemetry or supported remote cancellation
- **THEN** Studio prioritizes available status/logs, explains unavailable capabilities at their controls, and does not fabricate charts, training percentages or cancellation success

#### Scenario: User exports an available trained checkpoint
- **WHEN** a user opens Artifacts for a completed eligible training operation and explicitly exports its checkpoint
- **THEN** Studio starts the existing export operation, selects its identity and outputs, and preserves access to the source training record

### Requirement: Comparable observed metric presentation

Studio SHALL prioritize the reported total training/validation loss pair and plot that pair on a shared optimizer-step axis and shared value scale when available. Each series SHALL retain actual observations and clear labels/styles. Other metrics SHALL remain separately discoverable unless their compatibility is known. Missing samples SHALL NOT become zeroes, smoothed values or invented phases.

#### Scenario: Training and validation have different observation counts
- **WHEN** total-loss telemetry includes multiple training points and one validation point at a later step
- **THEN** the chart displays each at its recorded optimizer step, shows the single validation observation visibly, and does not align it by sample index or relabel the axis as epochs

#### Scenario: Metrics have unrelated meanings
- **WHEN** an operation reports total loss, accuracy and a custom scalar
- **THEN** accuracy and the custom scalar remain accessible with their own identity and scale rather than being normalized into the loss chart

### Requirement: Restrained responsive workbench hierarchy

Studio SHALL retain the approved corporate palette and bundled typography while making common controls and node labels readable. Desktop side panels SHALL have visible collapse and resize controls with keyboard and single-pointer alternatives. Narrow layouts SHALL expose non-overlapping panel/section views with reachable actions, visible focus, reduced motion and no document-level horizontal overflow.

#### Scenario: User resizes or collapses a desktop panel
- **WHEN** a user operates a panel separator or its non-drag width/collapse controls
- **THEN** the canvas responds within usable bounds, controls remain reachable, and focus is not trapped in hidden content

#### Scenario: User works on tablet or mobile
- **WHEN** Studio is used at 1024px or 390px viewport width
- **THEN** Pipeline panels and Training sections can be selected separately, Runs and Run review remain usable, touch controls have at least 44px targets, and the document has no horizontal overflow

#### Scenario: User relies on visual or keyboard accessibility
- **WHEN** common controls, status changes and selected nodes are displayed
- **THEN** normal text meets 4.5:1 contrast, primary actions retain black text on the approved blue, status uses text/icons as well as color, keyboard focus is visible and disclosure is not hover-only
