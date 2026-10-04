# Studio Workbench UX Delta

## Purpose

Provide an accessible, corporate-design local interface for configuring and observing existing NexuML scenarios, using a node graph without introducing new ML domain concepts.

## ADDED Requirements

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
