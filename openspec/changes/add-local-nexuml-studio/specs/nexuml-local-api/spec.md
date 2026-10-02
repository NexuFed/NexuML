# NexuML Local API

## Purpose

Expose existing NexuML operations to a local user interface through REST and WebSockets without introducing alternative library, scenario, pipeline, or execution semantics.

## ADDED Requirements

### Requirement: Existing runtime identity

The API SHALL run in the user-selected NexuML installation and report its NexuML version, Python executable, working directory, and interface version. It SHALL NOT combine libraries or dependencies from unrelated Python environments.

#### Scenario: API starts from a uv tool installation
- **WHEN** the server is launched through the tool-installed NexuML executable
- **THEN** its runtime identity and discovered installed libraries belong to that executable's isolated environment

#### Scenario: Required server dependencies are absent
- **WHEN** the selected installation lacks the optional API dependencies
- **THEN** startup fails with explicit installation guidance without installing dependencies or changing the selected environment

### Requirement: Authoritative discovery and library management

REST SHALL expose NexuML's existing library sources, registered scenarios/components, component identities and schemas, backend descriptions, and discovery diagnostics. Library add/remove operations SHALL use the existing normalized-root configuration and validation behavior rather than a Studio-specific library catalog.

#### Scenario: Installed or locally configured component is available
- **WHEN** NexuML discovers a component in the selected installation
- **THEN** the API exposes its existing kind, identity, version, configuration schema, and provenance

#### Scenario: Library root changes
- **WHEN** an authenticated user adds or removes a valid local library root
- **THEN** the existing NexuML library configuration changes and subsequent discovery reflects the change

#### Scenario: Library fails to import
- **WHEN** discovery records an import or registration failure
- **THEN** the API exposes that diagnostic without fabricating replacement components or hiding other successfully discovered entries

### Requirement: Configuration fidelity

REST SHALL resolve existing registered scenarios and load, validate, and save NexuML configuration through its existing persistence representation. Component identities/versions, stage and layer order, key mappings, evaluation, execution, and nonvisual configuration fields SHALL retain their meaning. Invalid input SHALL return structured diagnostics and SHALL NOT overwrite an existing file.

#### Scenario: Configuration round-trip
- **WHEN** a valid configuration is loaded and saved without semantic edits
- **THEN** the resulting configuration resolves to the same NexuML definitions and ordered behavior, regardless of formatting differences

#### Scenario: Invalid component field
- **WHEN** a configuration contains a field value rejected by NexuML
- **THEN** validation returns a field-addressable diagnostic and leaves the previously saved configuration unchanged

#### Scenario: Source file changed outside Studio
- **WHEN** a save supplies a base revision that no longer matches the file on disk
- **THEN** the API reports a conflict instead of silently overwriting external edits

### Requirement: Explicit executable build checks

Configuration validation and executable pipeline build checking SHALL be distinct operations. Build checks SHALL require an explicit request, use NexuML's existing compilation behavior, return only observed or declared diagnostics/shapes, and invalidate their result when relevant configuration changes. Catalog discovery and Python scenario resolution SHALL be identified as trusted local code operations, not as sandboxed inspection.

#### Scenario: User checks field constraints
- **WHEN** configuration validation is requested
- **THEN** the API does not run pipeline compilation or dummy forward passes as part of that check

#### Scenario: User requests a build check
- **WHEN** an authenticated user explicitly requests a build check
- **THEN** the API runs existing compilation outside request handling and associates its result with the checked configuration revision

#### Scenario: Build exceeds its configured deadline
- **WHEN** a build check exceeds its deadline
- **THEN** the API reports the timeout and releases the owned check process without publishing a successful result

### Requirement: Existing execution semantics

REST SHALL start training from a frozen, validated NexuML configuration and use the existing execution backend selected by that configuration. The API SHALL NOT implement a second training/evaluation lifecycle, provision infrastructure, silently change the backend, or bypass existing dependency and compatibility guards. At most one resource-consuming build or training operation SHALL be active per local API instance in v1.

#### Scenario: Local training is requested
- **WHEN** a valid local training configuration is submitted
- **THEN** the API executes NexuML's existing training, validation, post-training fit, and test lifecycle as applicable, without blocking discovery/status requests

#### Scenario: Existing Ray configuration is requested
- **WHEN** an existing configured Ray execution is submitted
- **THEN** NexuML's existing Ray entrypoint owns execution and its existing unsupported-scenario errors are preserved

#### Scenario: Draft changes after launch
- **WHEN** the user edits the scenario after starting training
- **THEN** the active invocation continues using its frozen launch configuration

#### Scenario: Another resource-consuming operation is active
- **WHEN** another build or training request is submitted while one is active
- **THEN** the API returns a busy response instead of starting hidden concurrent work or creating a new scheduling queue

### Requirement: Observable operation progress

REST SHALL expose status and terminal results for API-started invocations. WebSockets SHALL expose correlated, sequenced progress, numeric metrics where available, logs, and terminal events. Observability SHALL derive from actual NexuML/backend execution and SHALL NOT fabricate per-node execution, percentages, metrics, or memory estimates.

#### Scenario: Training produces a metric
- **WHEN** the observed backend produces a supported numeric metric
- **THEN** the WebSocket publishes that metric with its available step/epoch context and invocation identity

#### Scenario: Backend does not expose detailed progress
- **WHEN** the backend exposes only driver logs or terminal results
- **THEN** the API reports that limitation while continuing to expose those actual logs/results

### Requirement: Reconnection and interruption are explicit

Browser/WebSocket disconnection SHALL NOT cancel a running invocation. An authenticated reconnect SHALL obtain current status and replay recorded events after a supplied sequence, or receive an explicit replay-gap indication. Terminal results and launch configuration SHALL remain inspectable after API restart; invocations whose execution ownership cannot be recovered SHALL NOT be reported as still running or successfully cancelled.

#### Scenario: Browser reconnects during training
- **WHEN** the browser disconnects and reconnects while the server and training remain active
- **THEN** training continues and observation resumes without launching another invocation

#### Scenario: Server restarts during an invocation
- **WHEN** the API restarts with recorded nonterminal invocation metadata
- **THEN** it identifies the invocation as interrupted or ownership-unknown unless it can verify its live execution ownership

### Requirement: Supported cancellation only

The API SHALL expose cancellation for locally owned training processes, including their owned descendants, and return the resulting state. Cancellation SHALL NOT imply that a checkpoint was created or that remote workers stopped. Unsupported backend cancellation SHALL be reported explicitly rather than simulated by closing a browser or killing only a remote driver.

#### Scenario: User stops local training
- **WHEN** cancellation is requested for an API-owned local training process
- **THEN** the API interrupts its owned process tree, escalates if necessary, and reports completion of cancellation only after confirming local execution has stopped

#### Scenario: Remote cancellation is unsupported
- **WHEN** a cancellation request targets a backend without supported remote cancellation
- **THEN** the API reports that cancellation is unsupported and does not claim remote training has stopped

### Requirement: Local transport and file access protection

The API SHALL bind to loopback, authenticate REST and WebSocket access, restrict allowed browser origins and Host values, and prevent unauthorized cross-origin actions. Transport file reads/writes/downloads SHALL be restricted to the selected working directory and explicitly authorized artifact roots, including protection against path traversal and symlink escape. Credentials SHALL NOT appear in configuration exports, browser URLs, logs, or recorded events.

#### Scenario: Foreign browser origin requests an action
- **WHEN** an unauthorized origin requests a REST action or WebSocket connection
- **THEN** the API rejects it without executing NexuML operations

#### Scenario: File request escapes an authorized root
- **WHEN** an authenticated file request uses traversal or a symlink to escape its permitted roots
- **THEN** the API rejects the request without reading or modifying the target

#### Scenario: Authorized library root is outside the working directory
- **WHEN** the user explicitly adds a local library root through existing NexuML library management
- **THEN** its normal discovery behavior remains available without granting arbitrary HTTP file access to unrelated filesystem locations

### Requirement: Existing results and artifacts

REST SHALL expose the actual frozen configuration, logs, checkpoint references, evaluation results, and existing export outputs associated with observed invocations. Export requests SHALL use existing NexuML export operations and retain their support/dependency constraints; there SHALL be no Studio-only model artifact format.

#### Scenario: Training finishes with evaluation results
- **WHEN** an observed invocation finishes and NexuML returns evaluation results
- **THEN** the API exposes those results and their originating launch configuration

#### Scenario: User requests an existing export format
- **WHEN** a user requests an export supported by the selected NexuML installation and artifact
- **THEN** the API uses the existing export operation and exposes its output without substituting another format
