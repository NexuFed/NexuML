# Studio Run Target Selection

## Purpose

Let users choose a compatible execution backend, target and resource request in the existing Run review workflow, using authoritative schemas and capacity snapshots from their selected NexuML installation.

## ADDED Requirements

### Requirement: Runtime-driven backend choices
Studio SHALL populate execution choices from the selected NexuML installation's backend catalog, using reported labels, identities, configuration schemas and capability metadata. Studio SHALL NOT maintain an independent provider list or implement resource discovery. Unsupported dependencies and targets SHALL remain explainable rather than disappear or silently switch to Local.

#### Scenario: Runtime supplies the five initial modes
- **WHEN** NexuML reports Local, RayCluster, RayJob, Kubernetes Job and Kubernetes PyTorchJob definitions
- **THEN** Studio presents those choices with their actual availability and diagnostics

#### Scenario: Installed library adds a backend
- **WHEN** the runtime reports another compatible backend definition
- **THEN** Studio can select it and edit its reported schema without a new frontend provider implementation

#### Scenario: Selected backend is no longer installed
- **WHEN** an open configuration references a definition absent from the selected runtime
- **THEN** Studio retains the selection/configuration for diagnosis and blocks launch instead of replacing it

### Requirement: Selection belongs in the existing Run review
The Run action SHALL open a review dialog containing backend selection, discovered/manual target controls supplied by the backend, applicable namespace/template selection and supported resource settings. Existing execution settings SHALL edit the same ordinary configuration. Detailed template policy and complete YAML SHALL remain progressive Expert disclosure rather than mandatory cluster-administration forms.

#### Scenario: User changes launch settings
- **WHEN** a user selects another backend/target or resource request in the review dialog
- **THEN** the dialog requests authoritative validation/discovery for that selection and shows the resulting launch settings

#### Scenario: User cancels review
- **WHEN** a user dismisses the dialog before confirmation
- **THEN** staged launch edits are discarded, the prior draft remains intact and no job is submitted

### Requirement: Capacity summary reflects the selected request
The dialog SHALL display backend-appropriate node/worker counts, matching capacity, CPU/memory/accelerator resources, relevant quota and observation age where reported. It SHALL distinguish current Ray workers, Kubernetes nodes, intended job replicas and training processes, and show unknown/partial/stale states explicitly. Backend/target/template/request changes SHALL invalidate previous eligibility results, and out-of-order responses SHALL NOT replace a newer selection's results.

#### Scenario: User changes GPU request
- **WHEN** a request changes from one GPU per worker to a larger request
- **THEN** the dialog marks the previous eligibility summary outdated and renders only results associated with the new request

#### Scenario: Discovery returns partial data
- **WHEN** the backend reports allocatable resources but cannot estimate unallocated capacity
- **THEN** the dialog shows the known allocatable values with an unknown-capacity explanation, not zero or a Ready badge

#### Scenario: Slow response belongs to an earlier target
- **WHEN** discovery for the previous target completes after a newer selection's request
- **THEN** Studio does not use that response as the current target's capacity or readiness

### Requirement: Readiness and diagnostics are actionable
The dialog SHALL distinguish loading, missing setup, incompatible configuration, restricted visibility, unavailable target, potentially queued execution and launchable selection. Known blocking validation/access errors SHALL prevent launch; incomplete capacity alone SHALL NOT block a submission that the backend permits. Errors SHALL appear beside affected controls and in a focusable summary linking to those controls. Retry/refresh SHALL repeat discovery without launching work.

#### Scenario: PyTorchJob API is absent
- **WHEN** the backend reports that the selected cluster does not serve PyTorchJob
- **THEN** Studio explains the missing support and disables launch for that selection without changing its mode

#### Scenario: Capacity is unknown but submission is permitted
- **WHEN** backend preflight permits submission despite restricted resource visibility
- **THEN** Studio presents the limitation prominently and permits explicit confirmation without promising immediate scheduling

### Requirement: Launch uses the reviewed frozen configuration
Confirmation SHALL validate and freeze the exact backend, target, template content and supported resource overrides with the existing scenario/training revision. Changed draft or template content SHALL require renewed review. The confirmation action SHALL identify the selected mode and target, prevent duplicate submissions and commit reviewed execution edits as one ordinary undoable configuration change. Editing later drafts SHALL NOT alter the active invocation.

#### Scenario: Review changes after preflight
- **WHEN** the source draft or referenced template changes after a successful review
- **THEN** confirmation is rejected or requires renewed review rather than submitting unreviewed content

#### Scenario: User confirms a remote job
- **WHEN** a validated review is confirmed once
- **THEN** Studio submits that frozen selection once, opens the existing Execution view and shows the native submitted/pending/running state as reported

#### Scenario: User edits during execution
- **WHEN** a draft changes after a run starts
- **THEN** execution details remain associated with the frozen launch configuration and native reference

### Requirement: Execution controls reflect backend capabilities
The existing Execution view SHALL show native job references and only supported status/log/metric/artifact/cancellation/resume controls. Missing telemetry SHALL remain explicitly unavailable; local driver exit SHALL NOT be represented as remote cancellation. External artifact references SHALL NOT be falsely rendered as authorized local downloads.

#### Scenario: Backend supports logs but not scalar metrics
- **WHEN** a job reports native status and logs without scalar telemetry
- **THEN** Studio displays those observations without fabricated charts or training percentages

#### Scenario: Stop is unsupported
- **WHEN** the backend cannot confirm remote cancellation
- **THEN** Studio disables Stop with the reported reason rather than offering a local-driver termination as worker cancellation

### Requirement: Accessible corporate Run dialog
The dialog SHALL retain the existing approved corporate tokens, typography and components, with labeled keyboard-operable controls, visible focus, managed dialog focus and focus restoration to the invoking Run control. Status meaning SHALL use readable text rather than color alone. The dialog SHALL fit narrow viewports with internal scrolling and persistent reachable actions, announce important discovery/errors without announcing every numeric refresh, and preserve reduced-motion behavior.

#### Scenario: User configures execution by keyboard
- **WHEN** a keyboard user opens Run, selects a backend and target, follows a validation error and cancels
- **THEN** every control and diagnostic is reachable, focus stays in the open dialog and returns to Run on dismissal

#### Scenario: Dialog is used on a narrow screen
- **WHEN** the viewport cannot accommodate desktop capacity/details columns
- **THEN** content becomes a single-column internally scrollable dialog with reachable actions and no document-level horizontal overflow
