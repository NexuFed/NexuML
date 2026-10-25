# Spec: documentation-journey

## Purpose

The simplified NexuML reading journey: four primary sections, one canonical procedure per task, clear architecture and feature scope, and a verified CPU-first starting path.

## Requirements

### Requirement: Beginner Journey Navigation

The documentation SHALL expose four primary sections: Start here, Use NexuML, Understand NexuML, and Reference. Tutorials SHALL be reachable from Start here. Changelog, Contributing, and version selection SHALL remain readily accessible without competing with the beginner workflow.

#### Scenario: New reader opens the docs

- **WHEN** a new reader opens the NexuML documentation
- **THEN** the landing page explains what NexuML is and presents a recommended installation and first-run path
- **AND** the reader does not need to choose between overlapping Learn, Concepts, and lifecycle entry paths

#### Scenario: Existing feature docs remain discoverable

- **WHEN** a reader needs tuning, tracking, dataset export, model export/inference, backends, distributed execution, deployment boundaries, autoresearch, or extension guidance
- **THEN** the feature remains discoverable through meaningful task groups, contextual links, and search
- **AND** advanced details are not removed merely to reduce sidebar length

#### Scenario: Reader opens the generated API tree
- **WHEN** a reader opens Reference
- **THEN** generated CLI, configuration/integration details, and both public Python packages are easy to find
- **AND** large generated module trees do not overwhelm the initial navigation view

### Requirement: Diataxis Roles Preserved

The documentation SHALL distinguish first-run tutorials, task guides, conceptual explanations, and exact reference without requiring separate competing navigation sections for every role.

#### Scenario: Reader follows a tutorial

- **WHEN** a reader opens a tutorial page
- **THEN** the page guides the reader through a complete learning experience with a specific outcome and minimal decision-making
- **AND** complete external hands-on tutorials are linked rather than copied into a drifting second tutorial set

#### Scenario: Reader opens a how-to guide

- **WHEN** a reader opens a task guide
- **THEN** it focuses on completing that task and links to relevant concepts/reference instead of mixing exhaustive explanation inline
- **AND** it starts with the recommended working default before introducing optional alternatives

#### Scenario: Reader opens a reference page

- **WHEN** a reader opens a reference page
- **THEN** it provides compact configuration/integration facts or generated API/CLI information without becoming another tutorial

### Requirement: Lightning-Inspired Learning Progression

The documentation SHALL explain NexuML to PyTorch and PyTorch Lightning users within the canonical architecture path rather than requiring a separate Learn NexuML progression before useful work.

#### Scenario: Lightning user evaluates NexuML

- **WHEN** a user familiar with PyTorch Lightning reads the architecture path
- **THEN** it explains how NexuML relates to Lightning and what typed scenarios, reusable components, explicit data flow, persistence, and execution placement add

#### Scenario: Beginner learns the lifecycle

- **WHEN** a beginner reads the architecture explanation
- **THEN** it explains `ScenarioSpec`, resolved config, compiled pipeline, Lightning training, evaluation, and export in a consistent example
- **AND** evaluation is described as part of the implemented lifecycle rather than an invented standalone CLI command

### Requirement: Single authoritative procedure per task

Human-facing documentation SHALL keep each complete procedure in one canonical location. Consolidation SHALL remove duplicated explanations and redirect-only placeholder articles from the reading path while retaining unique feature instructions and safety constraints. Entry pages SHALL contain a short working path rather than advanced inventories or verification histories.

#### Scenario: User starts Studio

- **WHEN** a user opens the Studio README or starting guide
- **THEN** they can find installation, launch, one first-run example, and links to further help without reading hosting, API internals, or release engineering instructions

#### Scenario: User needs advanced details

- **WHEN** a user needs hosting, manual API access, or contributor setup
- **THEN** the entry guide links to the authoritative procedure
- **AND** authentication, trusted-code, filesystem, and process-ownership limits remain available with the relevant procedure

#### Scenario: User consults API documentation

- **WHEN** a reader needs REST request or response details
- **THEN** the guide points to the existing authenticated generated contract
- **AND** the handwritten reference retains necessary WebSocket, conflict, and lifecycle rules without repeating the UI manual

#### Scenario: Contributor consolidates overlapping articles
- **WHEN** scenario, train/run, architecture, or library-registration articles are merged
- **THEN** their unique supported commands, requirements, limitations, and security notes survive in canonical destinations
- **AND** old important URLs map to those destinations without leaving duplicate authoritative copies

### Requirement: Documentation without hard-coded version numbers

Evergreen documentation SHALL omit hard-coded product, runtime, dependency, operating-system, and protocol compatibility numbers from prose, badges, and installation examples. Machine-readable metadata, executable contracts, generated references, version selectors, and dated changelog release facts SHALL remain authoritative exceptions.

#### Scenario: Maintainer reviews documentation

- **WHEN** root/package READMEs, evergreen site pages, and skill bodies/references are reviewed
- **THEN** they contain no hard-coded release numbers, minimum-version ranges, numbered compatibility blocks, pinned image tags, or wheel/tarball release filenames
- **AND** technical metadata in skill frontmatter and example package manifests is not removed

#### Scenario: An example needs an executable contract value

- **WHEN** a working example contains a registered component identity, required configuration version field, or existing endpoint path
- **THEN** its executable meaning is preserved rather than deleting required fields or changing the protocol
- **AND** IP addresses, ports, numerical parameters, release-tag placeholders, and dates are not misclassified as release-number claims

#### Scenario: Reader selects a container image

- **WHEN** the container instructions require an explicit image tag
- **THEN** they show how to select and supply a tag from the registry without hard-coding a release or inventing an unsupported default tag

#### Scenario: Reader opens historical release information
- **WHEN** a dated changelog entry describes a release or a version selector lists published documentation
- **THEN** exact release identifiers and matching tag/release links are retained as historical or generated facts
- **AND** evergreen installation instructions do not duplicate those identifiers as manually maintained compatibility claims

### Requirement: Task guides have observable outcomes
Each task guide SHALL identify its goal and prerequisites, provide copyable commands or code, state expected output and artifact location or discovery method, and offer a clear next action. Optional complexity SHALL follow the recommended working path rather than precede it.

#### Scenario: User follows a task guide
- **WHEN** a user follows a create/run, training, data, tuning, export, or extension guide
- **THEN** they can identify what to run and how to confirm success without reading implementation inventories first
- **AND** complete flag/default tables are linked to their generated reference rather than copied across guides

### Requirement: Vision and feature coverage remain clear
The documentation SHALL explain why NexuML separates reusable implementations from experiment composition, summarize supported capabilities, and distinguish current functionality from future direction. Simplification SHALL reduce duplicate reading, not erase specialized features or imply unsupported capabilities.

#### Scenario: User evaluates NexuML
- **WHEN** a user reads the overview and architecture path
- **THEN** they can understand the reusable-component/scenario model, explicit TensorDict routing, Lightning lifecycle, portable artifacts, and execution boundaries
- **AND** feature summaries lead to the relevant task or reference without requiring marketing pages or an exhaustive API tour
