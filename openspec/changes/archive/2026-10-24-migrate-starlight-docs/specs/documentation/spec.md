# Spec Delta

## MODIFIED Requirements

### Requirement: Documentation site build
The system SHALL provide a reproducible Astro Starlight documentation project with MDX editorial content, locked JavaScript and Python extraction tooling, documented clean-checkout commands, and validation that fails on invalid content or broken critical links. Documentation builds SHALL NOT require GPU setup or optional ML runtime installations.

#### Scenario: Strict build succeeds
- **WHEN** a contributor follows the documented frozen dependency installation and runs `npm --prefix docs run check`
- **THEN** content validation, extraction, lint/type checks, static compilation, and rendered-link/asset checks succeed and produce previewable static output
- **AND** neither public Python package needs its runtime dependencies installed for documentation extraction

#### Scenario: Local preview
- **WHEN** a contributor runs the documented build and preview commands
- **THEN** navigation, full-text search, theme controls, and assets work at the documented local URL
- **AND** the contributor guide distinguishes live authoring from built-preview search behavior

#### Scenario: Invalid documentation is rejected
- **WHEN** an MDX page has invalid frontmatter or syntax, or a critical internal link or local asset does not resolve
- **THEN** documentation validation exits unsuccessfully and identifies the source page or rendered route

### Requirement: Automatic API reference
The system SHALL generate navigable API reference trees for the public `nexuml` and `nexuml_library` packages from source, Google-style docstrings, signatures, and type annotations using the Griffe-to-Starlight integration. Adding public modules or symbols SHALL NOT require handwritten API stubs or navigation edits.

#### Scenario: Both packages documented
- **WHEN** the site is built
- **THEN** Reference contains generated API pages for public modules under `src/nexuml` and `library/src/nexuml_library`
- **AND** module, class, function, and method documentation exposes signatures, annotations, and available docstrings
- **AND** repository-internal packages outside those two roots are not documented

#### Scenario: New module appears automatically
- **WHEN** a new public module is added to either package and the site is rebuilt
- **THEN** its page and navigation entry appear without a manual stub or navigation edit
- **AND** a new public symbol in an existing module appears after a rebuild

#### Scenario: Public exports and private implementations are distinguished
- **WHEN** a documented package includes explicit exports, public re-exports, underscore-prefixed modules, and private runtime helpers
- **THEN** the published reference honors declared public exports, resolves supported re-exports, and omits private implementation details
- **AND** representative typed configuration fields and their defaults remain visible

#### Scenario: Source links match the documented revision
- **WHEN** a reader follows a source link from an API object
- **THEN** the link identifies its implementation and lines at the revision used to build that documentation
- **AND** release documentation does not link to the moving main implementation

#### Scenario: Editorial content embeds an API symbol
- **WHEN** a contributor uses a supported Autodoc embed in an MDX page
- **THEN** it renders the source-derived API object with working anchors and source links
- **AND** documentation extraction completes without importing heavy or optional runtime stacks

### Requirement: CLI reference
The system SHALL generate the CLI reference from the actual Typer/Click command tree and verify it against root and recursively discovered subcommand help. The reference SHALL remain accurate without maintaining separate lists of commands, options, or defaults and without importing training or optional backend stacks during help extraction.

#### Scenario: CLI commands listed
- **WHEN** the site is built
- **THEN** the CLI reference includes public root and nested commands, arguments, options, defaults, and help text from the implementation

#### Scenario: CLI implementation changes
- **WHEN** a public command or option is added or changed and the documentation is rebuilt
- **THEN** the generated reference reflects the change without editing a handwritten flags table

#### Scenario: Generated reference matches real help
- **WHEN** the lightweight validation environment invokes actual CLI help for the root and discovered subcommands
- **THEN** the documented command and option contracts agree with help output
- **AND** the check uses neither substitute command definitions nor mocked ML modules

### Requirement: Mermaid diagram rendering
The system SHALL render Mermaid diagrams embedded in documentation pages so that architecture diagrams and pipeline diagrams produced by `nexuml build` display as graphics.

#### Scenario: Mermaid fenced block renders
- **WHEN** a docs page contains a Mermaid fenced code block and the site is built
- **THEN** a built-preview browser renders the block as a diagram rather than only literal code
- **AND** the diagram remains readable in supported themes and on mobile

#### Scenario: Generated pipeline diagram is displayed
- **WHEN** documentation includes a representative diagram produced by the existing pipeline diagram exporter
- **THEN** its supported Mermaid syntax renders successfully without changing the runtime diagram-export contract

### Requirement: Versioned documentation
The system SHALL publish separate whole-site `main` and `dev` snapshots from their matching branches, immutable `vX.Y.Z` release snapshots from stable tags, and a `latest` alias for the newest stable release. Branch snapshots SHALL be labeled unreleased. Publication SHALL preserve existing release archives without rebuilding historical releases.

#### Scenario: Release publishes a version
- **WHEN** a validated stable `vX.Y.Z` tag is published
- **THEN** its source revision produces a new immutable version directory
- **AND** `latest` and the default documentation entry point identify the newest stable release without moving backward on an older-tag publication

#### Scenario: Main publishes main
- **WHEN** a validated commit is published from main
- **THEN** the mutable `main` documentation and global changelog are updated
- **AND** dev, release snapshots, and stable latest remain unchanged

#### Scenario: Dev publishes dev
- **WHEN** a validated commit is published from dev
- **THEN** only the dev snapshot and shared version metadata are updated
- **AND** main, the global changelog/support, release snapshots, and stable latest remain unchanged
- **AND** stale-build checks compare only that branch's own publication history

#### Scenario: Existing archives survive migration
- **WHEN** the first Starlight publication replaces the existing publication process
- **THEN** every existing release directory remains byte-for-byte unchanged, including its old rendering technology
- **AND** `latest` continues to identify its existing release until a newer stable release is published

#### Scenario: Existing version is published again
- **WHEN** publication encounters an already archived version
- **THEN** it leaves that directory unchanged
- **AND** it accepts only a proven idempotent retry or rejects a conflicting revision before publication

#### Scenario: Reader selects another version
- **WHEN** a reader uses the keyboard-operable version selector
- **THEN** it lists main, dev, and published releases, including retained historical versions
- **AND** it opens the corresponding page when available or provides a clear fallback to the selected version's home when unavailable

### Requirement: GitHub Pages deployment
The system SHALL continue serving the static site at `https://nexufed.github.io/NexuML/` through validated GitHub Actions publication. The existing `gh-pages` tree SHALL remain the archive source for complete deployments, and publication SHALL not discard directories absent from the current build.

#### Scenario: Automated deploy
- **WHEN** documentation publication runs for main, dev, or a stable release tag
- **THEN** it validates the exact source revision, merges only the permitted output into the existing archive, and publishes the assembled static site to GitHub Pages
- **AND** the project URL and hosting provider remain unchanged

#### Scenario: Concurrent publication events occur
- **WHEN** main, dev, and release publication events overlap
- **THEN** serialized archive updates preserve both valid outputs and the newest stable alias
- **AND** an older build cannot replace newer output from the same branch
- **AND** only main can replace the global changelog and its support assets

#### Scenario: Validation or composition fails
- **WHEN** content checks, archive-preservation checks, or deployment preparation fail
- **THEN** the failing candidate is not deployed
- **AND** the currently live site and historical snapshots remain available

### Requirement: Complete core mental model documentation
The documentation SHALL provide one canonical architecture explanation covering scenario composition, persisted configuration, typed component materialization, TensorDict data flow, Lightning execution, and portable export. It SHALL explain NexuML's purpose and scope without requiring readers to visit multiple overlapping concept pages.

#### Scenario: User learns the end-to-end architecture
- **WHEN** a user opens the architecture explanation
- **THEN** it explains `ScenarioSpec`, `PipelineSpec`, `LayerSpec`, `keys_in`, `keys_out`, TensorDict data flow, Lightning training, and package export in one coherent flow
- **AND** it uses a consistent example and a readable diagram

#### Scenario: Page is grounded in implementation
- **WHEN** a contributor reviews the explanation
- **THEN** it maps to the relevant implementation modules, including `src/nexuml/core/types.py`, compiler/runtime modules, `src/nexuml/training/lightning.py`, and `src/nexuml/core/export.py`
- **AND** it distinguishes implemented behavior from future direction rather than presenting planned features as available

### Requirement: CLI lifecycle documentation
The documentation SHALL explain the implemented CLI lifecycle within canonical task guides rather than a second duplicate learning path. Readers SHALL find which commands to run, the accepted scenario sources, relevant outputs, and links to exact generated CLI contracts.

#### Scenario: User follows normal lifecycle
- **WHEN** a user opens the create/run workflow
- **THEN** it explains `resolve` to YAML, `build` compilation/validation and optional Mermaid export, and `train`
- **AND** it routes users to the canonical `export-dataset`, `export`, `smoke`, and `tune` procedures without duplicating their complete instructions

#### Scenario: User uses trusted Python scenario files
- **WHEN** a user reads the canonical scenario-file procedure
- **THEN** it documents `--scenario-file`, `--artifact-dir`, trusted Python execution, provenance snapshots, and optional `HYPOTHESIS`, `PARENT`, `TAGS`, `SEARCH_SPACE`, `TUNING_SPEC`, and `build(**params)` exports

#### Scenario: User manages libraries from CLI
- **WHEN** a user reads the library guide
- **THEN** it documents `nexuml library add`, `nexuml library delete`, and `nexuml library list`
- **AND** it distinguishes adding a source root from installing dependencies

### Requirement: Documentation examples and validation
Each task guide or tutorial SHALL provide a copy-pasteable working example or explicitly identify itself as reference-only. Validation SHALL cover content, rendered links/assets, representative interactions, and the documented first-run commands before publication.

#### Scenario: Examples are runnable
- **WHEN** a contributor reviews a new or consolidated task guide
- **THEN** the page includes setup assumptions, commands or code, expected output, actual artifact locations or how to discover them, and cleanup where relevant
- **AND** it does not invent checkpoint paths or a separate evaluation command

#### Scenario: Documentation build remains valid
- **WHEN** a contributor runs `npm --prefix docs run check`
- **THEN** the site passes content, extraction, lint/type, compilation, and critical link/asset validation

#### Scenario: OpenSpec validates
- **WHEN** a contributor validates this change with `openspec validate migrate-starlight-docs --strict`
- **THEN** the change validates successfully

#### Scenario: Browser smoke checks run
- **WHEN** validation tests a built preview at the production mount
- **THEN** search, representative prose/API/CLI pages, diagrams, tabbed examples, changelog ordering, version selection, and local asset requests succeed
- **AND** desktop/mobile screenshots and keyboard/accessibility checks are retained as review evidence

## ADDED Requirements

### Requirement: NexuFed documentation identity
The site SHALL follow the current NexuFed corporate visual identity while prioritizing readable developer documentation, accessible contrast, responsive navigation, and clear focus states. Branding SHALL apply consistently to editorial, API, CLI, and changelog pages.

#### Scenario: Reviewer compares the site with the corporate source
- **WHEN** a reviewer checks representative desktop and mobile pages
- **THEN** typography, logo, dark surfaces, restrained accents, spacing, and alignment match the observed corporate design
- **AND** the site avoids decorative dashboards, glowing network imagery, and unnecessary promotional copy

#### Scenario: Reader changes theme or input method
- **WHEN** a reader uses a supported light theme, keyboard navigation, or narrow viewport
- **THEN** text, code, diagrams, controls, and navigation remain legible and operable
- **AND** theme selection does not remove access to any documentation feature

### Requirement: Prefix-safe routes and hosting portability
The site SHALL derive internal links, assets, canonical URLs, redirects, search, and sitemap locations from configurable site origin and mount path. Editorial content SHALL NOT encode the current hosting hostname or repository prefix.

#### Scenario: Site builds under the GitHub Pages project prefix
- **WHEN** the assembled site is served under `/NexuML/`
- **THEN** dev, release, latest, global changelog, search, downloadable templates, media, and sitemap routes resolve within the configured mount

#### Scenario: Future host uses a root path
- **WHEN** validation builds the same sources with a different origin and root mount
- **THEN** new documentation and global changelog output works without editorial rewrites
- **AND** the migration guide explains DNS, GitHub Pages CNAME placement, deployment configuration, canonical/redirect implications, and compatibility handling for frozen archives

### Requirement: Legacy documentation URL continuity
The migration SHALL inventory existing editorial and generated API URLs and preserve or explicitly map important entry points. Changed current routes SHALL have usable static compatibility redirects where feasible, without modifying historical version directories.

#### Scenario: Reader follows a previously published link
- **WHEN** a reader opens an important unversioned or current-version legacy URL, including a consolidated page
- **THEN** the page remains available or directs the reader to its canonical replacement with an accessible fallback link
- **AND** mapped section links retain their intended destination where feasible

#### Scenario: Maintainer reviews links outside the site
- **WHEN** README, Python/Studio package metadata, tutorial entry links, and release documentation are audited
- **THEN** supported documentation entry links resolve and any unavoidable URL or fragment changes are documented in the migration mapping

### Requirement: Lightweight trusted documentation gates
Documentation validation SHALL be independent of GPU/runtime setup and required for the exact published revision. Untrusted pull-request validation SHALL NOT execute on institute self-hosted runners or receive publication credentials.

#### Scenario: Pull request changes documentation
- **WHEN** a pull request is validated
- **THEN** deterministic content, API/CLI extraction, lint/type, build, and critical link checks run without optional ML stacks
- **AND** the existing self-hosted-runner trust boundary is preserved

#### Scenario: Publisher considers a candidate revision
- **WHEN** a main or release candidate is eligible for publication
- **THEN** its required documentation checks have passed for that exact revision before any live deployment
- **AND** publication credentials are confined to the trusted publication stage
