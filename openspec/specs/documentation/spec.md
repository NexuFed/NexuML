# Documentation

## Purpose

Requirements for NexuML's branded Astro Starlight documentation, MDX content, static Python and authentic CLI reference extraction, diagrams, validated editorial updates, immutable version archives, and portable GitHub Pages publication.

## Requirements

### Requirement: Documentation site build
The system SHALL provide a reproducible Astro Starlight documentation project with MDX editorial content, locked JavaScript and Python extraction tooling, documented clean-checkout commands, and validation that fails on invalid content or broken critical links. Documentation builds SHALL NOT require GPU setup or optional ML runtime installations.

#### Scenario: Strict build succeeds
- **WHEN** a contributor follows the frozen dependency setup and runs `npm --prefix docs run check`
- **THEN** content validation, extraction, lint/type checks, compilation, and rendered-link/asset checks succeed and produce previewable static output without installing either public package's ML runtime dependencies

#### Scenario: Local preview
- **WHEN** a contributor runs the documented build and preview commands
- **THEN** navigation, search, theme controls, and assets work at the documented URL
- **AND** the contributor guide distinguishes live authoring from built-preview search

#### Scenario: Invalid documentation is rejected
- **WHEN** MDX syntax/frontmatter or a critical internal link/local asset is invalid
- **THEN** validation fails and identifies the offending source page or rendered route

### Requirement: Automatic API reference
The system SHALL generate navigable public API trees for `nexuml` and `nexuml_library` from source, Google docstrings, signatures, and annotations using the Griffe-to-Starlight integration. New public modules/symbols SHALL appear without handwritten API stubs or navigation edits. Extraction SHALL remain static and independent of heavy runtime imports.

#### Scenario: Both packages documented
- **WHEN** the site is built
- **THEN** Reference contains source-derived public module/class/function/method documentation from only `src/nexuml` and `library/src/nexuml_library`, with annotations and available docstrings

#### Scenario: New module appears automatically
- **WHEN** a new public module is added to either package and the site is rebuilt
- **THEN** its API page/navigation and new public symbols appear without manual stub or navigation edits

#### Scenario: Public exports and private implementations are distinguished
- **WHEN** packages contain explicit exports, public aliases, and private implementations
- **THEN** reference honors public exports and supported re-exports, excludes private implementations, and displays representative typed fields/defaults

#### Scenario: Source links match the documented revision
- **WHEN** a reader follows an API source link
- **THEN** it identifies implementation lines at the built revision, not moving main for a release

#### Scenario: Editorial content embeds an API symbol
- **WHEN** MDX uses a supported Autodoc embed
- **THEN** it renders the source-derived object, anchors, and source links without heavy runtime imports

### Requirement: CLI reference
The system SHALL generate CLI reference from the actual Typer/Click command tree and verify actual root and recursively discovered subcommand help, without duplicate command/option/default definitions or training/optional backend imports during extraction.

#### Scenario: CLI commands listed
- **WHEN** the site is built
- **THEN** public root/nested commands, arguments, options, defaults, and help text reflect the implementation

#### Scenario: CLI implementation changes
- **WHEN** a public command or option changes and the site rebuilds
- **THEN** reference updates without handwritten flags-table changes

#### Scenario: Generated reference matches real help
- **WHEN** lightweight validation invokes root and discovered subcommand help
- **THEN** command/option contracts agree with real help without substitute definitions or mocked ML modules

### Requirement: Google-style docstrings
The system SHALL standardize on Google-style docstrings for public API and enforce the convention with ruff pydocstyle rules.

#### Scenario: Convention enforced
- **WHEN** `ruff check` runs on `src/nexuml` and `library/src/nexuml_library`
- **THEN** pydocstyle (`D`) rules using the Google convention are applied and pass on public API

### Requirement: Mermaid diagram rendering
The system SHALL render embedded Mermaid architecture and generated pipeline diagrams as readable graphics in supported themes and on mobile, preserving the runtime diagram-export contract.

#### Scenario: Mermaid fenced block renders
- **WHEN** a docs page contains a mermaid fenced code block and the site is built
- **THEN** the block renders as a readable diagram rather than only literal code

#### Scenario: Generated pipeline diagram is displayed
- **WHEN** a page includes representative output from the pipeline diagram exporter
- **THEN** its supported Mermaid syntax renders without changing the exporter contract

### Requirement: Versioned documentation
The system SHALL publish separate whole-site `main` and `dev` snapshots from their matching branches, immutable stable-tag snapshots, and `latest` for the newest stable release. Branch snapshots SHALL be labeled unreleased. Existing release archives SHALL remain unchanged.

#### Scenario: Release publishes a version
- **WHEN** a validated stable `vX.Y.Z` tag publishes
- **THEN** its revision creates a new immutable version directory and latest/default select the newest stable release without rolling back for older tags

#### Scenario: Main publishes main
- **WHEN** a commit is pushed to `main`
- **THEN** CI updates main and the global changelog without overwriting dev, releases, or stable latest

#### Scenario: Dev publishes dev
- **WHEN** a validated commit is pushed to `dev`
- **THEN** CI updates only the dev snapshot and shared version metadata
- **AND** main, the global changelog/support, release snapshots, and stable latest remain unchanged
- **AND** stale-build checks compare only that branch's own publication history

#### Scenario: Existing archives survive migration
- **WHEN** Starlight first replaces the publication process
- **THEN** existing release trees remain byte-for-byte unchanged, including rendering technology, and latest retains its existing release until a newer stable tag

#### Scenario: Existing version is published again
- **WHEN** publication encounters an archived version
- **THEN** it confirms a proven idempotent retry or rejects a conflicting revision without overwriting that snapshot

#### Scenario: Reader selects another version
- **WHEN** a reader uses the keyboard-operable selector
- **THEN** it lists main, dev, and all retained releases and selects the corresponding page or clearly offers the selected version's home as fallback

### Requirement: GitHub Pages deployment
The system SHALL keep GitHub Pages at `https://nexufed.github.io/NexuML/` through validated Actions publication. The existing `gh-pages` tree SHALL remain the archive source for complete deployments; directories missing from the current build SHALL NOT be discarded.

#### Scenario: Automated deploy
- **WHEN** main, dev, or stable-tag publication runs
- **THEN** it validates the exact source revision, merges only permitted output into the archive, and deploys the complete static tree without changing URL/provider

#### Scenario: Concurrent publication events occur
- **WHEN** main, dev, and release events overlap
- **THEN** archive updates retain all valid outputs, latest never regresses, and older builds cannot replace newer output of the same branch
- **AND** only main may update the global changelog and its support assets

#### Scenario: Validation or composition fails
- **WHEN** content, preservation, or composition checks fail
- **THEN** the candidate is not deployed and live/historical output remains available

### Requirement: Complete core mental model documentation
The documentation SHALL provide one canonical architecture explanation covering scenario composition, persisted configuration, typed materialization, TensorDict data flow, Lightning execution/evaluation, and portable export. It SHALL explain purpose/scope and distinguish implemented behavior from future direction without overlapping concept detours.

#### Scenario: User learns the end-to-end architecture
- **WHEN** a user opens the architecture explanation
- **THEN** it explains `ScenarioSpec`, `PipelineSpec`, `LayerSpec`, `keys_in`, `keys_out`, TensorDict data flow, Lightning training, and package export in one coherent flow
- **AND** it uses a consistent example and a readable diagram

#### Scenario: Page is grounded in implementation
- **WHEN** a contributor reviews the page
- **THEN** it maps the explanation to the relevant implementation modules, including `src/nexuml/core/types.py`, compiler/runtime modules, `src/nexuml/training/lightning.py`, and `src/nexuml/core/export.py`
- **AND** it distinguishes implemented behavior from future direction rather than presenting planned features as available

### Requirement: Library element development documentation
The documentation site SHALL include a highest-priority developer guide for creating NexuML library elements and scenarios using the implemented discovery decorators and library layout.

#### Scenario: User adds a custom layer
- **WHEN** a user follows the custom-layer guide
- **THEN** the guide shows how to create a typed definition under a `layers/<category>/` package, decorate it with `@layer("...")`, expose it through package imports where needed, inspect it with `nexuml registry list layers`, and use it from `LayerSpec(component=MyLayer(...), ...)`

#### Scenario: User writes a custom composed scenario
- **WHEN** a user follows the composed-scenario tutorial
- **THEN** the page provides a copy-pasteable scenario function decorated with `@scenario("...")` that returns `ScenarioSpec` and can be run end to end with the CLI

#### Scenario: User distributes a library
- **WHEN** a user reads the distribution section
- **THEN** it explains built-in `nexuml_library` discovery, installed libraries via the `nexuml.libraries` entry-point group, user local roots via `nexuml library add`, and that local roots are rescanned every run without a persistent discovery cache

#### Scenario: User inspects registries
- **WHEN** a user reads the registry inspection section
- **THEN** it documents `nexuml registry list layers|data|scenarios|eval` and the `--verbose` / `-v` option for discovery errors exactly as implemented

#### Scenario: User creates other element types
- **WHEN** a user reads the decorator reference
- **THEN** it documents `@data_source`, `@eval_algorithm`, and `@scenario` alongside `@layer`, with runnable or minimal examples for each

### Requirement: Optuna tuning documentation
The documentation site SHALL include a tuning guide that accurately documents `nexuml tune`, `TuningSpec`, default and custom search spaces, pruning, storage, MLflow study runs, dashboard usage, and metric correctness constraints.

#### Scenario: User runs basic tuning
- **WHEN** a user follows the basic tuning example
- **THEN** the page shows a runnable `nexuml tune` command using implemented options `--n-trials`, `--metric`, `--direction`, `--storage`, `--prune/--no-prune`, `--override` / `-O`, `--scenario-file`, and `--artifact-dir`

#### Scenario: User understands defaults and overrides
- **WHEN** a user reads the tuning reference section
- **THEN** it states `TuningSpec` fields `n_trials`, `directions`, `metric_key`, `storage`, and `prune`, the built-in `DEFAULT_SEARCH_SPACE` of only `training.lr` and `training.batch_size`, and that CLI `--metric` and `--direction` override values loaded from `TUNING_SPEC`

#### Scenario: User uses advanced search spaces safely
- **WHEN** a user reads the advanced search-space section
- **THEN** it explains `float`, `int`, and `categorical` entries, conditional `when` branches, `derived` entries, structural/build-factory parameters, and that conditional, derived, and structural search spaces are Python-only and not YAML-exportable

#### Scenario: Missing tuning metric is diagnosed
- **WHEN** tuning optimizes a metric key that is absent from logged metrics and evaluation results
- **THEN** the docs state that tuning raises an error and explain that eval metrics such as `omega` must be surfaced through `evaluation.test_result_metrics` when users need them in test-result metric outputs

### Requirement: Dataset export and exported dataset reuse documentation
The documentation site SHALL include dataset export and reuse pages that document `export-dataset`, on-disk layout, preprocessing boundaries, and the `ExportedDataset` data source.

#### Scenario: User exports a raw dataset view
- **WHEN** a user follows the raw export how-to
- **THEN** it shows a runnable `nexuml export-dataset` command with implemented options `--config` / `-c`, `--output` / `-o`, `--backend`, repeatable `--split`, repeatable `--x-key`, repeatable `--y-key`, `--labels/--no-labels`, and `--dtype`

#### Scenario: User exports a preprocessed view
- **WHEN** a user follows the preprocessing export how-to
- **THEN** it documents `--preprocess/--no-preprocess` and repeatable `--preprocess-until-key`, and states that preprocessing runs the compiled pipeline until the requested TensorDict keys exist

#### Scenario: User understands export layout
- **WHEN** a user reads the dataset export reference
- **THEN** it describes `config.yaml` using `ExportConfig`, `metadata.parquet` with `metadata.csv` fallback, the `data/` directory, `extra.transform_applied`, `x_keys`, `y_keys`, `key_specs`, label prefixing, and split metadata

#### Scenario: User reuses an exported dataset
- **WHEN** a user follows the reuse tutorial
- **THEN** it shows an end-to-end `ExportedDataset` data source (`@data_source("ExportedDataset")`) flow for both raw exports and preprocessed exports

### Requirement: Model export and reload documentation
The documentation site SHALL include a separate model export page that distinguishes pipeline/model export from dataset export.

#### Scenario: User exports and reloads a package
- **WHEN** a user follows the model package example
- **THEN** it shows how to use `export_package`, `load_package` or `load_inference_package`, and `infer` with a TensorDict input

#### Scenario: User exports alternate formats
- **WHEN** a user reads the pipeline export reference
- **THEN** it documents implemented pipeline-export backends `package`, `safetensors`, and `onnx`, including `export_safetensors` and `export_onnx`

### Requirement: Backend registry documentation
The documentation site SHALL explain backends as multiple independent registries and document all categories and names reported by the implemented `nexuml backend list` command.

#### Scenario: User lists backend categories
- **WHEN** a user reads the backend overview
- **THEN** it documents the categories `data-export`, `data-loader`, `training`, `tracking`, `eval-storage`, and `pipeline-export`

#### Scenario: User selects data and export backends
- **WHEN** a user reads backend selection examples
- **THEN** it explains data-export names `numpy`, `numpy_mmap`, `torch`, `tensordict_memmap`, and `webdataset`, data-loader names `torch` and optionally `dali`, and where each is selected in CLI or config

#### Scenario: User understands optional DALI registration
- **WHEN** a user reads the DALI backend note
- **THEN** it states that the DALI data-loader backend is registered only if its optional import succeeds and is otherwise silently unavailable

#### Scenario: User registers a custom backend
- **WHEN** a user reads backend extension docs
- **THEN** it shows how to use `register_export_backend` for data export and `register_loader_backend` for loader backends, and explains that other categories are selected through their documented config/runtime surfaces

### Requirement: Automatic batch-size documentation
The documentation site SHALL include automatic batch-size documentation that describes the `AutoBatchSizeSpec` config, CUDA probe behavior, safety policies, and precedence rules exactly as implemented.

#### Scenario: User configures auto batch size
- **WHEN** a user reads the auto-batch how-to
- **THEN** it documents `mode="auto"`, `min`, `max`, `candidates="power_of_two"`, `safety="largest"|"previous_power_of_two"|"margin"`, and `margin`, with a runnable config example

#### Scenario: User understands CUDA and loader precedence
- **WHEN** a user reads the correctness notes
- **THEN** it states that automatic probing requires CUDA and raises otherwise, and that an explicit `data.loader.batch_size` takes precedence and disables probing

#### Scenario: User distinguishes defaults
- **WHEN** a user reads the defaults section
- **THEN** it states that the spec default safety is `previous_power_of_two` with `min=1`, while library `DEFAULT_AUTO_BATCH_SIZE` uses `min=8`, `safety="margin"`, and `margin=0.8`

### Requirement: CLI lifecycle documentation
The documentation SHALL explain the actual CLI lifecycle in canonical task guides, including supported scenario sources, runnable commands, real outputs, and generated contracts rather than a duplicate learning path.

#### Scenario: User follows normal lifecycle
- **WHEN** a user opens the create/run workflow
- **THEN** it explains resolve, build/validation with optional diagrams, and train, linking canonical export-dataset/export/smoke/tune procedures without duplicating them

#### Scenario: User uses trusted Python scenario files
- **WHEN** a user reads the agent-authored scenario section
- **THEN** it documents `--scenario-file`, `--artifact-dir`, trusted Python execution, provenance snapshots, and optional `HYPOTHESIS`, `PARENT`, `TAGS`, `SEARCH_SPACE`, `TUNING_SPEC`, and `build(**params)` exports

#### Scenario: User manages libraries from CLI
- **WHEN** a user reads the library CLI section
- **THEN** it documents `nexuml library add`, `nexuml library delete`, and `nexuml library list`
- **AND** it distinguishes source-root registration from dependency installation

### Requirement: Diagrams, roots, logging, checkpoints, and evaluation documentation
The documentation site SHALL document remaining implemented features that affect training outputs, reproducibility, and evaluation behavior.

#### Scenario: User exports Mermaid diagrams
- **WHEN** a user reads the diagram docs
- **THEN** it explains `logging.diagram` fields `enabled`, `depth`, `direction`, `show_params`, `show_shapes`, `show_metrics`, and `output_dir`, and links them to build/train diagram export behavior

#### Scenario: User configures roots
- **WHEN** a user reads environment-root docs
- **THEN** it explains `NEXUML_DATA_ROOT` and `NEXUML_LOGS_ROOT` effects on data, logs, MLflow, Optuna storage, preprocessing outputs, and diagram outputs

#### Scenario: User configures evaluation and checkpoints
- **WHEN** a user reads evaluation/checkpoint docs
- **THEN** it documents evaluation algorithms, eval storage names `memory` and `memmap`, `evaluation.test_result_metrics`, Lightning checkpoints, selective checkpoint/package loading, tracking backends `tensorboard`, `dvclive`, and `mlflow`, and related logging fields

### Requirement: Documentation examples and validation
Each task guide/tutorial SHALL provide a copy-pasteable working example or explicitly identify itself as reference-only. Validation SHALL cover content, rendered links/assets, representative interactions, and first-run commands before publication, without inventing checkpoint paths or a standalone evaluation command.

#### Scenario: Examples are runnable
- **WHEN** a contributor reviews a new or consolidated task guide
- **THEN** the page includes setup assumptions, commands or code, expected output, actual artifact locations or how to discover them, and cleanup where relevant
- **AND** it does not invent checkpoint paths or a separate evaluation command

#### Scenario: Documentation build remains valid
- **WHEN** a contributor runs `npm --prefix docs run check`
- **THEN** content, extraction, lint/type, compilation, and critical link/asset checks pass

#### Scenario: OpenSpec validates
- **WHEN** a contributor runs `openspec validate migrate-starlight-docs --strict`
- **THEN** the change validates successfully

#### Scenario: Browser smoke checks run
- **WHEN** built-preview validation runs at the production mount
- **THEN** search, prose/API/CLI, diagrams, tabs, changelog order, version selection, and assets work, with desktop/mobile screenshots and keyboard/accessibility evidence

### Requirement: NexuFed documentation identity
The site SHALL follow the actual current NexuFed identity across prose/API/CLI/changelog pages through restrained typography, logo, surfaces, and accents while preserving responsive navigation, readable code, accessible contrast, visible focus, and coherent dark/light behavior.

#### Scenario: Reviewer compares the site with the corporate source
- **WHEN** representative desktop/mobile pages are reviewed
- **THEN** their visual system matches the corporate source without decorative dashboards, glowing network imagery, or unnecessary promotion

#### Scenario: Reader changes theme or input method
- **WHEN** readers select light mode, use a keyboard, or open narrow viewports
- **THEN** text, code, diagrams, controls, and navigation remain legible and operable without losing features

### Requirement: Prefix-safe routes and hosting portability
Origin and mount configuration SHALL drive links, assets, canonical URLs, redirects, search, and sitemaps. Editorial content SHALL NOT encode the hosting hostname or repository prefix. A future provider/domain change SHALL require configuration/deployment changes rather than editorial rewrites.

#### Scenario: Site builds under the GitHub Pages project prefix
- **WHEN** output is served under `/NexuML/`
- **THEN** versioned docs, latest, global changelog, search, templates, media, and sitemaps resolve under that mount

#### Scenario: Future host uses a root path
- **WHEN** the same sources build with another origin and root mount
- **THEN** new output works without editorial rewrites and migration instructions cover DNS, CNAME/Pages settings, redirects/canonicals, and frozen-archive compatibility paths

### Requirement: Legacy documentation URL continuity
The migration SHALL inventory existing authored/generated API URLs, preserving or explicitly mapping important entry points with usable static compatibility redirects where feasible, without rewriting historical versions.

#### Scenario: Reader follows a previously published link
- **WHEN** an important unversioned/current legacy or consolidated-page URL is opened
- **THEN** it resolves or redirects to its canonical destination with an accessible fallback, retaining mapped sections where feasible

#### Scenario: Maintainer reviews links outside the site
- **WHEN** README, package metadata, tutorials, and release links are reviewed
- **THEN** supported entry links resolve and unavoidable route/fragment changes are recorded in the migration mapping

### Requirement: Lightweight trusted documentation gates
Exact-revision documentation validation SHALL remain independent of GPU/optional ML setup. Untrusted PR checks SHALL use no institute self-hosted runners or publication credentials; credentials SHALL be confined to trusted publication.

#### Scenario: Pull request changes documentation
- **WHEN** a PR is checked
- **THEN** deterministic content, extraction, lint/type, build, and link checks run without optional ML stacks and preserve the runner trust boundary

#### Scenario: Publisher considers a candidate revision
- **WHEN** main or a stable tag is eligible for publication
- **THEN** required documentation checks have passed for that exact revision before live deployment
