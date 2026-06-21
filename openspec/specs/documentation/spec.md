# Documentation

## Purpose

TBD — requirements for the NexuModular documentation site, including build tooling, API reference generation, CLI reference, docstring conventions, diagram rendering, versioning, and GitHub Pages deployment.

## Requirements

### Requirement: Documentation site build
The system SHALL provide a Material for MkDocs documentation site that builds with `mkdocs build --strict` using dependencies from the `docs` optional-dependency group.

#### Scenario: Strict build succeeds
- **WHEN** a contributor runs `uv sync --extra docs` and then `mkdocs build --strict`
- **THEN** the site builds with no warnings or errors and produces a `site/` directory

#### Scenario: Local preview
- **WHEN** a contributor runs `mkdocs serve`
- **THEN** the site is served locally with working navigation, search, and theme toggle

### Requirement: Automatic API reference
The system SHALL generate API reference pages from source for both the `nexuml` and `nexuml_library` packages using mkdocstrings with the Griffe backend, without hand-maintained stub pages.

#### Scenario: Both packages documented
- **WHEN** the site is built
- **THEN** the Reference section contains generated API pages for public modules under `src/nexuml` and `library/src/nexuml_library`

#### Scenario: New module appears automatically
- **WHEN** a new public module is added to either package and the site is rebuilt
- **THEN** a corresponding API reference page is generated without manual nav edits

### Requirement: CLI reference
The system SHALL document the `nexuml` Typer command-line interface as a reference page generated from the command definitions.

#### Scenario: CLI commands listed
- **WHEN** the site is built
- **THEN** the CLI reference page lists `nexuml` commands and their options

### Requirement: Google-style docstrings
The system SHALL standardize on Google-style docstrings for public API and enforce the convention with ruff pydocstyle rules.

#### Scenario: Convention enforced
- **WHEN** `ruff check` runs on `src/nexuml` and `library/src/nexuml_library`
- **THEN** pydocstyle (`D`) rules using the Google convention are applied and pass on public API

### Requirement: Mermaid diagram rendering
The system SHALL render Mermaid diagrams embedded in documentation pages so that pipeline diagrams produced by `nexuml build` display as graphics.

#### Scenario: Mermaid fenced block renders
- **WHEN** a docs page contains a mermaid fenced code block and the site is built
- **THEN** the block renders as a diagram rather than literal code

### Requirement: Versioned documentation
The system SHALL publish versioned documentation using mike, maintaining a `latest` alias for the most recent release.

#### Scenario: Release publishes a version
- **WHEN** a `vX.Y.Z` tag is pushed
- **THEN** CI deploys version `X.Y.Z`, updates the `latest` alias, and sets it as the default

#### Scenario: Main publishes dev
- **WHEN** a commit is pushed to `main`
- **THEN** CI deploys or updates the `dev` documentation version

### Requirement: GitHub Pages deployment
The system SHALL deploy the documentation to GitHub Pages from the `gh-pages` branch via a GitHub Actions workflow.

#### Scenario: Automated deploy
- **WHEN** the docs workflow runs on `main` or a release tag
- **THEN** it builds the site and pushes the result to the `gh-pages` branch, updating the live site

### Requirement: Complete core mental model documentation
The documentation site SHALL include a core mental model page that explains how a `ScenarioSpec` compiles into a `PipelineSpec` of `LayerSpec` stages, how each layer uses TensorDict `keys_in` and `keys_out`, how Lightning training executes the compiled pipeline, and how trained pipelines are exported as portable packages.

#### Scenario: User learns the end-to-end architecture
- **WHEN** a user opens the core mental model page
- **THEN** the page explains `ScenarioSpec`, `PipelineSpec`, `LayerSpec`, `keys_in`, `keys_out`, TensorDict data flow, Lightning training, and package export in one coherent flow

#### Scenario: Page is grounded in implementation
- **WHEN** a contributor reviews the page
- **THEN** it maps the explanation to the relevant implementation modules, including `src/nexuml/core/types.py`, compiler/runtime modules, `src/nexuml/training/lightning.py`, and `src/nexuml/core/export.py`

### Requirement: Library element development documentation
The documentation site SHALL include a highest-priority developer guide for creating NexuML library elements and scenarios using the implemented discovery decorators and library layout.

#### Scenario: User adds a custom layer
- **WHEN** a user follows the custom-layer guide
- **THEN** the guide shows how to create a layer under a `layers/<category>/` package, decorate it with `@layer("...")`, expose it through package imports where needed, inspect it with `nexuml registry list layers`, and use it from `LayerSpec(type_key=...)`

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
The documentation site SHALL include a CLI lifecycle guide covering all implemented top-level CLI commands and their role in normal NexuML workflows.

#### Scenario: User follows normal lifecycle
- **WHEN** a user opens the CLI lifecycle page
- **THEN** it explains `resolve` to YAML, `build` compile/validate plus Mermaid diagram export, `train`, `export-dataset`, `export`, `smoke`, and `tune`

#### Scenario: User uses trusted Python scenario files
- **WHEN** a user reads the agent-authored scenario section
- **THEN** it documents `--scenario-file`, `--artifact-dir`, trusted Python execution, provenance snapshots, and optional `HYPOTHESIS`, `PARENT`, `TAGS`, `SEARCH_SPACE`, `TUNING_SPEC`, and `build(**params)` exports

#### Scenario: User manages libraries from CLI
- **WHEN** a user reads the library CLI section
- **THEN** it documents `nexuml library add`, `nexuml library delete`, and `nexuml library list`

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
Every new how-to or tutorial page introduced by this change SHALL contain a copy-pasteable runnable example or explicitly state why the page is reference-only, and the documentation site SHALL continue to pass strict build validation.

#### Scenario: Examples are runnable
- **WHEN** a contributor reviews a new tutorial or how-to page
- **THEN** the page includes setup assumptions, commands or Python code, expected output, and cleanup notes where generated artifacts are created

#### Scenario: Documentation build remains valid
- **WHEN** a contributor runs `uv run mkdocs build --strict`
- **THEN** the documentation builds without warnings or errors

#### Scenario: OpenSpec validates
- **WHEN** a contributor runs `openspec validate expand-nexuml-feature-docs --strict`
- **THEN** the change validates successfully
