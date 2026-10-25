## MODIFIED Requirements

### Requirement: Normal User Install First

The documentation SHALL present ordinary released-package installation before contributor setup. Python installation SHALL use PyPI, and Studio installation SHALL use npm, without requiring candidate artifacts or a source checkout.

#### Scenario: User installs NexuML into their own project

- **WHEN** a user follows the primary install page
- **THEN** the docs instruct them to create or activate their own environment and install the framework with `pip install nexuml` or the equivalent `uv pip install nexuml`

#### Scenario: User installs the base library

- **WHEN** a user follows the primary install page and wants reusable components and scenarios
- **THEN** the docs show `pip install "nexuml[library]"` or its uv equivalent
- **AND** explain that the core framework also works without the optional base library
- **AND** keep platform-specific extras and indexes outside the portable first-run path

#### Scenario: User intentionally selects DALI

- **WHEN** a user chooses a scenario or loader that explicitly requires DALI
- **THEN** the docs identify DALI as an optional platform-specific backend
- **AND** provide its separate installation and package-index instructions outside the default first-run path.

#### Scenario: User locates default checkpoints

- **WHEN** the first-run guide trains a scenario with the reusable default callbacks
- **THEN** the docs explain that Lightning places checkpoints under the configured logger directory or trainer root
- **AND** tell the user how to identify the resulting run-specific checkpoint path.

#### Scenario: User needs development setup

- **WHEN** a contributor or framework developer needs an editable source checkout
- **THEN** the docs route them to one contributor setup procedure covering clone, dependency installation, activation, and editable library installation

#### Scenario: User starts Studio

- **WHEN** a user follows Studio setup in their chosen Python environment
- **THEN** the guide shows `pip install "nexuml[api,library]"`, `npm install -g @nexufed/nexuml-studio`, and `nexuml-studio .`
- **AND** briefly explains how to select that environment if its executable is not on PATH
- **AND** does not require a repository clone, frontend build, candidate wheel, release pin, or unpublished-package workaround

#### Scenario: User selects an isolated tool environment

- **WHEN** a user chooses uv-tool installation instead of a project environment
- **THEN** the linked runtime-selection instructions keep API dependencies and component libraries in that selected environment without assuming the shell Python can import them

#### Scenario: Reader opens release engineering documentation

- **WHEN** a maintainer intentionally tests or publishes distributions
- **THEN** the development guide retains applicable TestPyPI and immutable-artifact procedures
- **AND** those procedures are not presented as normal-user prerequisites

### Requirement: Install Examples Use Shell-Safe Quotes

Installation examples SHALL use straight quotes and copy-pasteable shell commands.

#### Scenario: User copies install command

- **WHEN** a user copies the base library install command from the docs
- **THEN** it uses straight ASCII quotes around `nexuml[library]` and does not contain curly quotes
