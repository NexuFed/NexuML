## MODIFIED Requirements

### Requirement: Normal User Install First

The documentation SHALL present normal-user PyPI installation before development installation.

#### Scenario: User installs NexuML into their own project

- **WHEN** a user follows the primary install page
- **THEN** the docs instruct them to create or activate their own environment and install the core framework with `uv pip install nexuml`.

#### Scenario: User installs the base library

- **WHEN** a user follows the primary install page and wants the bundled reusable components and scenarios
- **THEN** the docs instruct them to install the base library with `uv pip install "nexuml[library]"`
- **AND** the docs explain that the core framework works without this optional library
- **AND** the first portable built-in training path does not require the separate DALI extra or NVIDIA package index.

#### Scenario: User intentionally selects DALI

- **WHEN** a user chooses a scenario or loader that explicitly requires DALI
- **THEN** the docs identify DALI as an optional platform-specific backend
- **AND** provide its separate installation and package-index instructions outside the default first-run path.

#### Scenario: User locates default checkpoints

- **WHEN** the first-run guide trains a scenario with the reusable default callbacks
- **THEN** the docs explain that Lightning places checkpoints under the configured logger directory or trainer root
- **AND** tell the user how to identify the resulting run-specific checkpoint path.

#### Scenario: User needs development setup

- **WHEN** a contributor or framework developer needs editable source checkout
- **THEN** the docs route them to a separate development installation page with clone, `uv sync --all-extras`, activation, and editable library install instructions.
