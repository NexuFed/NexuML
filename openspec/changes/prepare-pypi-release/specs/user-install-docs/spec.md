## MODIFIED Requirements

### Requirement: Normal User Install First

The documentation SHALL present normal-user PyPI installation before development installation.

#### Scenario: User installs NexuML into their own project

- **WHEN** a user follows the primary install page
- **THEN** the docs instruct them to create or activate their own environment and install the core framework with `uv pip install nexuml`.

#### Scenario: User installs the base library

- **WHEN** a user follows the primary install page and wants the bundled reusable components and scenarios
- **THEN** the docs instruct them to install the base library with `uv pip install "nexuml[library]"`
- **AND** the docs explain that the core framework works without this optional library.

#### Scenario: User needs development setup

- **WHEN** a contributor or framework developer needs editable source checkout
- **THEN** the docs route them to a separate development installation page with clone, `uv sync --all-extras`, activation, and editable library install instructions.
