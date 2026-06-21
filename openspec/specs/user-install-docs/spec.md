# Spec: user-install-docs

## Purpose

Installation documentation that prioritises the normal-user install path and keeps development setup separate from the beginner tutorial flow.

## Requirements

### Requirement: Normal User Install First

The documentation SHALL present normal-user installation before development installation.

#### Scenario: User installs NexuML into their own project

- **WHEN** a user follows the primary install page
- **THEN** the docs instruct them to create or activate their own environment and install NexuML with `uv pip install "git+https://github.com/NexuFed/NexuML.git"`.

#### Scenario: User installs the base library

- **WHEN** a user follows the primary install page
- **THEN** the docs instruct them to install the base library with `uv pip install "git+https://github.com/NexuFed/NexuML.git#subdirectory=library"`.

#### Scenario: User needs development setup

- **WHEN** a contributor or framework developer needs editable source checkout
- **THEN** the docs route them to a separate development installation page with clone, `uv sync --all-extras`, activation, and editable library install instructions.

### Requirement: No Clone Requirement For Beginner Tutorial

The beginner tutorial SHALL NOT require users to clone the NexuML repository unless they are explicitly following development documentation.

#### Scenario: Beginner starts first tutorial

- **WHEN** a beginner opens the first training tutorial
- **THEN** the setup references normal-user install and does not require `git clone`, `cd NexuML`, or `uv sync --all-extras`.

### Requirement: Install Examples Use Shell-Safe Quotes

Installation examples SHALL use straight quotes and copy-pasteable shell commands.

#### Scenario: User copies install command

- **WHEN** a user copies the base library install command from the docs
- **THEN** the command uses straight ASCII quotes around the Git URL and does not contain curly quotes.
