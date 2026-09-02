## Purpose

Defines the validated and credential-safe process that publishes both NexuML distributions to PyPI from an intentional versioned release.

## ADDED Requirements

### Requirement: Release tags match distribution versions
The release process SHALL accept only semantic release tags whose version matches the `nexuml` and `nexuml-library` distribution versions selected for that release.

#### Scenario: Matching release tag is pushed
- **WHEN** tag `v0.2.0` is pushed while both distributions declare version `0.2.0`
- **THEN** package validation and publication may proceed.

#### Scenario: Release tag and metadata disagree
- **WHEN** a release tag version differs from either distribution version
- **THEN** the release workflow fails before uploading any artifact.

### Requirement: Built artifacts are validated before publication
The release workflow SHALL build wheels and source distributions for both packages, validate their metadata and contents, and install the built artifacts in isolated environments before publication.

#### Scenario: Core-only artifact smoke test succeeds
- **WHEN** the core wheel is installed into an isolated environment without repository source paths or `nexuml-library`
- **THEN** importing `nexuml` succeeds
- **AND** the NexuML CLI help command succeeds
- **AND** no base-library distribution is present.

#### Scenario: Library artifact smoke test succeeds
- **WHEN** the built core and base-library wheels are installed together into an isolated environment
- **THEN** the base library entry point is discoverable
- **AND** representative library components and scenarios can be listed.

#### Scenario: Artifact validation fails
- **WHEN** metadata, contents, source-to-wheel rebuilding, or either isolated installation check fails
- **THEN** no distribution is uploaded
- **AND** no GitHub release is created for the failed build.

### Requirement: Publication uses trusted identity
Production publication SHALL use PyPI Trusted Publishing with short-lived OpenID Connect credentials and a protected release environment rather than a long-lived repository API token.

#### Scenario: Validated production release is published
- **WHEN** all release checks pass for an authorized release tag
- **THEN** the workflow obtains a short-lived trusted-publisher credential
- **AND** uploads the validated artifacts for both distributions to PyPI.

#### Scenario: Untrusted workflow attempts publication
- **WHEN** a workflow run lacks the configured trusted-publisher identity or protected environment approval
- **THEN** PyPI rejects publication.

### Requirement: Test publication is available before production
Maintainers SHALL be able to publish the same validated artifacts to TestPyPI and verify installation without triggering a production release.

#### Scenario: Maintainer runs the TestPyPI path
- **WHEN** an authorized maintainer requests a test publication for versioned candidate artifacts
- **THEN** the artifacts are uploaded to TestPyPI
- **AND** installation is verified using TestPyPI for NexuML packages and the production index for third-party dependencies.

### Requirement: GitHub release follows successful package publication
The release workflow SHALL create the GitHub release only after both PyPI distributions have been uploaded successfully and SHALL attach or link the validated distribution artifacts.

#### Scenario: Both distributions publish successfully
- **WHEN** PyPI accepts all core and base-library artifacts
- **THEN** the corresponding GitHub release is created with generated release notes
- **AND** it identifies the published version and distribution artifacts.

#### Scenario: One distribution upload fails
- **WHEN** PyPI does not accept every distribution artifact in the release
- **THEN** the workflow reports the partial publication clearly
- **AND** it does not create a GitHub release that claims the release completed successfully.
