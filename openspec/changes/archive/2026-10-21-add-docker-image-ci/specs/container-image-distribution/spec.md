## Purpose

Defines a reproducible, versioned CUDA container that installs the complete NexuML workspace and is built and published safely from trusted repository revisions.

## ADDED Requirements

### Requirement: Complete locked workspace image

The container image SHALL install the locked `nexuml` and `nexuml-library` workspace packages together with every optional extra declared by either package. The image SHALL identify its NexuML, Python, Ubuntu, and CUDA versions and SHALL target Linux AMD64.

#### Scenario: Complete environment is built

- **WHEN** the container image is built from a current repository revision
- **THEN** both `nexuml` and `nexuml_library` import from the image environment
- **AND** dependencies selected by every workspace extra are installed from the committed lockfile.

#### Scenario: Lockfile is stale

- **WHEN** project metadata requires a dependency resolution that is not represented by the committed lockfile
- **THEN** the image build fails rather than resolving an uncommitted environment.

### Requirement: GPU-independent image construction

The container image SHALL build and complete its installation checks without access to an NVIDIA GPU. GPU-dependent execution SHALL remain available when the resulting image is run with a compatible NVIDIA driver and container runtime.

#### Scenario: Trusted runner has no GPU assigned

- **WHEN** the image workflow runs on `ubuntu-latest` without exposing a GPU to the build
- **THEN** image construction and package verification complete without invoking GPU hardware.

#### Scenario: User starts a GPU container

- **WHEN** a user runs the published image with NVIDIA GPU access
- **THEN** the installed CUDA-enabled PyTorch environment can discover the GPU without rebuilding the image.

### Requirement: Trusted container workflow

The repository SHALL provide a dedicated container workflow that runs on `ubuntu-latest` for pushes to `main`, semantic `vX.Y.Z` tags, and explicit manual dispatches. The workflow SHALL NOT execute the expensive Docker build from pull-request code.

#### Scenario: Integrated change reaches main

- **WHEN** a commit is pushed to `main`
- **THEN** `ubuntu-latest` builds and verifies the container from that exact commit.

#### Scenario: Maintainer validates a revision manually

- **WHEN** a maintainer dispatches the workflow for a trusted revision
- **THEN** `ubuntu-latest` builds and verifies the image without publishing registry tags.

#### Scenario: Pull request changes the Docker build

- **WHEN** unmerged pull-request code changes the Dockerfile or workspace dependencies
- **THEN** the container workflow does not execute that code or expose registry publication credentials.

### Requirement: Verification precedes publication

The workflow SHALL verify that the built image contains importable core and library packages, exposes the NexuML CLI, reports the expected Python and CUDA versions, and has a usable default command before publishing any tag.

#### Scenario: Image verification succeeds

- **WHEN** the built image passes all package, CLI, version, and startup checks
- **THEN** a trusted push run may publish the verified image.

#### Scenario: Image verification fails

- **WHEN** any required package, CLI, version, or startup check fails
- **THEN** the workflow fails and publishes no image tag for that revision.

### Requirement: GHCR publication uses explicit CUDA variants

Successful trusted push runs SHALL publish to `ghcr.io/nexufed/nexuml` using repository-scoped GitHub credentials. Every published tag SHALL include the exact CUDA patch version, and the workflow SHALL NOT publish an unqualified `latest` tag.

#### Scenario: Main image is published

- **WHEN** a verified commit is pushed to `main` with CUDA `12.8.1`
- **THEN** the workflow publishes `edge-cuda12.8.1` and an immutable `sha-<short-sha>-cuda12.8.1` tag for that commit.

#### Scenario: Stable release image is published

- **WHEN** verified tag `v0.2.0` is pushed with CUDA `12.8.1`
- **THEN** the workflow publishes `0.2.0-cuda12.8.1`, `0.2-cuda12.8.1`, and an immutable commit tag for the same image.

#### Scenario: User omits the image tag

- **WHEN** a user attempts to pull `ghcr.io/nexufed/nexuml` without an explicit tag
- **THEN** the registry does not silently select an unspecified CUDA environment through a project-managed `latest` alias.

### Requirement: Published image carries source metadata

Each published image SHALL include OCI metadata identifying the source repository, exact source revision, NexuML version or channel, license, and image creation time.

#### Scenario: User inspects a published image

- **WHEN** a user or registry inspects the image metadata
- **THEN** the image can be traced to the exact NexuML source revision and its declared release identity.

### Requirement: Docker Hub mirroring remains disabled

The workflow SHALL include a clearly disabled Docker Hub mirror example naming the required username variable and token secret, but SHALL NOT authenticate to or publish on Docker Hub until maintainers deliberately enable and validate that path.

#### Scenario: Normal container workflow runs

- **WHEN** the container workflow builds or publishes a GHCR image
- **THEN** it makes no Docker Hub authentication request and pushes no Docker Hub tag.
