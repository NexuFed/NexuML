## ADDED Requirements

### Requirement: Published Container Installation Path

The installation documentation SHALL present the GHCR CUDA image as an optional complete-environment path after the normal Python package installation path. It SHALL identify the image as containing both NexuML workspace packages and all optional extras.

#### Scenario: User chooses the container environment

- **WHEN** a user follows the container installation section
- **THEN** the docs provide a copy-pasteable pull command for `ghcr.io/nexufed/nexuml:<nexuml-version>-cuda<cuda-version>`
- **AND** recommend an explicit version-and-CUDA tag instead of omitting the tag.

#### Scenario: User runs the image with a GPU

- **WHEN** a user wants CUDA execution from the published image
- **THEN** the docs show the required Docker GPU option
- **AND** state that a compatible NVIDIA host driver and NVIDIA Container Toolkit are prerequisites.

#### Scenario: User evaluates environment compatibility

- **WHEN** a user selects a published image tag
- **THEN** the docs identify the corresponding NexuML, Python, Ubuntu, and CUDA versions and explain the edge and immutable commit tag alternatives.

#### Scenario: User looks for a Docker Hub image

- **WHEN** Docker Hub mirroring has not been enabled
- **THEN** the docs name GHCR as the supported registry and do not claim that a Docker Hub image is available.
