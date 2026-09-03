## Why

NexuML has a CUDA development image, but the image is neither validated by CI nor available from a registry, so users must build a stale and currently broken Dockerfile locally. A reproducible GHCR image would provide a documented, versioned environment containing both workspace packages and every optional dependency.

## What Changes

- Repair the Docker build so it installs the locked uv workspace, including `nexuml`, `nexuml-library`, and all extras, without relying on the removed root `requirements.txt` or build-time host kernel changes.
- Add a dedicated GitHub Actions workflow that uses `ika-runner` to build and verify the Linux AMD64 CUDA image on trusted `main`, semantic release-tag, and manual runs.
- Publish successful `main` and semantic release-tag builds to `ghcr.io/nexufed/nexuml` using `GITHUB_TOKEN`; manual runs validate without publishing.
- Generate explicit image tags that combine the NexuML release with the CUDA patch version, such as `0.2.0-cuda12.8.1`, plus appropriate edge, release-series, and immutable commit tags without relying on an ambiguous bare `latest` tag.
- Include a disabled, commented Docker Hub login and image-target example so maintainers can enable mirroring later without claiming that Docker Hub publication is active.
- Document how to pull and run the GHCR image, recommend an explicit version-and-CUDA tag, and identify its Python, Ubuntu, CUDA, and all-extras environment contract.

## Capabilities

### New Capabilities

- `container-image-distribution`: Defines the reproducible all-extras CUDA image, trusted CI build and GHCR publication behavior, image tags, and image verification contract.

### Modified Capabilities

- `user-install-docs`: Adds the published container image as an optional installation path with explicit pull and GPU runtime instructions.

## Impact

- Container build: `Dockerfile` and `.dockerignore`.
- Automation: a new `.github/workflows/docker.yml` workflow running on `ika-runner`, with GHCR package permissions and dormant Docker Hub configuration comments.
- Documentation: `README.md` and the installation documentation under `docs/start/`.
- External systems: GitHub Container Registry package visibility and repository linkage must be configured for anonymous public pulls; Docker Hub remains disabled.
- Operations: the `ika-runner` must provide Docker with Buildx and enough disk space for the CUDA development image.
