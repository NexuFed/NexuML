## 1. Reproducible Container Environment

- [x] 1.1 Replace the stale requirements-based and duplicate editable installs with a locked uv workspace sync using `--all-packages --all-extras`, and verify an AMD64 image build fails with a stale lockfile but succeeds with the committed `uv.lock`.
- [x] 1.2 Restructure the Dockerfile into cacheable dependency and source layers, pin the uv source image to an explicit release, and verify a source-only rebuild reuses the external dependency layer.
- [x] 1.3 Remove build-time `sysctl` mutations and the devcontainer diagnostic entrypoint, retain the devcontainer's explicit post-create hook, and verify the image builds without GPU access and starts a non-interactive Bash command successfully.
- [x] 1.4 Add image checks for `nexuml`, `nexuml_library`, CLI help, Python 3.13, and CUDA 12.8, then verify all checks pass inside the locally built image with GPU access disabled.

## 2. Trusted Image Workflow

- [x] 2.1 Add `.github/workflows/docker.yml` for `main`, semantic `vX.Y.Z` tags, and manual dispatch on `ika-runner`, with concurrency cancellation and no pull-request trigger; verify the parsed workflow exposes only those events and runner labels.
- [x] 2.2 Configure Docker metadata and OCI labels for `ghcr.io/nexufed/nexuml`, generating CUDA-suffixed edge, semantic minor/full, and immutable SHA tags; verify fixture refs for `main` and `v0.2.0` produce the tag sets required by the spec and no bare `latest`.
- [x] 2.3 Build and load one Linux AMD64 image, execute all image checks before registry authentication, and conditionally publish only successful `main` and release-tag push runs using `GITHUB_TOKEN`; verify manual dispatch performs no login or push and a forced failed check reaches no publish step.
- [x] 2.4 Add a minimal disabled Docker Hub image target and login example using `vars.DOCKERHUB_USERNAME` and `secrets.DOCKERHUB_TOKEN`, and verify active workflow evaluation contains no Docker Hub authentication or push.

## 3. Container Documentation

- [x] 3.1 Add a README container pointer and an installation-guide container section showing `ghcr.io/nexufed/nexuml:0.2.0-cuda12.8.1`, `docker pull`, and `docker run --gpus all`; verify every command is copy-pasteable and uses an explicit tag.
- [x] 3.2 Document the image's NexuML, Python, Ubuntu, CUDA, Linux AMD64, base-library, and all-extras contract together with NVIDIA driver/Container Toolkit prerequisites and edge/SHA tag meanings; verify the docs do not claim Docker Hub or a bare `latest` tag is available.
- [x] 3.3 Run `DISABLE_MKDOCS_2_WARNING=true uv run mkdocs build --strict` and verify the updated installation pages and README links pass strict documentation validation.

## 4. End-to-End Validation And Publication

- [ ] 4.1 Merge through protected `main`, verify the image checks gate publication of matching `edge-cuda12.8.1` and `sha-<short-sha>-cuda12.8.1` images, then link the GHCR package to `NexuFed/NexuML` and make it public.
- [ ] 4.2 Run the now-default-branch workflow manually on `ika-runner`, verify the build and image checks pass without a GPU or registry publication, and record the compressed image size and available runner disk headroom.
- [ ] 4.3 Pull the main SHA image anonymously in a clean environment, inspect its OCI source/revision/version/license labels, and run the documented non-GPU and GPU startup checks.
- [ ] 4.4 On the validated `v0.2.0` release commit, verify `0.2.0-cuda12.8.1`, `0.2-cuda12.8.1`, and the SHA tag resolve to the same tested image content while an unqualified `latest` tag remains unpublished.
- [x] 4.5 Run `openspec validate add-docker-image-ci --strict` and verify the implementation, documentation, and recorded workflow evidence satisfy every container-distribution and user-install scenario.
