## Context

See `proposal.md` for motivation. The root Dockerfile currently builds from `nvidia/cuda:12.8.1-devel-ubuntu24.04`, installs Python 3.13 and CUDA PyTorch, copies the repository, and installs editable projects into `/env`. It cannot be reproduced from the current checkout because it still reads the removed root `requirements.txt`, installs overlapping dependency sets outside the workspace lock, and attempts to change host `sysctl` values during the build. Its entrypoint is a devcontainer post-create diagnostic script rather than a general container command.

The repository is a public uv workspace with `nexuml` and `nexuml-library`, existing CI on self-hosted runners, and release tags validated as `vX.Y.Z`. GHCR can be written with the repository `GITHUB_TOKEN`; Docker Hub would add an external credential. The image workflow needs a runner with a working Docker daemon and must not publish untrusted pull-request code.

## Goals / Non-Goals

**Goals:**

- Produce one Linux AMD64 CUDA development/training image containing every workspace package and extra from `uv.lock`.
- Validate integrated and manually selected trusted revisions on `ubuntu-latest` and publish trusted pushes to GHCR.
- Make app and CUDA compatibility visible in every supported image tag.
- Give users a copy-pasteable, GPU-aware pull and run path.

**Non-Goals:**

- Do not produce a CPU image, runtime-minimal image, or multi-architecture manifest in this change.
- Do not enable Docker Hub publication, keyless signing, SBOM policy gates, or vulnerability-policy enforcement.
- Do not run the expensive Docker build for pull requests.
- Do not replace Python package installation as the primary user installation path.

## Decisions

### D1 - Use a separate trusted-ref workflow on `ubuntu-latest`

Create `.github/workflows/docker.yml` rather than adding registry permissions and a large CUDA build to the existing source/package CI. Trigger it for `main`, semantic release tags, and manual dispatch. Main and release-tag pushes may publish; manual dispatch builds and verifies only.

Pull requests are deliberately excluded because the all-extras CUDA build is expensive and does not need registry credentials before integration. The GitHub-hosted runner is ephemeral and provides the Docker daemon required by Buildx.

Alternative considered: add a Docker job to `.github/workflows/ci.yml`. Rejected because it would place an expensive, privileged build on every public pull request and mix package CI with registry permissions.

### D2 - Publish GHCR directly and leave Docker Hub dormant

The active image name is lowercase `ghcr.io/nexufed/nexuml`. The workflow grants `contents: read` and `packages: write`, logs in with `github.actor` and `GITHUB_TOKEN` only for publishable push events, and performs no registry login for manual validation.

The workflow includes a short commented Docker Hub image target and login block using `vars.DOCKERHUB_USERNAME` and `secrets.DOCKERHUB_TOKEN`. It remains visibly unsupported and is not mentioned as an available registry in user docs. Enabling it later requires a Docker Hub repository, scoped access token, tag-parity verification, and a deliberate follow-up change.

Alternative considered: publish both registries now. Rejected because no Docker Hub namespace or credential has been configured, and an unverified mirror adds a second partial-publication path.

### D3 - Treat CUDA as an image variant of the NexuML release

Application version precedes the CUDA suffix, following the common `<application-version>-<variant>` convention:

```text
source event        published tags
main                edge-cuda12.8.1
main                sha-<short-sha>-cuda12.8.1
v0.2.0              0.2.0-cuda12.8.1
v0.2.0              0.2-cuda12.8.1
v0.2.0              sha-<short-sha>-cuda12.8.1
```

The full semantic tag is the documented stable pull target. The minor alias receives compatible patch releases, `edge` tracks `main`, and the SHA tag is immutable. A bare `latest` tag is omitted because it hides the CUDA contract and silently changes both application and environment versions.

Python 3.13, Ubuntu 24.04, the exact source revision, and the base image are recorded as OCI labels and in documentation rather than extending the tag. If those dimensions gain supported variants later, the tag policy must be revisited.

Alternative considered: publish a CUDA-only tag such as `cuda12.8.1`. Rejected because it does not identify which NexuML release it contains.

### D4 - Install the uv workspace once from its lockfile

Set `UV_PROJECT_ENVIRONMENT=/env` and use uv's project interface rather than independent `uv pip install` commands. The dependency layer uses the workspace lock and excludes workspace source; after source is copied, the final locked sync uses both `--all-packages` and `--all-extras`. This installs the root and library workspace members and every declared optional extra, including platform-specific extras, without a separate requirements file or duplicate editable-library command.

The Dockerfile should follow uv's two-layer workspace pattern: copy lock/project metadata, perform a frozen dependency sync without workspace source, copy the source tree, then perform the locked complete sync. Pin the copied uv image to an explicit release rather than `latest`. Keep the exact CUDA base tag as the build argument default and pass the same value from CI for tag generation.

Remove build-time `sysctl` writes because images cannot set host kernel policy. Any required inotify tuning belongs in host/runtime documentation. Remove the devcontainer post-create script as the image entrypoint; `.devcontainer/devcontainer.json` already invokes that script explicitly. Use a normal shell command for interactive container startup.

Alternative considered: preserve the current sequential pip-compatible installs. Rejected because they duplicate resolution, bypass the committed workspace environment, and cannot express the requested all-workspace/all-extras contract in one operation.

### D5 - Verify a locally loaded image before pushing its tags

Build one AMD64 image with Buildx and load it into the runner's Docker daemon. Before registry login or push, run focused checks with GPU access disabled:

- import `nexuml`, `nexuml_library`, and CUDA-enabled `torch`;
- verify the installed NexuML version against project metadata on release tags;
- verify Python 3.13 and `torch.version.cuda == "12.8"` for the CUDA 12.8.1 image;
- run `nexuml --help`; and
- start the image's default shell command non-interactively.

Only after those checks pass does a publishable push log in and push every metadata-generated tag. This ordering prevents a known-bad tag from being published. Manual dispatch stops after verification.

Alternative considered: let `docker/build-push-action` push immediately and test by pulling afterward. Rejected because failed verification would leave a published tag requiring cleanup.

### D6 - Keep image metadata and tag inputs centralized in the workflow

Use Docker's metadata action to generate tags and standard OCI labels from the Git ref. Define the expected CUDA version once in workflow environment, pass it as the Docker build argument, and use it as the tag suffix. Add a focused check that the built image reports the expected value so a Dockerfile/workflow mismatch fails before publication.

The exact NexuML version comes from a semantic Git tag for stable images and from source revision metadata for edge images. Release publication accepts only the existing `vX.Y.Z` convention.

### D7 - Document only verified, public pull paths

Add a concise container section to the normal installation documentation and a short pointer in the README. Show an explicit stable pull command and a GPU run command using `--gpus all`. Explain that the image contains the full all-extras development/training environment and is much larger than the normal Python installation.

The GHCR package must be linked to the repository and made public before anonymous pull instructions are considered operational. Docker Hub is omitted from user docs while its workflow example remains disabled.

## Risks / Trade-offs

- [The all-extras CUDA image may exhaust runner disk or exceed registry upload limits] -> Check free disk before building, keep a single architecture, use bounded Buildx caching, and record the resulting compressed image size.
- [A trusted build can publish an unintended image] -> Restrict publication to protected `main` and validated tags; expose no Docker Hub secret.
- [CUDA version text can drift between the workflow and image] -> Pass the workflow value as the build argument and verify the installed runtime before push.
- [Some optional extras may become incompatible with Python 3.13 or CUDA 12.8] -> Treat the locked all-extras build as the compatibility gate and fail publication rather than dropping an extra silently.
- [Commented Docker Hub configuration can become stale] -> Keep it minimal, label it unsupported, and require a follow-up validation before activation.
- [GHCR tag publication is not transactional across multiple tags] -> Push only after local verification and retain the immutable SHA tag as the revision identity.

## Migration Plan

1. Repair and locally validate the Dockerfile contract without changing any registry state.
2. Add the trusted-ref workflow and run its build, startup, import, CLI, and CUDA checks locally without publishing.
3. Merge to protected `main`; its verification gate publishes the first edge and SHA tags only after the image checks pass, then link the GHCR package to the repository and set public visibility.
4. Once the workflow exists on the default branch, dispatch it manually on `ubuntu-latest` and verify that the manual path records image size and disk headroom without publishing.
5. Verify an anonymous pull from a clean environment, then verify semantic CUDA tags on the next validated `vX.Y.Z` release.

Rollback consists of disabling the workflow, removing affected GHCR tags or package versions, and reverting the container documentation. No application data or package-format migration is involved.
