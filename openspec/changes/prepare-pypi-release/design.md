## Context

See `proposal.md` for motivation. The repository is a uv workspace containing two Hatchling distributions:

```text
nexuml 0.1.0 ───────────────▶ nexuml-library 0.1.0
      ▲                              │
      └──────── nexuml>=0.1.0 ───────┘
```

This mandatory cycle contradicts the existing core/base-library separation contract, which says framework commands remain available when only core is installed. Installed-library discovery already checks whether `nexuml_library` exists before scanning it, so removing the core-to-library dependency does not require a new plugin mechanism.

The root package currently mixes framework dependencies, base-library dependencies, optional feature dependencies, and unused declarations. A separate legacy `requirements.txt` carries a third, divergent dependency list. The exact versions in `uv.lock` and in generated model-export dependency manifests serve different reproducibility purposes and must not be mistaken for public distribution constraints.

The existing release workflow creates a GitHub release for any `v*` tag but does not build, inspect, install, or publish either Python distribution. Both package names currently appear available on PyPI, but availability is not guaranteed until publication.

## Goals / Non-Goals

**Goals:**

- Make `pip`/`uv` install the framework without the base library unless the user asks for it.
- Give each runtime dependency one clear owner and keep published constraints abstract enough for downstream resolution.
- Produce complete, independently installable wheels and source distributions.
- Publish both version `0.2.0` distributions through a validated, trusted release path.
- Make Apache-2.0 licensing and PyPI installation unambiguous in artifacts and documentation.

**Non-Goals:**

- Do not make PyTorch, TensorDict, Pydantic, or other dependencies required by the public framework API optional merely to minimize the dependency count.
- Do not redesign component discovery, training, export formats, or runtime behavior.
- Do not relax generated model-export environment snapshots; they intentionally capture exact external runtime versions.
- Do not bump serialized component identity versions or export schema versions as part of the package release.
- Do not add upper dependency bounds without evidence of an incompatibility.
- Do not create a custom release/version management framework.

## Decisions

### D1 - Use a one-way optional library dependency

The target graph is:

```text
nexuml[library] ──optional──▶ nexuml-library
                                  │
                                  └──requires──▶ nexuml>=0.2
```

`nexuml` removes `nexuml-library` from mandatory dependencies and adds:

```toml
library = ["nexuml-library"]
```

The convenience extra deliberately has no exact library version. `nexuml-library` retains a minimum compatible core dependency, `nexuml>=0.2`, because it imports framework APIs. No dependency points from core back to the library.

Alternative considered: rename the framework to `nexuml-core` and make `nexuml` a meta-package. Rejected because the existing package already has the correct framework/library boundary and discovery behavior; renaming adds migration cost without solving an unmet requirement.

### D2 - Assign dependencies by runtime ownership

The dependency audit uses these rules:

- A package imported by ordinary framework imports or normal training/CLI paths remains a core dependency.
- A package used only by `nexuml_library` moves to the library distribution.
- A package used only when a named optional feature is invoked belongs to that feature extra and is imported lazily with an actionable error.
- Development lock versions remain in `uv.lock`, not in wheel metadata.
- Exact or upper-bounded public constraints remain only when an existing documented compatibility requirement justifies them. The bounded Ray API range is preserved.

The initial known moves include `torchaudio`, `torchvision`, and `torchmetrics` out of mandatory core metadata because their code ownership is in the base library. Existing audio, data, pretrained, and evaluation dependencies are audited with the same ownership rule rather than copied into both distributions. Core-owned optional integrations such as tracking, tuning, export, Ray, S3, and DALI remain core extras.

The `all` extra represents user runtime features only: it excludes development tooling and does not force hardware/index-specific DALI installation. Development dependencies remain separately requested by contributors.

Alternative considered: remove every non-imported package automatically. Rejected because integrations can rely on delayed imports or runtime plugin calls. Every removal or move must be checked against source usage and an isolated artifact test.

### D3 - Preserve unused declarations as comments for this release

At the user's request, declarations confirmed unused are commented out in their existing TOML files with a short reason instead of being deleted. Initial candidates are:

```text
root:     rapids, numba, psutil
library:  huggingface_hub, ffmpeg, einops, omegaconf
```

Dependencies that remain necessary but belong elsewhere are moved, not commented out. `torchcodec` is retained in the feature that invokes it until its PyTorch/TorchAudio compatibility requirements are verified; an exact pin is relaxed only if the supported range is proven.

Git history would normally be enough to recover removed declarations, but comments are retained because that behavior was explicitly requested. Each comment records why the package is inactive so it is not mistaken for supported metadata.

### D4 - Use standard Apache-2.0 package licensing

The root `LICENSE` is replaced with the unmodified Apache License 2.0 text. A matching `library/LICENSE` is added because the library builds from its own project root and cannot rely on a parent file being included. Both `pyproject.toml` files use the SPDX expression:

```toml
license = "Apache-2.0"
license-files = ["LICENSE"]
```

Project metadata identifies NexuFed AI as maintainer/copyright owner and includes repository, documentation, issue tracker, and changelog/release URLs. No unnecessary `NOTICE` file is created unless the project has notices that Apache-2.0 requires downstream users to retain.

### D5 - Keep versions static and validate them centrally

Both distributions are set to `0.2.0` for the initial public release. `nexuml.__version__` reads the installed `nexuml` distribution metadata via the standard library instead of duplicating a third hard-coded value. The release workflow reads both project versions and requires the tag to equal `v<version>` for each.

Git-derived dynamic version plugins are not introduced. Static PEP 621 metadata plus one release consistency check is easier to inspect and works for source distributions without repository history.

### D6 - Keep one dependency source for each purpose

The repository-root `requirements.txt` is deleted. Dependency roles become:

```text
pyproject.toml                     public compatibility and extras
library/pyproject.toml             base-library compatibility and extras
uv.lock                            exact repository development/CI resolution
exported-model requirements.txt    exact model runtime snapshot
```

Removing the root file does not alter `src/nexuml/core/export.py` or its generated `requirements.txt` contract.

### D7 - Validate built artifacts, not editable checkouts

CI builds wheel and source distributions for each project into separate directories, validates metadata/README rendering, inspects required files, and proves that each source distribution can produce a wheel.

Two isolated smoke paths install only built artifacts:

1. Core-only: install the core wheel with public dependencies, verify `nexuml-library` is absent, import `nexuml`, inspect version `0.2.0`, run CLI help, and exercise a framework operation that needs no library component.
2. Core plus library: install both wheels, load installed entry points, and list representative base-library components and scenarios without repository `PYTHONPATH` entries.

Editable workspace tests remain useful for development but do not satisfy this release gate.

### D8 - Publish with trusted identities after validation

The release workflow uses this order:

```text
tag/version check
        │
        ▼
build both distributions
        │
        ▼
metadata/content/install checks
        │
        ▼
PyPI Trusted Publishing
        │
        ▼
GitHub release + artifacts
```

Production upload uses a protected GitHub environment and PyPI Trusted Publishing through OIDC, not a stored API token. A manually dispatched TestPyPI path reuses the same build and validation steps. Its installation check selects TestPyPI for the two NexuML projects and the production index for third-party dependencies.

Uploading two PyPI projects cannot be transactional. Core uploads first because the library depends on it; the GitHub release is created only after both uploads succeed. A partial upload is reported explicitly rather than hidden by a successful GitHub release.

### D9 - Teach PyPI installation as the normal path

The primary commands become:

```bash
uv pip install nexuml
uv pip install "nexuml[library]"
```

The documentation explains that core contains the framework and CLI while the library adds bundled components and scenarios. Git checkout and editable installation move entirely to contributor documentation. CUDA-specific PyTorch and DALI index instructions remain explicit advanced installation guidance because package metadata cannot carry uv's repository-local index configuration.

## Risks / Trade-offs

- Unpinned `nexuml[library]` can select a future incompatible library -> Keep the library's minimum core requirement accurate, follow compatible public APIs, and test the latest published pair before releases.
- Commented dependency declarations can become stale clutter -> Require a reason on each comment and revisit them after the first stable public release.
- Moving dependencies can expose eager optional imports -> Use isolated core/library wheel tests and delay imports only at actual optional feature boundaries.
- Custom PyTorch/NVIDIA indexes are not propagated by wheel metadata -> Test public-index CPU installation and document accelerator-specific installation separately.
- Two PyPI uploads are not atomic -> Validate everything before upload, publish core first, suppress the GitHub release on partial failure, and publish a corrective patch rather than overwriting immutable artifacts.
- Apache-2.0 changes downstream rights materially -> Apply the license consistently to repository files, both distribution artifacts, metadata, and documentation in one release.
- Project names can be claimed before release -> Configure both PyPI projects and trusted publishers before creating the production tag.

## Migration Plan

1. Update licensing, package metadata, versions, dependency ownership, extras, and runtime version exposure; remove the legacy root requirements file and regenerate `uv.lock`.
2. Update normal-user and contributor installation documentation and ensure examples distinguish core from the base library.
3. Add artifact builds and isolated installation tests to ordinary CI, then inspect both wheel and source-distribution contents.
4. Configure the `nexuml` and `nexuml-library` TestPyPI/PyPI projects, trusted publishers, and protected GitHub environments.
5. Publish and install the `0.2.0` candidates through TestPyPI.
6. Push `v0.2.0`; publish core first, library second, then create the GitHub release.

PyPI artifacts are immutable. If validation missed a release-breaking defect after publication, yank the affected `0.2.0` project release, leave an explanatory release note, fix forward as `0.2.1`, and do not reuse the version or tag.
