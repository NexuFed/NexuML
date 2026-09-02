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
- Make the implicit loader usable with the advertised `nexuml[library]` installation while preserving DALI as an explicit optimized backend.
- Ensure the published distance-estimator storage configuration reaches its matching feature-store runtime.
- Keep reusable callback defaults free of scenario-specific paths and prevent publication from an unintegrated or stale commit.

**Non-Goals:**

- Do not make PyTorch, TensorDict, Pydantic, or other dependencies required by the public framework API optional merely to minimize the dependency count.
- Do not redesign component discovery, the training lifecycle, evaluation algorithms, or export formats beyond the narrow release-default and storage-wiring corrections described below.
- Do not relax generated model-export environment snapshots; they intentionally capture exact external runtime versions.
- Do not bump serialized component identity versions or export schema versions as part of the package release.
- Do not add upper dependency bounds without evidence of an incompatibility.
- Do not create a custom release/version management framework.
- Do not alias `ram` and `memory`; they name different feature-store and TensorDict-storage families.
- Do not introduce a NexuML run-ID or checkpoint-directory allocation system when Lightning already owns that behavior.

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
2. Core plus library: install both wheels, load installed entry points, list representative base-library components and scenarios, and create a first batch from a lightweight implicit-loader scenario without DALI or repository `PYTHONPATH` entries.

Editable workspace tests remain useful for development but do not satisfy this release gate.

### D8 - Publish with trusted identities after validation

The release workflow uses this order:

```text
main ancestry + exact-commit check
        │
        ▼
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

Production upload uses a protected GitHub environment and PyPI Trusted Publishing through OIDC, not a stored API token. A production tag is accepted only when its commit is contained in `main` and the required source, supported-Python, package, and strict documentation checks succeeded for that exact commit. A manually dispatched TestPyPI path runs only for a frozen integrated candidate and reuses the same build and validation steps. Its installation check selects TestPyPI for the two NexuML projects and the production index for third-party dependencies.

Uploading two PyPI projects cannot be transactional. Core uploads first because the library depends on it; the GitHub release is created only after both uploads succeed. A partial upload is reported explicitly rather than hidden by a successful GitHub release.

### D9 - Teach PyPI installation as the normal path

The primary commands become:

```bash
uv pip install nexuml
uv pip install "nexuml[library]"
```

The documentation explains that core contains the framework and CLI while the library adds bundled components and scenarios. Git checkout and editable installation move entirely to contributor documentation. CUDA-specific PyTorch and DALI index instructions remain explicit advanced installation guidance because package metadata cannot carry uv's repository-local index configuration.

The portable first-run path does not require DALI. It uses the implicit PyTorch loader and explains that scenarios selecting DALI explicitly require the separate platform-specific installation. Because the reusable checkpoint callback no longer fixes a directory, the first-run guide explains how Lightning derives the checkpoint path from the active logger or trainer root and how to locate the resulting run-specific checkpoint.

### D10 - Use PyTorch as the portable loader default

`LoaderSpec` creates `TorchLoader()` by default. PyTorch is already a mandatory framework dependency, so this makes implicit-loader scenarios runnable from the documented `nexuml[library]` installation without adding a second package index or optional native dependency.

Scenarios designed around DALI's file decoding, sharding, or performance behavior continue to select `DaliLoader()` explicitly. The built-in AudioSet and DCASE scenario fragments already do so; the default change affects portable CIFAR, MNIST, and synthetic data fragments that currently omit a loader.

Alternative considered: include DALI in `nexuml[library]`. Rejected because DALI is platform-specific, uses a separate package index, and is intentionally excluded from the runtime `all` extra.

### D11 - Route distance-estimator storage to feature stores

The two evaluation storage families remain distinct:

```text
DistanceEstimatorSpec                 TensorDict evaluation buffers
ram     -> RAMFeatureStore            memory  -> list-backed TensorDict storage
memmap  -> MemmapFeatureStore         memmap  -> LazyMemmapStorage
```

`DistanceEstimatorSpec` exposes a feature-store construction boundary that passes `storage_backend`, `storage_path`, `max_samples`, and `retain_storage` to `create_feature_store`. Configuration round trips preserve `ram`; no `ram`/`memory` translation or compatibility alias is added. This turns the existing storage configuration into executable behavior without inventing a concrete distance estimator or conflating it with visualizer and temporary-buffer storage.

Alternative considered: rename `DistanceEstimatorSpec.storage_backend` to `memory`. Rejected because its fields and intended ownership match the feature-store API, where `ram` is already the implemented and tested spelling.

### D12 - Let Lightning own default checkpoint placement

The reusable `default_callbacks()` keeps its checkpoint policy (`monitor`, `mode`, top-k, filename, and last-checkpoint behavior) but omits `dirpath`. Lightning then derives checkpoint placement from the configured logger's versioned directory or from `Trainer.default_root_dir` when no logger determines one.

This removes CIFAR naming from a generic helper and uses logger versioning for run isolation when available without adding framework-specific path allocation. Scenarios that need a stable explicit directory can still supply their own checkpoint callback rather than changing the shared default.

Alternative considered: add a checkpoint-directory parameter and keep the current CIFAR path at the caller. Rejected because it preserves shared state across repeated runs while duplicating path behavior Lightning already provides.

## Risks / Trade-offs

- Unpinned `nexuml[library]` can select a future incompatible library -> Keep the library's minimum core requirement accurate, follow compatible public APIs, and test the latest published pair before releases.
- Commented dependency declarations can become stale clutter -> Require a reason on each comment and revisit them after the first stable public release.
- Moving dependencies can expose eager optional imports -> Use isolated core/library wheel tests and delay imports only at actual optional feature boundaries.
- Custom PyTorch/NVIDIA indexes are not propagated by wheel metadata -> Test public-index CPU installation and document accelerator-specific installation separately.
- Two PyPI uploads are not atomic -> Validate everything before upload, publish core first, suppress the GitHub release on partial failure, and publish a corrective patch rather than overwriting immutable artifacts.
- Apache-2.0 changes downstream rights materially -> Apply the license consistently to repository files, both distribution artifacts, metadata, and documentation in one release.
- Project names can be claimed before release -> Configure both PyPI projects and trusted publishers before creating the production tag.
- Changing the loader default changes resolved configs for scenarios that omitted a loader -> Treat it as an intentional pre-`0.2.0` breaking correction, update repository-owned snapshots, and test explicit DALI scenarios separately.
- Wiring the previously dormant distance-estimator storage can expand into algorithm design -> Expose only feature-store construction, cover both storage backends, and keep concrete estimator implementations out of this release.
- Lightning-derived checkpoint paths are less fixed than the current CIFAR directory -> Document logger/root placement and verify the first-run guide shows users how to locate `last.ckpt`.
- TestPyPI candidate files are immutable -> Run the `0.2.0` candidate only after integration, runtime fixes, exact-head checks, and publisher setup are complete.
- A matching tag can otherwise be created from any branch -> Protect `main` with required checks and enforce `main` ancestry in the workflow rather than relying only on maintainer convention.

## Migration Plan

1. Update licensing, package metadata, versions, dependency ownership, extras, and runtime version exposure; remove the legacy root requirements file and regenerate `uv.lock`.
2. Switch the implicit loader to PyTorch, wire distance-estimator storage to feature stores, and delegate reusable callback checkpoint paths to Lightning.
3. Update normal-user and contributor documentation and ensure examples distinguish core from the base library, optional DALI, and Lightning-owned checkpoint locations.
4. Add artifact builds and isolated installation tests to ordinary CI, including a DALI-free first-batch check, then inspect both wheel and source-distribution contents.
5. Integrate the stacked changes into `main`, resolve conflicts, and obtain successful source, supported-Python, package, and strict documentation checks for the exact candidate commit.
6. Configure the `nexuml` and `nexuml-library` TestPyPI/PyPI projects, trusted publishers, protected GitHub environments, and production main-ancestry gate.
7. Publish and install the frozen `0.2.0` candidate through TestPyPI.
8. Push `v0.2.0` on the validated `main` commit; publish core first, library second, then create the GitHub release.

PyPI artifacts are immutable. If validation missed a release-breaking defect after publication, yank the affected `0.2.0` project release, leave an explanatory release note, fix forward as `0.2.1`, and do not reuse the version or tag.
