## 1. Licensing and Version Metadata

- [x] 1.1 Replace the root proprietary license with the unmodified Apache License 2.0 text, add the same `library/LICENSE`, and verify both files contain the standard Apache-2.0 terms.
- [x] 1.2 Change both PEP 621 license declarations to the SPDX expression `Apache-2.0`, retain `license-files = ["LICENSE"]`, and verify both built metadata records identify Apache-2.0.
- [x] 1.3 Set `nexuml` and `nexuml-library` to version `0.2.0`, derive `nexuml.__version__` from installed distribution metadata, and verify an installed core wheel reports `0.2.0` through both interfaces.
- [x] 1.4 Add accurate maintainer, repository, documentation, issue-tracker, release/changelog URL, development-status, and supported-Python metadata to both projects, and verify package metadata validation and README rendering succeed.

## 2. Dependency Ownership

- [x] 2.1 Remove `nexuml-library` from mandatory core dependencies, add unpinned `nexuml-library` under the core `library` extra, and verify resolved core metadata has no mandatory library requirement.
- [x] 2.2 Change the library's framework requirement to the one-way minimum `nexuml>=0.2`, and verify the two built dependency graphs contain no mandatory cycle or exact cross-package pin.
- [x] 2.3 Audit every declared dependency against direct, delayed, and plugin imports; move library-owned requirements such as TorchAudio, TorchVision, and TorchMetrics out of mandatory core metadata, and verify each active declaration has a current runtime owner.
- [x] 2.4 Keep core-owned optional integrations in their feature extras, move audio/data/pretrained/evaluation requirements to the library metadata where used, and verify representative feature imports either succeed with their extra or fail with an actionable extra-specific message.
- [x] 2.5 Comment out confirmed-unused root declarations for `rapids`, `numba`, and `psutil` with concise reasons, and verify none appears in the core wheel's `Requires-Dist` metadata.
- [x] 2.6 Comment out confirmed-unused library declarations for `huggingface_hub`, `ffmpeg`, `einops`, and `omegaconf` with concise reasons, and verify none appears in the library wheel's `Requires-Dist` metadata.
- [x] 2.7 Check TorchCodec's actual Torch/TorchAudio compatibility requirement, replace `torchcodec==0.10.0` with the broadest proven feature constraint or document why the exact constraint must remain, and verify the relevant feature installation resolves.
- [x] 2.8 Preserve `ray[default,train]>=2.57,<2.59`, keep DALI separately requested, exclude development tooling and DALI from the user runtime `all` extra, and verify the published extras match those boundaries.
- [x] 2.9 Delete the repository-root `requirements.txt`, update any repository references to use project metadata or the lockfile, and verify generated model-export `requirements.txt` behavior remains covered by export tests.
- [x] 2.10 Regenerate `uv.lock` from the revised workspace metadata and verify `uv lock --check` and a clean `uv sync --all-extras` succeed.

## 3. Import and Installation Boundaries

- [x] 3.1 Trace imports reached by `import nexuml`, core configuration/compilation, and CLI startup after dependency moves; delay imports only at optional feature boundaries and verify base imports do not require library-owned packages.
- [x] 3.2 Add a subprocess or isolated-environment core-only test that installs the built core wheel without repository `PYTHONPATH`, verifies `nexuml-library` is absent, imports `nexuml`, checks version `0.2.0`, and runs CLI help.
- [x] 3.3 Extend the core-only artifact test with one lightweight framework operation that uses no library component and verify the existing core/base-library separation requirement is satisfied.
- [x] 3.4 Add an isolated core-plus-library test that installs both built wheels, loads the `nexuml.libraries` entry point, and verifies representative components and scenarios are discoverable.
- [x] 3.5 Add focused missing-extra tests for any optional import boundary changed during the audit and verify errors identify the extra or package the user must install.

## 4. Distribution Artifacts

- [x] 4.1 Build wheel and source distributions for the root and library projects into separate output directories and verify all four expected artifacts are produced.
- [x] 4.2 Validate every artifact's metadata and long description with a standard package metadata checker and verify no warnings or errors remain.
- [x] 4.3 Inspect wheel and source-distribution contents and verify each contains its package source, README metadata, Apache license, and required non-Python package data such as library YAML resources.
- [x] 4.4 Build a wheel from each generated source distribution in a clean environment and verify the rebuilt wheel passes the same metadata and installation checks.
- [ ] 4.5 Test the core wheel using public-index CPU dependencies on a clean hosted Linux runner and verify repository-local PyTorch or NVIDIA index configuration is not required for the default installation.

## 5. User Documentation

- [x] 5.1 Update the root README installation section and badges to present `uv pip install nexuml` and `uv pip install "nexuml[library]"`, and verify the rendered README explains the core/library distinction.
- [x] 5.2 Update `docs/start/install.md` and first-run references to use PyPI for normal users while retaining Git checkout and editable installs only in contributor guidance, and verify all commands use shell-safe ASCII quoting.
- [x] 5.3 Expand `library/README.md` with its purpose, direct and convenience installation commands, framework compatibility policy, and discovery entry point, and verify it renders as valid package long-description Markdown.
- [x] 5.4 Document CUDA-specific PyTorch selection and DALI's platform/index requirements separately from default installation, and verify the default instructions require no custom package index.
- [x] 5.5 Run `uv run mkdocs build --strict` and verify the updated documentation builds without warnings or broken links.

## 6. CI Artifact Gates

- [x] 6.1 Add a hosted-runner CI job that builds both distributions and runs metadata, contents, source-rebuild, core-only, and core-plus-library artifact checks; verify the job never imports from editable workspace paths.
- [x] 6.2 Upload the validated wheel and source-distribution files as CI artifacts and verify downstream release jobs consume those exact files instead of rebuilding them.
- [x] 6.3 Add a release version gate that compares a semantic `vX.Y.Z` tag with both project versions and verify mismatched tags fail before artifact upload.
- [x] 6.4 Keep the existing source tests, Ruff, type checks, and documentation checks required before artifact publication, and verify the release job depends on all relevant successful gates.

## 7. Trusted Publishing

- [x] 7.1 Add a manually dispatched TestPyPI job using a protected `testpypi` environment and OIDC trusted publishing, and verify it reuses validated CI artifacts without a stored API token.
- [x] 7.2 Add a protected production `pypi` environment and OIDC publication job for release tags, publish core before the dependent library, and verify no upload step can run before all validation gates pass.
- [x] 7.3 Move GitHub release creation after both PyPI uploads, attach or link the validated distribution artifacts, and verify a failed or partial upload cannot produce a successful GitHub release.
- [x] 7.4 Configure or document the required PyPI and TestPyPI project ownership, trusted-publisher identities, protected environments, and approvals for both package names; verify maintainers can audit the configuration without repository secrets.
- [ ] 7.5 Publish `0.2.0` candidates through the TestPyPI path and verify core-only and `nexuml[library]` installation using TestPyPI for NexuML projects and the production index for third-party dependencies.

## 8. Final Verification

- [x] 8.1 Run the normal non-slow test suite and all new package-artifact tests, and verify they pass without external datasets or a GPU.
- [x] 8.2 Run Ruff and the repository type checker over both source trees and verify no errors are introduced by dependency or version changes.
- [x] 8.3 Run strict OpenSpec validation for `prepare-pypi-release` and verify the proposal, delta specs, design, and completed task evidence remain coherent.
- [x] 8.4 Review the final package metadata and release workflow against `python-package-distribution` and `pypi-release-publishing`, and verify no production `v0.2.0` tag is created until a maintainer explicitly approves publication.
