## Why

NexuML cannot be published safely to PyPI while its two distributions depend on each other, its license prohibits redistribution, and releases do not build or validate installable artifacts. The `0.2.0` release should establish a clean public packaging contract in which the framework works without the optional base library and both packages can be installed and released through standard Python tooling.

## What Changes

- Replace the proprietary license with Apache License 2.0 for both distributions and publish SPDX-compatible package metadata.
- Release both `nexuml` and `nexuml-library` as version `0.2.0`, expose the installed core version consistently, and require release tags to match distribution metadata.
- **BREAKING** Remove `nexuml-library` from the mandatory `nexuml` dependencies so a default core installation no longer installs built-in components and scenarios.
- Add an unpinned `nexuml[library]` convenience extra while retaining the one-way `nexuml-library -> nexuml>=0.2` framework dependency.
- Move required dependencies to the distribution or optional feature that imports them, preserve intentional compatibility ranges such as Ray's, and comment out currently unused declarations with a short reason rather than deleting them.
- Remove the repository-root legacy `requirements.txt`; keep `pyproject.toml` as public dependency metadata and `uv.lock` as the exact development/CI resolution.
- Keep generated model-export `requirements.txt` snapshots, serialized component versions, and export schema versions unchanged because they are separate compatibility contracts.
- Add complete PyPI metadata, build and metadata checks, clean wheel/sdist installation tests, core-only and library-discovery smoke tests, and trusted PyPI publishing from validated release tags.
- Replace Git-based normal-user installation instructions with PyPI commands while retaining source-checkout instructions for contributors.

## Capabilities

### New Capabilities

- `python-package-distribution`: Defines licensing, versioning, dependency ownership, optional-library installation, and wheel/sdist contracts for the `nexuml` and `nexuml-library` distributions.
- `pypi-release-publishing`: Defines validated, trusted publication of both distributions to PyPI from release tags.

### Modified Capabilities

- `user-install-docs`: Changes normal-user installation from Git URLs to the published core package and optional base-library extra on PyPI.

## Impact

- Packaging and dependency metadata: `pyproject.toml`, `library/pyproject.toml`, `uv.lock`, and removal of the root `requirements.txt`.
- Licensing and version exposure: `LICENSE`, a standalone `library/LICENSE`, and `src/nexuml/__init__.py`.
- Release validation and publication: `.github/workflows/ci.yml`, `.github/workflows/release.yml`, and isolated distribution smoke tests.
- User documentation: `README.md`, `library/README.md`, `docs/start/install.md`, and related first-run links or examples.
- Default installs no longer include `nexuml-library`; users who want bundled components install `nexuml[library]` or `nexuml-library` explicitly.
- Existing exact model-export environment snapshots and the bounded Ray compatibility range remain intact.
