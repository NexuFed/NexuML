# Proposal

## Why

Releasing currently requires manually synchronizing two Python project versions, Studio metadata, and lockfiles before creating a matching tag. Maintainers should instead choose a stable Git tag for a validated commit and let CI derive the shared release version without committing version bumps back to the repository.

## What Changes

- Treat protected `vX.Y.Z` tags as the release-version source for `nexuml`, `nexuml-library`, and `@nexufed/nexuml-studio`.
- Prepare owned package-version fields and lockfile metadata in disposable CI checkouts before building. Keep valid checked-in development metadata; do not introduce literal `SET_BY_CI` values or Git-dependent local installs.
- Run the existing mandatory CI checks on the stamped tag checkout, build the release candidate once, and test the actual wheels, source distributions, and npm tarball.
- Bind candidate artifacts to the tag, peeled source commit, successful CI run/attempt, artifact IDs, and file integrity. Publishing waits for this candidate rather than selecting pre-tag branch artifacts by commit alone.
- Preserve protected, tokenless publication: tag pushes publish to TestPyPI/npm `next`; publishing the matching GitHub Release publishes the same Python files to PyPI and promotes the existing npm version to `latest`.
- **BREAKING (release operations):** manual package-publication dispatch must target an existing stable release tag, not an unversioned branch. Publishing rejects missing, expired, ambiguous, or mismatched candidates rather than rebuilding or choosing a newer substitute.
- Leave dependency compatibility ranges, the API interface version, historical documentation, changelog history, and unrelated dependency versions unchanged. Version stamping does not assert compatibility with a new minor/major line.

## Capabilities

### New Capabilities

None. Shared versioning and registry publication already belong to the capabilities below.

### Modified Capabilities

- `pypi-release-publishing`: derive Python release versions from tags, validate tag-stamped candidates, select immutable tag-and-commit-specific artifacts, and retain candidate-to-production identity.
- `studio-local-distribution`: prepare the shared Python/npm release identity automatically, test the stamped tarball, and publish/promote only the matching tag-specific candidate.

## Impact

- `.github/workflows/ci.yml`, `.github/workflows/release.yml`, and `.github/workflows/publish-studio.yml`; trusted publisher filenames and environment identities remain unchanged.
- Release preparation/validation helpers under `scripts/`, including `check-release-versions.py`, and a shared read-only candidate resolver used by both publishing workflows.
- Temporary release-checkout edits to `pyproject.toml`, `library/pyproject.toml`, `uv.lock`, `studio/package.json`, and `studio/package-lock.json`; no generated version-bump commits.
- Packaging/workflow tests, installed-version assertions, and the operator guide at `docs/src/content/docs/development/release.mdx`.
- No public runtime API changes, new versioning dependency, new registry credential, registry-setting mutation, or change to institute-runner/platform coverage policy.
