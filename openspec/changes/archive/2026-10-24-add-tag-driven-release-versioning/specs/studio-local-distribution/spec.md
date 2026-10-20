# Spec Delta

## MODIFIED Requirements

### Requirement: Immutable coordinated npm release
Tag CI SHALL build the stamped Studio tarball once, retain it immutably, and test those exact bytes against the stamped Python wheels. The required gate SHALL reject failed, skipped, or cancelled mandatory checks. npm publication and promotion SHALL use the same verified tag-and-commit-specific candidate as Python, without rebuilding. Untrusted fork code SHALL NOT execute on institute runners.

#### Scenario: Shared candidate tag is pushed
- **WHEN** a shared stable `vX.Y.Z` tag is pushed for an integrated commit with successful branch CI
- **THEN** publication waits for successful version-stamped tag CI and verifies protected-main membership, candidate identity, file integrity, and package metadata
- **AND** after protected-environment approval it publishes that tested tarball to npm `next` using Trusted Publishing and provenance
- **AND** `latest` remains unchanged while the matching Python packages are tested through TestPyPI

#### Scenario: Shared release is published
- **WHEN** the matching GitHub Release is published and npm promotion is approved
- **THEN** the workflow revalidates the original tag-specific candidate and compares the registry version's integrity to its tested tarball
- **AND** only that existing matching version is promoted to `latest`, without rebuilding or uploading another package
- **AND** missing or mismatched registry packages block promotion

#### Scenario: Manual candidate publication
- **WHEN** a maintainer dispatches Publish Studio against an existing stable tag with an unpublished, validated candidate
- **THEN** the same candidate, version, and integrity checks apply
- **AND** publication targets `next`, not `latest`

#### Scenario: Manual npm publication targets a branch
- **WHEN** Publish Studio is dispatched against `main` or another unversioned branch
- **THEN** it fails without publishing or inferring a release version from development metadata

#### Scenario: No verified tag-specific candidate is available
- **WHEN** the requested candidate is failed, cancelled, missing, expired, ambiguous, or inconsistent with the requested tag and peeled commit
- **THEN** publication fails before requesting publishing credentials
- **AND** no pre-tag branch artifact or later replacement candidate is substituted

### Requirement: Shared Python and Studio versions
The shared stable `vX.Y.Z` tag SHALL determine all three packages' release version `X.Y.Z`. CI SHALL prepare owned project and lockfile version metadata before release builds, reject drift, and leave the source commit unchanged. Untagged development SHALL remain usable with valid checked-in metadata and no release-only environment variable. Preparation SHALL NOT rewrite compatibility policy or historical version references.

#### Scenario: Package versions diverge
- **WHEN** either prepared Python project, Studio package, owned lockfile version, or built release artifact differs from the tag-derived version
- **THEN** CI and publication fail before any registry write

#### Scenario: Tag supplies the version without a source bump
- **WHEN** a maintainer tags an eligible commit as `v0.2.2` while its checked-in package metadata still declares `0.2.1`
- **THEN** CI prepares both Python distributions and Studio as `0.2.2`
- **AND** the Python workspace lock entries and both Studio lockfile root versions agree with `0.2.2`
- **AND** package filenames and installed Python runtime version reporting reflect `0.2.2`
- **AND** CI does not push a generated commit or modify the tag

#### Scenario: Stamping is repeated for the same tag
- **WHEN** version preparation runs again on a checkout already prepared for that tag
- **THEN** owned versions remain unchanged and validation succeeds without creating another tag or commit

#### Scenario: Normal development has no release tag
- **WHEN** a contributor installs the checkout or normal branch CI runs without a release tag
- **THEN** existing valid checked-in project and lockfile metadata remains usable
- **AND** no CI-only version variable, placeholder substitution, or Git-derived development-version service is required
- **AND** branch CI does not publish packages

#### Scenario: Other version numbers appear in the checkout
- **WHEN** release preparation encounters third-party dependency pins, cross-package compatibility ranges, Studio/backend interface checks, historical changelog entries, archived docs, or versioned test examples
- **THEN** it leaves those values unchanged instead of globally replacing matching text

#### Scenario: Tag version is outside the declared compatibility line
- **WHEN** a maintainer tags a version that the declared Python dependency constraints or Studio/backend checks do not support
- **THEN** installed-package acceptance fails and blocks publication
- **AND** preparation does not widen ranges, change the API interface version, or declare compatibility from the tag alone
