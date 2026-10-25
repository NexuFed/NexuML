# Spec Delta

## MODIFIED Requirements

### Requirement: Release tags match distribution versions
The release process SHALL derive the versions of `nexuml` and `nexuml-library` from a stable `vX.Y.Z` tag before building. The built distributions SHALL declare `X.Y.Z` regardless of the checkout's initial development versions. Release preparation SHALL NOT create a version-bump commit or accept an invalid version placeholder in a published distribution.

#### Scenario: Matching release tag is pushed
- **WHEN** tag `v0.2.2` is pushed for an eligible commit whose checked-in development metadata declares `0.2.1`
- **THEN** CI prepares both Python projects as version `0.2.2` before building
- **AND** package validation and publication require the built metadata to match `0.2.2`
- **AND** the tagged source commit remains unchanged

#### Scenario: Release tag and metadata disagree
- **WHEN** a prepared project or built Python distribution declares a different version from the release tag
- **THEN** the release workflow fails before uploading any artifact to a registry

#### Scenario: Tag does not identify a supported stable version
- **WHEN** a requested release ref is a branch, a deleted tag, or a tag such as `v01.2.3`, `v1.2`, `v1.2.3rc1`, or `studio-v1.2.3`
- **THEN** release preparation fails before a release build or registry publication

### Requirement: Production releases originate from integrated commits
The repository SHALL protect `main` with required release checks. Publication SHALL require an immutable stable tag whose peeled commit is contained in `main`, successful required branch CI for that exact commit, and successful tag CI covering the stamped source, supported-Python, package-artifact, and strict documentation checks.

#### Scenario: Tag points to a validated main commit
- **WHEN** a stable tag points to a commit contained in `main` with successful required branch checks and successful version-stamped tag CI for that same commit
- **THEN** publication proceeds only with that tag's verified candidate artifacts

#### Scenario: Tag points outside main
- **WHEN** a stable tag points to a feature or stacked-branch commit that is not contained in `main`
- **THEN** release eligibility fails before tag-build execution on institute runners, requesting publishing credentials, or uploading to a registry

#### Scenario: Exact release commit lacks current evidence
- **WHEN** required checks are missing, stale, unsuccessful, or belong to a different tag or commit
- **THEN** the release workflow does not publish artifacts

#### Scenario: Change is proposed for main
- **WHEN** a pull request would update `main`
- **THEN** branch protection requires the release checks to succeed before merge

### Requirement: Built artifacts are validated before publication
Tag CI SHALL build and validate wheels and source distributions for both stamped Python projects, install the artifacts in isolation, and retain the files immutably. Publication SHALL consume the verified candidate for the exact tag and commit, not a pre-tag branch build. Production SHALL reuse the candidate's original files without rebuilding or restamping them.

#### Scenario: Core-only artifact smoke test succeeds
- **WHEN** the core wheel is installed into an isolated environment without repository source paths or `nexuml-library`
- **THEN** importing `nexuml` and invoking CLI help succeed
- **AND** no base-library distribution is present
- **AND** installed metadata and `nexuml.__version__` report the tag-derived version

#### Scenario: Library artifact smoke test succeeds
- **WHEN** the built core and base-library wheels are installed together into an isolated environment
- **THEN** the base-library entry point is discoverable and representative components and scenarios can be listed
- **AND** both installed distribution versions match the tag-derived version

#### Scenario: Artifact validation fails
- **WHEN** metadata, contents, source-to-wheel rebuilding, either isolated installation, or any mandatory tag-CI check fails, is skipped, or is cancelled
- **THEN** no distribution is published to a registry
- **AND** no release is marked as successfully completed for that candidate

#### Scenario: Source distribution is rebuilt without the repository
- **WHEN** a wheel is built from either candidate source distribution without Git metadata or a release-version environment variable
- **THEN** the wheel retains that source distribution's tag-derived version

#### Scenario: Publisher starts before tag CI completes
- **WHEN** a tag push starts publication while the corresponding tag build is queued or running
- **THEN** publication waits for the exact candidate within a bounded interval without publishing credentials
- **AND** failure, cancellation, or timeout blocks publication instead of selecting branch artifacts

### Requirement: Test publication is available before production
Maintainers SHALL publish and verify the tag-specific validated Python files on TestPyPI before their production publication. TestPyPI SHALL use separate protected `testpypi` and `testpypi-library` environments and trusted publisher identities. Manual candidate publication SHALL require an existing stable tag and the same candidate validation as automatic publication.

#### Scenario: Stable candidate tag triggers test publication
- **WHEN** stable `vX.Y.Z` is pushed for an integrated commit and its stamped tag CI succeeds
- **THEN** the workflow publishes the selected candidate's core then library files to TestPyPI and verifies their installation
- **AND** production PyPI publication remains gated on publishing the GitHub Release, not the tag push
- **AND** ordinary main/dev branch pushes do not trigger package publication

#### Scenario: Maintainer runs the TestPyPI path
- **WHEN** an authorized maintainer dispatches candidate publication against an existing stable tag with an unpublished, verified candidate
- **THEN** that candidate's files are uploaded to TestPyPI
- **AND** installation uses TestPyPI for the NexuML packages and the production index for third-party dependencies

#### Scenario: Manual publication targets an unversioned branch
- **WHEN** an authorized maintainer dispatches publication against `main` or another branch
- **THEN** the workflow rejects the request without deriving a release version from checked-in development metadata

#### Scenario: Candidate publishing prerequisites are incomplete
- **WHEN** TestPyPI projects, trusted publisher identities, environment approval, the integrated tagged commit, verified candidate files, or required validation evidence are absent
- **THEN** the workflow does not upload the immutable candidate files

### Requirement: GitHub release follows successful package publication
Publishing the matching GitHub Release SHALL request production publication of the already tested candidate. The workflow SHALL attach or link that candidate's Python distributions only after both production uploads succeed. It SHALL NOT automatically create another release or claim completion after a partial publication.

#### Scenario: Both distributions publish successfully
- **WHEN** the matching GitHub Release has been published, candidate validation is complete, and PyPI accepts both distributions
- **THEN** the existing release receives the validated distribution artifacts
- **AND** completion identifies their tag-derived version and candidate identity

#### Scenario: One distribution upload fails
- **WHEN** PyPI does not accept every distribution artifact
- **THEN** the workflow reports partial publication clearly
- **AND** it does not mark the release as successfully completed or create another release that claims completion

## ADDED Requirements

### Requirement: Candidate identity is immutable across release operations
A release candidate SHALL bind its repository, stable tag, peeled commit, successful CI run and attempt, immutable artifact identifiers, filenames, and integrity values. Both registries' workflows SHALL use the same candidate identity. Missing, expired, conflicting, or ambiguous candidates SHALL block publication; a later run SHALL NOT silently replace a published candidate.

#### Scenario: Two tags identify the same source commit
- **WHEN** `v0.2.2` and `v0.2.3` point to the same eligible commit and each has a successful stamped build
- **THEN** each publication selects only the candidate carrying its requested tag and version
- **AND** sharing a commit does not make their artifacts interchangeable

#### Scenario: Candidate identity or bytes are tampered with
- **WHEN** a candidate's tag, peeled commit, CI origin, artifact identifier, package version, or file integrity does not match the requested release
- **THEN** validation fails before publishing credentials are requested

#### Scenario: Multiple successful candidate identities exist for one tag
- **WHEN** more than one distinct eligible candidate exists for the same tag
- **THEN** publication rejects the ambiguity rather than choosing the newest run

#### Scenario: Candidate files expire before production
- **WHEN** a candidate has been published for testing but its original immutable CI files are no longer available
- **THEN** production publication is blocked
- **AND** the workflow does not rebuild replacement bytes under the same version

#### Scenario: Release tag is moved after a candidate is built
- **WHEN** a tag resolves to a different commit from its candidate identity
- **THEN** publication is rejected and the workflow does not adopt a new candidate silently
