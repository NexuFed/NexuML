# studio-local-distribution Specification

## Purpose

Make Studio installable through npm and runnable locally against a user-installed NexuML environment, including isolated uv tools, without provisioning a second Python runtime or requiring a repository checkout.

## Requirements

### Requirement: npm-distributed local application

Studio SHALL provide an npm-installable package and launcher containing production frontend assets. Normal use SHALL NOT require cloning the repository, building frontend source, starting a development server, creating a hosted account, or having internet access after the required packages/assets are installed.

#### Scenario: User installs the published frontend package
- **WHEN** the user installs Studio through npm and invokes its launcher
- **THEN** the production frontend starts locally without a source checkout or user-run build step

#### Scenario: Installed application is opened offline
- **WHEN** the installed application is started without internet access and its selected runtime is available
- **THEN** its shell, fonts, styles, graph editor, and local API communication work without external asset requests

### Requirement: User-owned NexuML installation

Studio SHALL use an existing user-selected NexuML installation. It SHALL NOT silently install Python, NexuML, API dependencies, optional libraries, or GPU dependencies, mutate Python environment files, or upgrade installed packages.

#### Scenario: NexuML cannot be found
- **WHEN** Studio cannot locate the selected executable or selected Python environment
- **THEN** it fails with actionable selection/installation guidance without creating an environment or falling back to an unrelated installation

#### Scenario: API dependencies are missing
- **WHEN** the selected NexuML installation lacks the API extra
- **THEN** Studio explains the required user-run installation step and does not run it automatically

### Requirement: uv tool installation support

Studio SHALL support NexuML installed through `uv tool install`, including the API extra when the server is needed. Server startup SHALL use the tool entrypoint's environment instead of assuming that the current shell's `python` can import the installed tool. Required custom-library dependencies SHALL come from that selected environment or existing configured local library roots.

#### Scenario: Tool-installed NexuML starts the API
- **WHEN** the user installs `nexuml[api]` with `uv tool install` and selects its executable
- **THEN** Studio launches the API through that installation and reports its isolated Python executable

#### Scenario: Base library is included in a tool installation
- **WHEN** the user installs `nexuml[api,library]` as a uv tool
- **THEN** the API discovers the installed base library using NexuML's existing discovery behavior

### Requirement: Project environment support

Studio SHALL also support NexuML in an existing project environment, whether installed with `uv add` or `uv pip install`. Installation selection SHALL be explicit when more than one environment is available, and the selected installation SHALL be visible in Studio. Starting Studio SHALL NOT trigger implicit uv environment synchronization or package installation.

#### Scenario: Project environment is selected
- **WHEN** the user selects an existing project Python environment containing NexuML and its API dependencies
- **THEN** the API starts with that environment's interpreter and libraries, not a globally installed tool

#### Scenario: Project and tool installations coexist
- **WHEN** both a project installation and a uv-tool installation are available
- **THEN** Studio honors the explicit selection and does not merge their registries or choose a different interpreter silently

### Requirement: Safe launch and lifecycle

The launcher SHALL start local services on loopback, establish protected runtime connection information, report readiness/startup failures, and distinguish owned services from an explicitly attached existing API. Secrets SHALL NOT be embedded in package assets or browser URLs. Browser closure SHALL NOT stop training; launcher/server shutdown SHALL report and handle active owned work without claiming unsupported remote cancellation.

#### Scenario: Owned API startup fails
- **WHEN** API initialization fails before readiness
- **THEN** the launcher reports the selected runtime's error, cleans up processes it owns, and does not open a misleading connected UI

#### Scenario: Studio attaches to an already running local API
- **WHEN** the user explicitly selects an existing local API and supplies its connection credentials
- **THEN** Studio verifies interface/runtime compatibility and does not terminate that API when the frontend launcher exits

#### Scenario: Backend package releases independently
- **WHEN** Studio connects to the selected backend
- **THEN** it accepts stable NexuML >=0.2.1,<0.3.0 with API interface 3 and the native-contract catalog, reports NexuML/Python version and interpreter, and rejects unsupported versions before exposing the workbench/catalog with actionable required-version guidance

#### Scenario: Authenticated hosted access is explicitly configured
- **WHEN** an operator supplies a public HTTPS origin
- **THEN** the Studio listener still defaults to 127.0.0.1 and requires a separate private proxy credential injected by an externally authenticated reverse proxy; non-loopback listening requires an explicit host and security warning
- **AND** direct bootstrap/API access without that credential is rejected even with spoofed public/local Host/Origin headers, and proxy credentials are stripped before forwarding to Python

#### Scenario: Installed frontend and backend use different catalog interfaces
- **WHEN** either the launcher or browser encounters an earlier, missing or otherwise unsupported interface identity
- **THEN** it rejects the mismatch before fetching or exposing editable catalog-dependent views
- **AND** it does not adapt legacy schema-extension fields, guess absent native contracts or automatically install or upgrade either package

#### Scenario: Matching packages are installed independently
- **WHEN** the built frontend and backend with the native-contract interface are installed in project and isolated-tool environments
- **THEN** each selected installation supplies working forms, graph authoring, CPU training, reconnection and artifact inspection without a source checkout or startup dependency mutation
- **AND** hosted HTTP/WebSocket authentication and owned-versus-attached shutdown behavior remain unchanged

### Requirement: Released package portability

Release validation SHALL test the packaged application, rather than only source-checkout development, on supported Linux, macOS, and Windows environments. Python core commands SHALL remain usable without the API extra, and Python artifacts SHALL NOT inadvertently include frontend build/dependency directories.

#### Scenario: Maintainer temporarily restricts platform coverage
- **WHEN** the 2026-10-20 maintainer override disables macOS/Windows CI
- **THEN** Linux project/tool checks remain mandatory on ika-runner and cross-platform acceptance stays unverified until the other targets are restored and pass

#### Scenario: Packaged application is tested on a supported operating system
- **WHEN** release validation installs the npm package into a clean location
- **THEN** the launcher, production assets, runtime selection, REST/WebSocket connection, and shutdown work using that operating system's path/process conventions

#### Scenario: Core-only NexuML is installed
- **WHEN** NexuML is installed without the API extra
- **THEN** existing nonserver CLI operations remain available and do not require frontend or FastAPI imports

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
