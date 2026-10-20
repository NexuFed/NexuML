## MODIFIED Requirements

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
