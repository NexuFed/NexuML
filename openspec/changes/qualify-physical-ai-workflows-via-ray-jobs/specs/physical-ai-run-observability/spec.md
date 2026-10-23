# Spec Delta

## Purpose

Give users actionable and truthful information about remote training, viewing, artifacts and recovery without requiring a new dashboard or confusing job submission with completed learning.

## ADDED Requirements

### Requirement: O1 Run summaries connect execution and outputs

Submission and worker summaries SHALL identify the scenario/candidate, job and attempt, selected backend/device, budget, tracking destinations, viewer access and durable output paths when known. Completion SHALL provide reproducible resume and fresh-evaluation commands for the actual produced artifacts. Summaries SHALL not expose credentials or promise paths that were not created.

#### Scenario: A remote policy export completes
- **WHEN** the worker reports its result
- **THEN** users can identify the complete-policy path, associated checkpoint and tracking runs
- **AND** the supplied replay command targets that policy on remote resources rather than a guessed local path

### Requirement: O2 Progress reports observable phases and counters

Progress SHALL distinguish submission, resource waiting, setup, native initialization, collection/update, evaluation, export and terminal status where observable. Unknown phases SHALL be labelled unknown rather than fabricated. Physical frames, optimizer steps, environment counts and episode counts SHALL retain distinct meanings; absence of log output SHALL NOT alone be called failure.

#### Scenario: Ray reports running while dependencies install
- **WHEN** no learning progress has yet been observed
- **THEN** the user sees setup or an explicitly unknown execution phase
- **AND** the interface does not claim completed updates or a fabricated learning ETA

### Requirement: O3 Access instructions are private and nonconflicting

Suggested tracking and viewer commands SHALL use private bindings by default, distinguish worker and client addresses and avoid port collisions among active components. Access SHALL use existing approved tunnels or authenticated routes. Conflicting ports SHALL fail visibly instead of silently replacing servers or switching endpoints.

#### Scenario: Viser and MLflow are selected together
- **WHEN** their connection instructions are displayed
- **THEN** distinct actual ports and the required private access route are shown
- **AND** the default instructions do not expose either service through a public wildcard listener

### Requirement: O4 Explicit tracking failures fail visibly

An explicitly configured tracking backend that cannot initialize or deliver required outputs SHALL cause a visible job failure naming the backend and affected stage. Previously delivered valid outputs SHALL remain identifiable without promises of transactional rollback. Intentionally disabled logging SHALL remain valid and SHALL NOT create tracking services or writers.

#### Scenario: MLflow cannot initialize
- **WHEN** a real run explicitly requests MLflow and its initialization fails
- **THEN** the job does not continue as an apparently tracked success with MLflow silently omitted
- **AND** cleanup preserves the original error and any already-created run identities

#### Scenario: One of two artifact destinations fails
- **WHEN** delivery succeeds to one backend and fails to the other
- **THEN** the result names the successful and failed destinations and remains unsuccessful overall
- **AND** valid prior uploads are not deleted or described as atomically rolled back

### Requirement: O5 Policy provenance identifies physical training progress

New policy exports and evaluation receipts SHALL record known source physical frames and source model/checkpoint identity when available. Unknown frame counts SHALL remain explicitly unknown, even if a backend requires numeric step zero. Optimizer steps SHALL NOT be relabelled as physical frames, and a zero presentation step SHALL NOT imply an untrained baseline.

#### Scenario: A newly trained policy is replayed
- **WHEN** its recorded training-frame count is available
- **THEN** the receipt and tracking presentation identify that count and actual policy/checkpoint source
- **AND** records remain consistent after checkpoint resume, export and fresh-process reload

#### Scenario: An older policy has no frame provenance
- **WHEN** evaluation logs at presentation step zero
- **THEN** source frames are labelled unavailable and the policy identity is retained
- **AND** the run is not automatically classified as a frame-zero baseline

### Requirement: O6 Outputs and recovery remain unambiguous

User-facing output SHALL distinguish resumable checkpoints, complete inference policies, recordings and tracking stores. New attempts SHALL use unique locations rather than overwrite existing evidence. Cancellation, failed finalization and lost viewer connections SHALL have distinct statuses; a closed viewer after a finished short run SHALL provide a separate replay route rather than imply a hung training job.

#### Scenario: A short training example ends before the browser opens
- **WHEN** the user inspects its completion summary
- **THEN** the summary states that the live job ended and provides an explicitly enabled remote replay command
- **AND** training is not silently delayed, reset or rerun merely to keep a viewer alive

### Requirement: O7 Metrics disclose their scope and semantics

Reports SHALL distinguish training reward, evaluation episode return, measured task success, timeouts, action rejections and per-agent/group statistics. A first-episode recording SHALL be labelled as such and not represented as aggregate or typical performance. Physical-frame counts SHALL not multiply by the number of agents, and capture limits SHALL not truncate metric episode budgets.

#### Scenario: A recorded episode succeeds while other episodes fail
- **WHEN** aggregate metrics and the clip are shown
- **THEN** the full evaluation denominator and failure categories remain visible
- **AND** the clip is not presented as proof that all or representative episodes succeeded

### Requirement: O8 Documentation is an executed user journey

Quickstarts SHALL provide the exact remote submission, status, artifact retrieval, resume and viewing/evaluation steps exercised for the candidate. Required extras, worker prerequisites, evaluation override namespace and persistent paths SHALL be explicit. Commands requiring unavailable integrations SHALL be labelled blocked or experimental, not included as already-qualified default paths.

#### Scenario: A user follows the accepted quickstart
- **WHEN** its commands are executed without editing application source
- **THEN** they select the documented remote runtime and produce the named artifact types
- **AND** unsupported nested overrides, missing worker files or absent prerequisites produce an actionable next step
