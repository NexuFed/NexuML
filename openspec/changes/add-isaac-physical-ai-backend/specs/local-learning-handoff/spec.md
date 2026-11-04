# Local Learning Handoff — New Capability

## ADDED Requirements

### Requirement: Bounded local imitation learning is an ordinary Python SDK operation
NexuML SHALL expose a documented Python API around its existing compiled model/session/training lifecycle for an installed trusted caller to fit a compatible, explicitly supplied initial numeric model state on authorized locally owned episode data under a deterministic bounded budget. It SHALL return validated named updated tensors, approved finite metrics, eligible training-window and episode counts and explicit optimizer reset/continuation semantics. No duplicate trainer, `FederatedTrainer`, HTTP server or dependency on NexuFL SHALL be required.

#### Scenario: Different externally supplied model states
- **WHEN** the trusted caller invokes two bounded local fits sequentially with different compatible initial weights
- **THEN** the exact requested starting state is used on each real optimizer run and both results are reproducible and attributable
- **AND** no previous optimizer or model state leaks into the next invocation unless explicitly versioned continuation has been separately supported and proven.

#### Scenario: No REST installation
- **WHEN** the approved client imports and uses the Python API without FastAPI/Uvicorn
- **THEN** training, validation and portable export SHALL continue to work.

### Requirement: A real SO-101 imitation policy is evaluated closed-loop
The first reference SHALL collect validated actual SO-101 simulated reaching demonstrations, train a genuine state-observation behavior-cloning actor in NexuML, clean-export/reload its complete policy, and measure success on fresh seeded simulator episodes; teacher-only privileged inputs SHALL NOT enter deployed student observations.

#### Scenario: Expert succeeded but BC did not
- **WHEN** successful expert trajectories or lower offline loss exist without measured successful closed-loop trained-policy episodes
- **THEN** the early proof is incomplete and its result SHALL not be presented as learned-policy success.

### Requirement: Shareable model state is numeric and physically compatible
A local fit SHALL validate architecture, named tensor roles/shapes/dtypes/finiteness, base version and trainable scope; the exact SO-101 ordered joint/action units/limits/control period and observation/processor identity; and local dataset/split and counted-window provenance. Executable packages stay explicitly operator-approved locally; shareable model outputs SHALL be numeric-only, with raw episode and optimizer data local.

#### Scenario: Equal-shaped but physically incompatible weights
- **WHEN** the model identity matches tensor widths but action order/scale/period or observation processors differ
- **THEN** the caller SHALL reject it before fitting.

#### Scenario: Data/source disclosure attempt
- **WHEN** a task supplies a remote executable model, unapproved local data binding or asks to export raw private episodes
- **THEN** the SDK SHALL reject the attempt and emit no training results or private payload.

### Requirement: An early PR gate is independently reviewable
The proposal SHALL stop after proven native SO-101 demonstrations, real local training, clean export/held-out simulator evaluation and portable Python local-fit semantics for a minimal upstream PR, while keeping ACT, VLA, full Isaac programme, FL, gRPC, Kubernetes and real hardware as separate follow-ups.

#### Scenario: Minimal proof is verified
- **WHEN** the small reference and contract tests pass
- **THEN** the implementation may be packaged for a scoped PR and pinned by an external NexuFL `ClientProgram`
- **AND** unverified later Physical AI requirements remain open.
