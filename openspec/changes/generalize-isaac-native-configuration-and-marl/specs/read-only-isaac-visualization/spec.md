# Spec Delta

## Purpose

Provide mandatory native Viser spectator viewing during training and replay without exposing simulation mutation or public endpoints.

## ADDED Requirements

### Requirement: V1 Private read-only construction is an early feasibility gate

Before substantial implementation, the pinned native viewer SHALL be qualified for private read-only construction using supported APIs or a narrow isolated adapter reusing native scene rendering. Vendor-file edits, process-global monkeypatches and a second renderer SHALL NOT be used. Unqualified construction SHALL fail closed; Viser SHALL remain a required incomplete deliverable, not be deferred or removed.

#### Scenario: Native construction relies on global server reassignment
- **WHEN** no supported hook or qualified instance-local adapter can avoid that path and bind privately before listening
- **THEN** the feasibility gate records the precise blocker and dependent work stops without exposing a viewer or claiming this pass complete

### Requirement: V2 Spectator requests cannot mutate simulation

Supported viewer callbacks and HTTP/WebSocket requests SHALL have no effective path to pause/resume/reset simulation, change task/configuration, inject actions/forces, alter actor/control state or modify timing/budgets. Denial SHALL exist at the actual mutation handoff as well as callback registration. Hiding or disabling controls alone SHALL NOT satisfy read-only behavior.

#### Scenario: Direct or forged mutation is attempted
- **WHEN** tests invoke callbacks directly or send unknown/stale/forged mutation messages through supported dispatchers
- **THEN** physics/task/actor/control/timing/budget state remains unchanged and no native mutator executes
- **AND** inability to guarantee denial prevents viewer-requested startup

### Requirement: V3 Viewing is isolated from policy and learning state

The spectator SHALL support camera movement, selected environments, markers, supported plots and display-only pause without changing physics, policy sensors/history, RNG, environment count or frame budgets. Training activation SHALL be explicit; evaluation/replay SHALL require separate activation. Browser absence, disconnect and reconnect SHALL NOT stop, reset or pause learning.

#### Scenario: A browser moves its camera and reconnects
- **WHEN** the user navigates, changes displayed environments, disconnects and reconnects during training
- **THEN** native physical/control trajectories at equivalent logical steps and learning counters match the qualified nonmutating path, while the spectator changes only display state

#### Scenario: A policy trained with viewing is replayed
- **WHEN** evaluation starts without a separate viewer opt-in
- **THEN** no listener opens and saved training activation does not change replay distribution or the actor contract

### Requirement: V4 Private binding and owned cleanup are observable

Viser SHALL bind to localhost by default with sharing disabled and support private SSH/VS Code forwarding. The actual listener address and requested port SHALL be verified. Startup errors, port conflicts/reassignment or lost safety qualification SHALL fail the viewer-requested job and clean partial resources. Completion/failure/cancellation SHALL stop owned listeners/resources; headless jobs SHALL remain independent.

#### Scenario: The requested port is occupied
- **WHEN** native server construction fails or silently chooses another port
- **THEN** initialization fails explicitly and closes any partial listener instead of exposing a different endpoint or silently claiming viewing succeeded

#### Scenario: A connected job ends
- **WHEN** training or replay completes, fails or is cancelled
- **THEN** owned viewer threads/listeners and native resources are released with correct job status, without public exposure or lingering listeners

### Requirement: V5 Browser acceptance is distinct from headless and sensor proof

Viser completion SHALL require real browser sessions during native training and separate replay, camera interaction, mutation attempts, disconnect/reconnect and lifecycle cleanup. Mocked components, headless Isaac or policy sensor buffers SHALL NOT establish viewing acceptance. Missing graphics/Vulkan prerequisites SHALL remain explicit blockers, not successful tests.

#### Scenario: Only API or headless checks pass
- **WHEN** real native browser and request-denial checks have not passed
- **THEN** Viser native/browser acceptance and Milestone C remain open regardless of implementation or unit-test status
