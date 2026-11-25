## ADDED Requirements

### Requirement: Teleoperation reuses qualified native input

A thin public CLI/SDK and optional recorder SHALL reuse supported native keyboard/gamepad/SpaceMouse/leader/XR devices and shared interaction/viewer lifecycle. Remote operator input SHALL use qualified existing transport independently of viewer, with freshness/latency and safe disconnect. Unsupported input/modes reject.

#### Scenario: An operator teleoperates simulated LeKiwi
- **WHEN** approved native input is selected
- **THEN** contracted commands control the simulated robot and disconnect safely stops or hands off control.

### Requirement: Action ownership and demonstrations preserve execution truth

One authorized action owner per controlled resource SHALL be explicit, with qualified human takeover, native arbitration and controller safety. Native HDF5/MCAP/LeRobot recording/conversion SHALL preserve requested/executed actions, timing, episodes, interventions, sensors/calibration, outcome and source grouping. Human and scripted/generated data SHALL be labeled separately.

#### Scenario: A user takes over a rollout
- **WHEN** the approved session records the intervention
- **THEN** the recording retains distinct proposed/executed actions and interval, and supports validated offline training without leakage.
