## ADDED Requirements

### Requirement: G1 variants and native workflows remain explicit

Primary reference SHALL use exact 29-DoF G1 body and verified hand/actuator/control mapping, official native sources, separate locomotion and manipulation learning. Optional 23-DoF SHALL use separate contract. Matching Isaac/MuJoCo SHALL verify actor/control compatibility, outcomes and isolated DDS system test. Native XR or maintained human teleop SHALL produce validated episodes for IL and playback.

#### Scenario: A G1 actor is transferred between simulators
- **WHEN** matching body/hands and action/observation semantics have been validated
- **THEN** the compatible actor executes both native paths with separate provenance; incompatible variants reject.

### Requirement: Later humanoid deployment is supervised G1

The first humanoid physical proof SHALL follow the SO-101 and LeKiwi physical learning milestones and use the exact approved G1 EDU with its qualified locomotion, balance and whole-body controller. Software readiness may proceed independently. Unattended exploration, implicit actuation, automatic promotion and mid-episode policy swaps SHALL NOT be allowed. Only approved compatible experience SHALL feed the foundation local-fit handoff. Hardware evidence remains independent.

#### Scenario: A supervised G1 session is requested
- **WHEN** an operator requests a physical session
- **THEN** it can proceed only with recorded-input, sim-to-sim and full-stack parity evidence, a protective workspace and support, an independent emergency stop and fallback, and explicit authorization for the session.
- **AND** the session is limited to bounded, supervised manipulation and local capture.

#### Scenario: G1 hardware is not available
- **WHEN** simulation and imitation evidence is qualified
- **THEN** software gates may pass but physical and actual real-experience gates remain blocked/unverified.
