## ADDED Requirements

### Requirement: G1 variants and native workflows remain explicit

Primary reference SHALL use exact 29-DoF G1 body and verified hand/actuator/control mapping, official native sources, separate locomotion and manipulation learning. Optional 23-DoF SHALL use separate contract. Matching Isaac/MuJoCo SHALL verify actor/control compatibility, outcomes and isolated DDS system test. Native XR or maintained human teleop SHALL produce validated episodes for IL and playback.

#### Scenario: A G1 actor is transferred between simulators
- **WHEN** matching body/hands and action/observation semantics have been validated
- **THEN** the compatible actor executes both native paths with separate provenance; incompatible variants reject.

### Requirement: Later humanoid deployment is supervised G1

The later humanoid physical proof SHALL target approved exact G1 EDU after the SO-101 and LeKiwi physical learning milestones (software readiness may proceed independently) with retained qualified locomotion/balance/whole-body controller, recorded-input and sim-to-sim/full-stack parity, protective workspace/support, independent emergency stop/fallback and explicit authorization. Bounded supervised manipulation and local capture only; no unattended exploration, implicit actuation, automatic promotion or mid-episode swap. Approved compatible experience SHALL feed foundation local-fit handoff. Hardware evidence remains independent.

#### Scenario: G1 hardware is not available
- **WHEN** simulation and imitation evidence is qualified
- **THEN** software gates may pass but physical and actual real-experience gates remain blocked/unverified.
