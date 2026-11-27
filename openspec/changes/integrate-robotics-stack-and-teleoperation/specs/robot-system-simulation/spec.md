## ADDED Requirements

### Requirement: Learning and system simulation remain distinct

Native vectorized learning SHALL remain independent of per-step ROS/DDS. Deployment-equivalent simulation SHALL reuse actual approved robot software, native launch and compatible controller/hardware interfaces against simulated hardware, validating clock, frames, lifecycle and cleanup.

#### Scenario: A policy is exercised against simulated hardware
- **WHEN** a qualified system launch selects the simulation driver
- **THEN** actual policy commands traverse the compatible controller and native observations return without addressing physical endpoints.

### Requirement: Simulation and physical transports are isolated

ROS and native Unitree DDS SHALL use explicitly isolated simulation domains/interfaces/namespaces and reject ambiguous or physical destinations. Native failure, stale data, reset, cancellation and repeated startup/shutdown SHALL preserve ownership. Independent Gazebo acceptance remains required.

#### Scenario: A simulated G1 job runs beside a robot network
- **WHEN** its isolated simulation transport is started or fails
- **THEN** no physical command is emitted and owned resources close without modifying borrowed infrastructure.
