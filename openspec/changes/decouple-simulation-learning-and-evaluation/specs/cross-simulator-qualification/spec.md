## ADDED Requirements

### Requirement: A second GPU simulator demonstrates portability

A qualified native MuJoCo GPU/Warp or maintained Playground reference SHALL use the existing interaction/collector/learner boundary, preserve native batch/device/reset semantics, and retain existing CPU Gymnasium references. The selected implementation and tensor interoperability SHALL be proven against pinned dependencies before support is claimed.

#### Scenario: A matched Isaac and MuJoCo task is configured
- **GIVEN** The observation and control semantics are compatible or explicitly mapped
- **WHEN** the corresponding supported workflow is executed
- **THEN** The same policy architecture and learning algorithm run through the same public interface, with distinct native provenance and measured outcomes rather than asserted physics equivalence.

### Requirement: System-level providers retain native time and safety semantics

Gazebo and CARLA integrations SHALL use native stepping, clock/tick ownership, synchronized observations, reset/cleanup and isolation. ROS 2 physical interaction SHALL use explicit bounded commands, timing, watchdog and safe-stop semantics, without assuming arbitrary reset or accelerated time. A provider SHALL NOT be advertised as qualified without actual supported reference evidence.

#### Scenario: A ROS 2 or client-server simulator is selected
- **GIVEN** The provider requires external transport and clock ownership
- **WHEN** the corresponding supported workflow is executed
- **THEN** The declared runtime validates prerequisites, owns only its authorized resources, and preserves native timing and cleanup; unsupported modes fail explicitly.

### Requirement: Optional runtime profiles and evidence are reproducible

Qualified provider and learner profiles SHALL use the existing NexuML optional-dependency and image/CI mechanisms. Simulator-free inference and remote learner installations SHALL remain possible. Runtime versions, driver/ABI, assets, assigned resources, actual native GPU execution, learning, throughput, network bytes and cleanup SHALL be recorded for claimed topologies; missing hardware SHALL be reported as blocked/unverified rather than pass.

#### Scenario: A new heterogeneous Ray deployment is qualified
- **GIVEN** The learner and simulation workers use distinct compatible pinned images
- **WHEN** the corresponding supported workflow is executed
- **THEN** Real public workflow evidence identifies exact placement, dependency versions, native execution and learning outcomes; a missing GPU or incompatible image prevents a support claim.


### Requirement: Software delivery and live evidence have independent gates

Implementation SHALL use the existing optional CI/image mechanisms and immutable role-specific profiles, including native headless and separately qualified graphics/Kit-less profiles where supported. The work queue SHALL proceed through independently feasible source and hermetic tests despite unavailable hardware, authorized recordings, approved worker artifacts or external display/client. A blocked ledger SHALL identify exact missing prerequisites, work done and repro command. Tests or local evidence SHALL NOT be credited as unexecuted native/multi-node/hardware proof, and a software-ready handoff SHALL NOT assert full acceptance. No automatic infrastructure mutation, downloads or physical actuation beyond separate authorization.

#### Scenario: Hardware or a private WebRTC client is unavailable
- **WHEN** the acceptance prerequisite is missing
- **THEN** the specific live gate remains blocked and other implementation, tests, documentation and qualified profile delivery continue without repeated user intervention.
