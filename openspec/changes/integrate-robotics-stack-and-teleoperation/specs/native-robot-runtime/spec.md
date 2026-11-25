## ADDED Requirements

### Requirement: Native robot interfaces and inference are optional

NexuML SHALL reuse existing interaction/transport seams and native ROS 2 and Unitree SDK2/DDS with explicit units, frame, timestamps, freshness, rate and calibration. Complete policy inference SHALL not require learner or simulator. Native dependencies SHALL be optional and pinned; incompatible runtime boundaries require actual wire/ABI proof.

#### Scenario: A native G1 controller is tested
- **WHEN** an approved compatible policy and simulator endpoint are selected
- **THEN** the simulator receives validated commands, and native observations and latency/parity evidence are retained.

### Requirement: Physical command authority is independently gated

Physical execution SHALL require separate explicit operator authorization and independently validated native controller safety. Generic simulation play/teleop SHALL NOT implicitly actuate hardware. Real-time fallback and stop stay local.

#### Scenario: An unauthorized physical endpoint is selected
- **WHEN** the runtime validates the configuration
- **THEN** it rejects before sending any physical command.
