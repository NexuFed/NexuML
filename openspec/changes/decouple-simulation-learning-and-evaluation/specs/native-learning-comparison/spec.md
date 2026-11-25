## ADDED Requirements

### Requirement: Native learning comparison is an early architectural decision gate

NexuML SHALL compare native Isaac Lab training and NexuML on a matched reward-bearing SO-101 reference before broad architectural expansion. The comparison SHALL freeze matching task, control, hardware, budget, seeds and evaluation protocol, disclose unavoidable algorithm differences, and measure correctness, learning outcomes, wall time, throughput, peak memory, implementation/dependency burden and external-weight local-fit suitability. It SHALL use an upstream-supported learner and a minimal native TorchRL reference where supported to isolate overhead, predeclare acceptance criteria and record the retain/simplify/narrow decision. Failure or missing evidence SHALL NOT count as approval.

#### Scenario: Native and NexuML policies are trained on the same task
- **WHEN** the predeclared matched comparison is executed
- **THEN** revision-bound multi-seed learning and repeated performance results, deviations, maintainability evidence and a decision are published before broad distributed or provider expansion.

### Requirement: Native-first policy adoption is explicit and qualified

NexuML SHALL qualify one named native Isaac Lab actor import or initialization path with trusted local architecture/parameter mapping, processor/normalization and physical action semantics. It SHALL distinguish inference-only import from verified trainable compatibility. Unsupported architectures, executable artifacts, unvalidated mappings and optimizer continuation SHALL be rejected rather than implicitly converted.

#### Scenario: A supported native policy is supplied
- **WHEN** the operator supplies the approved local checkpoint and explicit compatible mapping
- **THEN** known-input inference parity, held-out native evaluation and bounded external-weight fine-tuning are demonstrated without requiring initial training in NexuML.
