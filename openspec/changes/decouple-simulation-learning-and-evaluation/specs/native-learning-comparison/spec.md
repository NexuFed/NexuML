## ADDED Requirements

### Requirement: Native learning comparison is an early architectural decision gate

Before broad architectural expansion, NexuML SHALL compare native Isaac Lab and NexuML training on a matched reward-bearing SO-101 task. It SHALL use an upstream-supported learner and, where supported, a minimal native TorchRL reference to isolate overhead. The comparison SHALL use predeclared acceptance criteria and record whether to retain, simplify or narrow the architecture. Failed or missing evidence SHALL NOT count as approval.

#### Scenario: Native and NexuML policies are trained on the same task
- **GIVEN** the task, control settings, hardware, budget, seeds and evaluation protocol are fixed across runs
- **WHEN** the native and NexuML learners run the matched comparison
- **THEN** the revision-bound report gives correctness, multi-seed learning outcomes, repeated wall-time, throughput and peak-memory measurements, implementation and dependency burden, and suitability for local fits from external weights.
- **AND** it discloses unavoidable algorithm differences and other deviations and publishes the decision before broad distributed or provider expansion.

### Requirement: Native-first policy adoption is explicit and qualified

NexuML SHALL qualify one named native Isaac Lab actor import or initialization path with trusted local architecture/parameter mapping, processor/normalization and physical action semantics. It SHALL distinguish inference-only import from verified trainable compatibility. Unsupported architectures, executable artifacts, unvalidated mappings and optimizer continuation SHALL be rejected rather than implicitly converted.

#### Scenario: A supported native policy is supplied
- **WHEN** the operator supplies the approved local checkpoint and explicit compatible mapping
- **THEN** known-input inference parity, held-out native evaluation and bounded external-weight fine-tuning are demonstrated without requiring initial training in NexuML.


### Requirement: Native GPU scaling is evaluated at representative workload sizes

NexuML SHALL retain TorchRL and compare co-located direct and qualified distributed GPU PPO with the native learner on matched hardware/task/precision/global frames and held-out outcomes. Report realistic native vector widths where viable, startup/collect/GAE/optimization/synchronization/transfer timing, actual device-wide memory and scaling efficiency. Profile and simplify measured overhead without weakening transition correctness; aim for 90 percent of matched native warm throughput, report the actual gap without guaranteeing speedup. The small state MLP and CPU learner tests SHALL NOT establish visual/VLA performance. Visual and configurable memory-pressure references SHALL measure GPU inference/learning placement; actual large model claims require a separately qualified model/objective and approved resources.

#### Scenario: A state-only benchmark passes
- **WHEN** a larger image or memory-pressure workload is requested
- **THEN** its placement, memory, latency and learning evidence are measured separately and the historical state-only result is not extrapolated to a VLA.
