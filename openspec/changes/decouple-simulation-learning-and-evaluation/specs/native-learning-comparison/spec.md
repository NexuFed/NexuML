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

NexuML SHALL retain TorchRL and compare co-located GPU PPO with the native
learner under matched hardware, task, precision, frame budgets and held-out
evaluation. Profile before optimizing and preserve transition correctness.
Report the actual gap against the 90% warm-throughput target.

#### Scenario: A matched native throughput run completes
- **THEN** Report realistic vector widths, startup, collection, GAE, optimization, synchronization, transfers, device memory and scaling efficiency where qualified.

#### Scenario: A state-only benchmark passes
- **WHEN** a larger image or memory-pressure workload is requested
- **THEN** Report placement, memory, latency and learning on visual and configurable pressure references. Real large-model claims require their own approved model, objective and resource evidence.
