## ADDED Requirements

### Requirement: Local learning consumes one common versioned actor

NexuML SHALL reuse and version the existing local-fit SDK to accept an approved shared actor state and bounded local learning recipe for compatible simulation or physical experience. The shared aggregation manifest SHALL specify architecture, named tensor roles/shapes/dtypes, trainable scope, semantic policy contract and processor/buffer ownership independently of client-specific trusted recipe, simulator, task and data provenance. The exact initial tensor digest and global version SHALL be checked. Matching tensor shape alone SHALL NOT authorize learning or aggregation. Existing NEX-234 dataset-only behavior SHALL remain supported.

#### Scenario: Different providers receive one global actor
- **GIVEN** compatible Isaac and MuJoCo clients have distinct trusted local recipes and a matching shared manifest
- **WHEN** each receives the same approved base policy
- **THEN** each fit begins from the exact supplied tensors and records its own recipe/runtime provenance without rejecting solely because provider identities differ.

#### Scenario: A local recipe is incompatible
- **WHEN** a supplied actor has a wrong base digest, tensor schema, control semantics, unapproved mapping or untrusted executable content
- **THEN** the fit is rejected before native interaction or optimization.

### Requirement: Bounded local updates are numeric, attributable and private by default

The SDK SHALL return validated full numeric updates or an explicitly supported versioned delta, base and manifest identity, trainable scope, local provenance, actual eligible frames/windows/episodes and optimizer-step counts, objective and approved finite metrics. Unapproved changes to frozen state SHALL be rejected. Raw observations, trajectories, optimizer/critic/replay state, native handles and sensitive local identifiers SHALL remain local unless separately authorized by an explicit future contract. Initial fits SHALL reset optimizer and define critic state ownership; on-policy RL SHALL use fresh compatible policy-version experience. Offline data SHALL NOT be misrepresented as on-policy experience.

#### Scenario: A client performs a second externally initialized fit
- **WHEN** a compatible new base is supplied under an approved bounded recipe
- **THEN** the exact new base is used and local optimizer/critic behavior, actual counts and changed shared tensors are attributable without leaking private data or silently reusing old trajectories.

### Requirement: Cross-provider handoff is demonstrated before production federation

NexuML SHALL demonstrate compatible Isaac and MuJoCo local fits from a common actor, an explicit deterministic test-only reference combination, candidate reload/evaluation and a second bounded round. The eventual three-domain gate SHALL additionally use approved actual ROS 2 experience with a validated matching physical contract; fake transport and simulated episodes SHALL NOT satisfy real-data acceptance. Use a common behavior-cloning objective from genuine compatible episodes for the initial three-domain proof, with explicit equal-client and equal-domain weighting references. The real-data gate SHALL be independently executable without waiting for distributed GPU qualification. Mixed RL/imitation objectives or shared-subset configurations SHALL require separate evidence. Per-domain learning and worst-domain regressions SHALL be reported against local-only and simulator-only baselines. Physical online learning SHALL require authorized safe control and supervised bounded operation; no automatic physical deployment follows aggregation.

#### Scenario: Compatible local results are combined in a reference fixture
- **WHEN** valid numeric updates from the same base are returned by qualified providers
- **THEN** a deterministic test fixture combines only declared shared parameters, reloads the candidate, evaluates it and starts a subsequent fit without claiming a deployed NexuFL round or guaranteed improvement.

#### Scenario: Real ROS 2 data or hardware is unavailable
- **WHEN** only simulated or fake-transport evidence exists
- **THEN** the simulator-only handoff can be qualified independently and the three-domain real-experience gate remains blocked or unverified.

### Requirement: NexuFL owns federation rather than the local learning SDK

NexuML SHALL expose a documented, pinned numeric model/update contract and local fit/evaluation interface consumable by NexuFL without importing NexuFL or requiring a server. Federation scheduling, aggregation policy, global versions, cross-client stale/duplicate handling, transport, privacy policy and promotion/rollback SHALL remain NexuFL responsibilities. No production federation implementation SHALL be added to NexuML by this change.

#### Scenario: A trusted NexuFL client program calls the SDK
- **WHEN** it supplies the approved base state, manifest and locally installed recipe
- **THEN** NexuML returns a bounded validated update receipt through its ordinary local-learning interface, independently of the federation transport.

### Requirement: Mixed local objectives are qualified explicitly and domain weighting is observable

After the common-objective proof, NexuML SHALL support a bounded reference experiment combining compatible local simulation RL and authorized real-world offline learning updates from the exact same shared actor base. The experiment SHALL validate common actor parameter roles and physical semantics, preserve private training state, report objective and budget/count differences and update conflicts, and compare equal-client/equal-domain weighting references against common-objective, local-only and simulation-only baselines. Simulator frame count SHALL NOT silently determine real-domain contribution. Negative or inconclusive outcomes SHALL remain visible; no automatic promotion or claim of general mixed-objective compatibility follows from numeric compatibility. Production aggregation remains NexuFL-owned.

#### Scenario: Simulators learn with PPO while a robot learns from demonstrations
- **GIVEN** Qualified local RL and offline learning modes with the same approved shared actor manifest and starting state
- **WHEN** Their bounded numeric updates are combined in the reference experiment
- **THEN** The result is evaluated on held-out domains against predeclared comparison criteria, with explicit objective and weighting provenance; incompatible shared roles or regressive outcomes are reported rather than silently promoted.

#### Scenario: Real-world experience is scarce relative to simulation
- **WHEN** A large simulated frame count and a smaller approved real-world episode set contribute updates
- **THEN** The experiment reports distinct count semantics and an equal-domain comparison rather than implicitly equating raw transitions with eligible demonstrations.
