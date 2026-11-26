## ADDED Requirements

### Requirement: Local learning consumes one common versioned actor

The existing local-fit SDK SHALL accept an approved shared actor and trusted bounded recipe for compatible simulated or physical experience through a versioned contract. It SHALL validate the global version, exact initial tensor digest and shared manifest separately from each client's recipe and provenance. Matching shapes alone SHALL NOT authorize learning or aggregation. Existing NEX-234 dataset-only fits SHALL remain supported.

#### Scenario: Different providers receive one global actor
- **GIVEN** compatible Isaac and MuJoCo clients have distinct trusted local recipes and a shared manifest specifying architecture, named tensor roles, shapes and dtypes, trainable scope, semantic policy contract and processor/buffer ownership
- **WHEN** each receives the same approved base policy
- **THEN** each fit begins from the exact supplied tensors and records its own recipe, simulator, task and data provenance without rejecting solely because provider identities differ.

#### Scenario: A local recipe is incompatible
- **WHEN** a supplied actor has a wrong base digest, tensor schema, control semantics, unapproved mapping or untrusted executable content
- **THEN** the fit is rejected before native interaction or optimization.

### Requirement: Bounded local updates are numeric, attributable and private by default

The SDK SHALL return a validated full numeric update or explicitly supported versioned delta. Its receipt SHALL identify the base, manifest, trainable scope, local provenance, objective, actual eligible frame/window/episode and optimizer-step counts, and approved finite metrics. It SHALL reject unapproved changes to frozen state. Private data and training state SHALL remain local unless a future contract explicitly authorizes disclosure.

#### Scenario: A client returns an update receipt
- **WHEN** the client returns a numeric update under the current contract
- **THEN** raw observations, trajectories, optimizer/critic/replay state, native handles and sensitive local identifiers remain local.

#### Scenario: A client initializes local training state
- **WHEN** the client initializes a fit from an approved actor
- **THEN** it resets the optimizer and identifies who owns the critic state. On-policy RL uses fresh experience from the compatible policy version; offline data does not qualify as on-policy experience.

#### Scenario: A client performs a second externally initialized fit
- **WHEN** a compatible new base is supplied under an approved bounded recipe
- **THEN** the exact new base is used and local optimizer/critic behavior, actual counts and changed shared tensors are attributable without leaking private data or silently reusing old trajectories.

### Requirement: Cross-provider handoff is demonstrated before production federation

NexuML SHALL demonstrate compatible Isaac and MuJoCo fits from one actor, a deterministic test-only combination, candidate reload and evaluation, and a second bounded round. The three-domain proof SHALL additionally require approved real ROS 2 experience with a validated matching physical contract and SHALL run independently of distributed GPU qualification. Mixed RL/imitation objectives or shared-subset configurations SHALL require separate evidence.

#### Scenario: Compatible local results are combined in a reference fixture
- **WHEN** valid numeric updates from the same base are returned by qualified providers
- **THEN** a deterministic test fixture combines only declared shared parameters, reloads the candidate, evaluates it and starts a subsequent fit without claiming a deployed NexuFL round or guaranteed improvement.

#### Scenario: Genuine compatible episodes are available from all three domains
- **WHEN** Isaac, MuJoCo and approved physical ROS 2 episodes are used for the initial proof
- **THEN** all three fits use the same behavior-cloning objective. Results compare equal-client and equal-domain weights and report per-domain learning and worst-domain regressions against local-only and simulator-only baselines.

#### Scenario: A candidate is considered for physical online learning
- **WHEN** the reference fixture produces a candidate
- **THEN** aggregation does not deploy it. Physical online learning requires authorized safe control and supervised, bounded operation.

#### Scenario: Real ROS 2 data or hardware is unavailable
- **WHEN** only simulated or fake-transport evidence exists
- **THEN** the simulator-only handoff can be qualified independently and the three-domain real-experience gate remains blocked or unverified.

### Requirement: NexuFL owns federation rather than the local learning SDK

NexuML SHALL expose a documented, pinned numeric model/update contract and local fit/evaluation interface consumable by NexuFL without importing NexuFL or requiring a server. Federation scheduling, aggregation policy, global versions, cross-client stale/duplicate handling, transport, privacy policy and promotion/rollback SHALL remain NexuFL responsibilities. No production federation implementation SHALL be added to NexuML by this change.

#### Scenario: A trusted NexuFL client program calls the SDK
- **WHEN** it supplies the approved base state, manifest and locally installed recipe
- **THEN** NexuML returns a bounded validated update receipt through its ordinary local-learning interface, independently of the federation transport.

### Requirement: Mixed local objectives are qualified explicitly and domain weighting is observable

After the common-objective proof, NexuML SHALL compare bounded simulation RL and authorized real-world offline updates from the exact same actor base. The reference experiment SHALL validate shared parameter roles and physical semantics and keep training state private. Numeric compatibility SHALL NOT imply general mixed-objective compatibility or automatic promotion. Production aggregation remains NexuFL-owned.

#### Scenario: Simulators learn with PPO while a robot learns from demonstrations
- **GIVEN** Qualified local RL and offline learning modes with the same approved shared actor manifest and starting state
- **WHEN** Their bounded numeric updates are combined in the reference experiment
- **THEN** held-out results compare equal-client and equal-domain weighting against common-objective, local-only and simulation-only baselines using predeclared criteria.
- **AND** the report identifies each objective, budget, count and weighting choice, update conflicts and incompatible shared roles. Negative or inconclusive outcomes remain visible.

#### Scenario: Real-world experience is scarce relative to simulation
- **WHEN** A large simulated frame count and a smaller approved real-world episode set contribute updates
- **THEN** The experiment reports distinct count semantics and an equal-domain comparison rather than implicitly equating raw transitions with eligible demonstrations.
