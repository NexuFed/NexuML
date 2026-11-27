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

NexuML SHALL demonstrate compatible fits from one actor across native Isaac, native MuJoCo and an offline ROS 2 lane using episodes recorded through the live Isaac Sim ROS 2 Bridge with a validated matching observation/control contract. The proof SHALL run independently of distributed GPU qualification and need no physical robot. Mixed RL/imitation objectives or shared-subset configurations SHALL require separate evidence.

#### Scenario: Compatible local results are combined in a reference fixture
- **WHEN** valid numeric updates from the same base are returned by qualified providers
- **THEN** a deterministic test fixture combines only declared shared parameters, reloads the candidate, evaluates it and starts a subsequent fit without claiming a deployed NexuFL round or guaranteed improvement.

#### Scenario: Compatible episodes are available from all three simulated lanes
- **WHEN** native Isaac, native MuJoCo and bridge-recorded Isaac Sim episodes are used for the common-objective proof
- **THEN** all three fits use the same behavior-cloning objective. Results compare equal-client and equal-lane weights and report per-lane and worst-lane held-out results against local-only and native-simulator-only baselines.
- **AND** The candidate is reloaded and evaluated, and a second bounded round starts from it. The bridge lane is labeled simulated, not a third physics engine.

#### Scenario: Only mocked ROS 2 transport is available
- **WHEN** the live bridge has not produced compatible episodes
- **THEN** existing native Isaac and MuJoCo evidence remains valid, but the ROS 2 lane is unverified. Neither lane authorizes physical actuation or deployment.

### Requirement: NexuFL owns federation rather than the local learning SDK

NexuML SHALL expose a documented, pinned numeric model/update contract and local fit/evaluation interface consumable by NexuFL without importing NexuFL or requiring a server. Federation scheduling, aggregation policy, global versions, cross-client stale/duplicate handling, transport, privacy policy and promotion/rollback SHALL remain NexuFL responsibilities. No production federation implementation SHALL be added to NexuML by this change.

#### Scenario: A trusted NexuFL client program calls the SDK
- **WHEN** it supplies the approved base state, manifest and locally installed recipe
- **THEN** NexuML returns a bounded validated update receipt through its ordinary local-learning interface, independently of the federation transport.

### Requirement: Mixed local objectives are qualified explicitly and lane weighting is observable

After the common-objective proof, NexuML SHALL compare bounded native simulation RL and offline updates from bridge-recorded simulated episodes, starting from the exact same actor base. The reference experiment SHALL validate shared parameter roles and physical semantics and keep training state private. Numeric compatibility SHALL NOT imply general mixed-objective compatibility or automatic promotion. Production aggregation remains NexuFL-owned.

#### Scenario: Native simulators use PPO while the ROS 2 lane learns offline
- **GIVEN** Qualified local RL and offline learning modes with the same approved shared actor manifest and starting state
- **WHEN** Their bounded numeric updates are combined in the reference experiment
- **THEN** held-out results compare equal-client and equal-lane weighting against common-objective, local-only and native-simulator-only baselines using predeclared criteria.
- **AND** the report identifies each objective, budget, count and weighting choice, update conflicts and incompatible shared roles. Negative or inconclusive outcomes remain visible.

#### Scenario: Offline demonstration counts differ from PPO frame counts
- **WHEN** fresh simulator rollouts and bridge-recorded demonstrations contribute updates
- **THEN** the experiment reports distinct count semantics and an equal-lane comparison rather than implicitly equating raw transitions with eligible demonstrations.
