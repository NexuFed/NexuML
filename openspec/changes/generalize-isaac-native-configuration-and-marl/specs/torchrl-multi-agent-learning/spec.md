# Spec Delta

## Purpose

Integrate native TorchRL IPPO and MAPPO numerics with existing NexuML graph composition, Lightning updates and checkpoint ownership.

## ADDED Requirements

### Requirement: TL1 Both algorithms reuse native numerical components

IPPO and MAPPO SHALL use pinned native TorchRL losses and multi-agent GAE through the current registered algorithm/Lightning lifecycle. NexuML SHALL own graph construction, key/group routing and update scheduling, not copied clipping, entropy, critic-regression or GAE formulas. Ordinary single-agent PPO numerics SHALL remain unchanged; MARL SHALL NOT become an offline imitation prerequisite.

#### Scenario: A grouped algorithm performs an update
- **WHEN** IPPO or MAPPO runs on a qualified grouped rollout
- **THEN** native loss and estimator components produce targets/objectives, existing Lightning ownership performs real updates, and numerical comparisons match native references and small independent test oracles

### Requirement: TL2 Critics reflect declared local or centralized semantics

Actor/critic construction SHALL reuse composable NexuML graphs and suitable native multi-agent modules. IPPO critics SHALL remain local; MAPPO critics SHALL consume declared centralized inputs with correct per-agent outputs. Heterogeneous widths SHALL use supported cross-group composition rather than padding or assuming within-group centralization covers all roles.

#### Scenario: Heterogeneous centralized values are constructed
- **WHEN** differing observation widths feed a qualified native cross-group critic or equivalent existing compiled graph
- **THEN** outputs have each group's value shape, gradients reach intended encoders/trunk/heads, and no centralized feature becomes an actor input

### Requirement: TL3 Masks and normalization preserve valid targets

Initial MARL defaults SHALL be `normalize_advantage=False`, `average_gae=False`, `value_norm=None`. Group loss masks SHALL exclude inactive/padded samples without repurposing `shifted_valid`. Real terminal/truncation boundaries SHALL remain intact. Optional normalization SHALL require public-API proof of validity-aware statistics, update frequency and continuation for the exact layout or reject that combination.

#### Scenario: Inactive unused reward and target values change
- **WHEN** finite invalid suffix entries are perturbed without changing valid physical critic state
- **THEN** valid advantages, targets and gradients remain unchanged and true termination still prevents bootstrap while time limits bootstrap from final observations

#### Scenario: A generic mask is proposed as shifted validity
- **WHEN** the pinned counterexample with a true-terminal target is tested
- **THEN** the estimator-specific `shifted_valid` behavior is exposed and the integration uses genuine boundaries plus a separate group loss mask, not that substitution

#### Scenario: Unqualified normalization is requested
- **WHEN** its statistics would include invalid entries or its update/checkpoint semantics are unproven
- **THEN** configuration rejects that option rather than silently using biased statistics or a custom numerical workaround

### Requirement: TL4 Unique parameters have unique optimizer ownership

Shared actor/critic objects SHALL preserve parameter identity and unique optimizer state. Shared centralized parameters SHALL be stepped once after applicable group gradients accumulate. Inactive independent actors SHALL NOT move through stale gradients, momentum or weight decay; all-invalid losses SHALL NOT cause invalid reductions or optimizer steps. Valid peers may update shared actors.

#### Scenario: An independent actor becomes inactive after momentum warm-up
- **WHEN** later batches contain no valid samples for that actor but other groups remain active
- **THEN** its parameters and optimizer state remain unchanged, while active actors update and the shared centralized critic receives exactly one owned step

### Requirement: TL5 Checkpoints restore grouped numerical continuation

Current-schema checkpoints SHALL restore actor/critic weights, shared identities, unique optimizer slots, group ordering, frame counters and qualified normalization buffers. Controlled seeded continuation SHALL reproduce the corresponding uninterrupted update sequence. Critics, losses and optimizer state SHALL remain training-only and SHALL NOT become decentralized inference prerequisites.

#### Scenario: Shared grouped training resumes
- **WHEN** a checkpoint is restored under the same supported configuration and controlled rollout/RNG inputs
- **THEN** sharing/optimizer identities and subsequent targets/updates/counters match uninterrupted continuation without duplicated critic steps or extra simulator allocation for exhausted budgets

### Requirement: TL6 Native algorithm claims require real grouped learning

Acceptance SHALL include actual IPPO/MAPPO actor and critic updates on heterogeneous Pendulum MARL and homogeneous Handover sharing, with source/asset/backend compatibility qualified. It SHALL include partial-agent numerical tests, checkpoint/resume and decentralized export/replay. Missing prerequisites SHALL leave corresponding gates open; convergence or algorithm superiority SHALL NOT be required.

#### Scenario: Native grouped acceptance is attempted
- **WHEN** `Isaac-Pendulum-MARL-Direct` and `Isaac-Shadow-Handover-Direct` are qualified through NexuML
- **THEN** both algorithms demonstrate intended local/centralized critic dependencies, real finite actor/critic updates, independent/shared identities and actor-only reload/replay
- **AND** untested grouped image/vector or differing-action combinations are not advertised from shape-only tests
