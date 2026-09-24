## Purpose

Defines the artifact boundary for RL-trained NexuML policies so deployment and future NexuFL federation do not accidentally depend on local environment experience or training-only networks.

## ADDED Requirements

### Requirement: Primary RL export is the deployable policy pipeline

The normal NexuML model export for an RL scenario SHALL use `scenario.pipeline` / its compiled actor policy as the primary portable model artifact.

#### Scenario: RL policy is exported

- **WHEN** an RL run exports its trained model
- **THEN** the exported primary pipeline contains the trained deployable policy parameters
- **AND** can be loaded for policy inference without constructing the training environment.

### Requirement: Training-only RL state is not part of the normal policy artifact

Replay buffers, collected trajectories, live environments, ROS handles, and algorithm-owned critic/Q/target networks SHALL NOT be silently embedded into the ordinary portable policy artifact.

#### Scenario: PPO run is exported

- **WHEN** PPO has a value network and collected rollout state
- **THEN** the ordinary policy package excludes the rollout/replay state and value-network runtime
- **AND** exports only state required by the deployable policy contract unless a future explicit artifact kind requests more.

### Requirement: Export records RL provenance

RL policy export metadata SHALL identify that the policy was trained with reinforcement learning and SHALL record stable algorithm/environment identities sufficient to diagnose how the policy was produced.

#### Scenario: User inspects RL export metadata

- **WHEN** an exported RL policy's metadata is inspected
- **THEN** it reports reinforcement-learning training mode
- **AND** stable algorithm kind/name/version
- **AND** stable environment kind/name/version
- **AND** relevant resolved-config/provenance hashes.

### Requirement: Resume checkpoint and deployable policy are distinct artifacts

A Lightning checkpoint for an RL run MAY include optimizer and training-only algorithm state needed for local resume, but SHALL NOT redefine the portable policy package contract.

#### Scenario: User resumes local PPO training

- **WHEN** a compatible RL Lightning checkpoint is supplied
- **THEN** NexuML may restore actor, critic, optimizer, and algorithm state required for continuation
- **AND** the separately exported policy remains loadable without that checkpoint.

### Requirement: Artifact boundary permits later federated RL

The RL export contract SHALL keep the actor/policy state separable from environment experience so a future NexuFL client procedure can exchange policy updates without requiring raw trajectories.

#### Scenario: Future NexuFL client consumes the policy

- **WHEN** a downstream federated-RL client loads the NexuML policy artifact
- **THEN** the actor/policy parameters can be identified independently of replay/trajectory/environment state
- **AND** no normal export contract requires raw local experience to leave the client.
