## Purpose

Defines the artifact boundary for RL-trained NexuML policies so deployment and future NexuFL federation receive the complete inference policy while local experience, environments, and training-only learner state remain separate.

## ADDED Requirements

### Requirement: Primary RL export is a complete deployable policy

The normal NexuML model export for an interactive/RL scenario SHALL export a complete `CompiledPolicy` rather than treating the raw `CompiledPipeline` as sufficient when additional action adaptation is required.

The deployable policy contract SHALL contain or reconstruct:

- the trained `scenario.pipeline` parameters;
- the serialized `PolicySpec` / action-adapter definition;
- the portable interaction/policy I/O contract needed to validate observations and actions;
- any adapter state required for inference.

It SHALL NOT require constructing the training environment or RL algorithm runtime.

#### Scenario: Continuous PPO policy is exported

- **WHEN** the neural pipeline emits Gaussian parameters and the action adapter performs bounded sampling
- **THEN** the exported policy includes enough adapter configuration/state to map observations to valid actions
- **AND** loading the artifact does not require PPO, a critic, replay state, or a live environment.

#### Scenario: Direct-action policy is exported

- **WHEN** a pipeline already emits valid action tensors
- **THEN** the direct action adapter and contract are preserved
- **AND** inference returns the expected action keys through the same `CompiledPolicy` interface.

### Requirement: Trainable pipeline identity remains stable

`scenario.pipeline` SHALL remain the primary trainable model graph and stable learned-weight identity. The complete policy wrapper SHALL not obscure which parameter subset corresponds to the neural pipeline.

#### Scenario: Future NexuFL selects federated policy weights

- **WHEN** a downstream system needs the trainable policy parameters
- **THEN** the pipeline state can be identified independently of environment/replay state
- **AND** adapter/config metadata remains available to reconstruct executable inference.

### Requirement: Training-only RL state is not part of the normal policy artifact

Replay buffers, collected trajectories, live environments, simulator state, ROS handles, collectors, optimizer state, and algorithm-owned critic/Q/target networks SHALL NOT be silently embedded into the ordinary portable policy artifact.

#### Scenario: PPO run is exported

- **WHEN** PPO has a value network and collected rollout state
- **THEN** the ordinary policy package excludes that training-only state
- **AND** includes only state required for deployable policy inference.

### Requirement: Interaction contract is exported without a live environment

The export SHALL preserve the resolved portable interaction/policy I/O contract or an equivalent stable representation sufficient to validate expected observation and action keys, shapes, dtypes, modalities, and bounds.

#### Scenario: Robot policy is deployed

- **WHEN** an exported policy expects camera, joint-state, and force observations
- **THEN** the artifact can report those expected inputs without connecting to ROS or a simulator
- **AND** consumers can validate incoming TensorDicts before inference.

### Requirement: Export records RL and environment provenance

RL policy export metadata SHALL identify that the policy was trained with reinforcement learning and SHALL record stable provenance sufficient to diagnose how the policy was produced.

Metadata SHALL include at least:

- training mode;
- RL algorithm kind/name/version;
- environment kind/name/version;
- action-adapter kind/name/version where present;
- resolved interaction-contract/config hash;
- relevant resolved scenario/config provenance hashes.

#### Scenario: User inspects RL export metadata

- **WHEN** an exported RL policy's metadata is inspected
- **THEN** its algorithm, environment, adapter, and interaction-contract identities are visible
- **AND** no live runtime state is needed to inspect them.

### Requirement: Resume checkpoint and deployable policy are distinct artifacts

A Lightning checkpoint for an RL run MAY include optimizer and training-only algorithm state required for local resume, but SHALL NOT redefine the portable policy package contract.

#### Scenario: User resumes local PPO training

- **WHEN** a compatible RL Lightning checkpoint is supplied
- **THEN** NexuML restores actor/pipeline parameters, critic/value parameters, optimizers, counters, and explicitly checkpointable algorithm state
- **AND** environment/collector resources are reconstructed from definitions rather than unpickled as live resources.

#### Scenario: User deploys the policy

- **WHEN** the separately exported policy artifact is loaded
- **THEN** no checkpoint-only critic/optimizer/replay state is required
- **AND** deterministic or stochastic action inference is available according to the exported action adapter.

### Requirement: Artifact boundary permits later federated and fleet learning

The export contract SHALL keep learned policy state separable from local environment experience so a future NexuFL client procedure can exchange selected policy updates without requiring raw trajectories.

#### Scenario: Future NexuFL client consumes the policy

- **WHEN** a downstream federated client loads the NexuML policy artifact
- **THEN** trainable policy parameters can be selected independently of replay/trajectory/environment state
- **AND** the client has enough policy/interaction metadata to reconstruct local inference/collection.

#### Scenario: Local Physical-AI data remains local

- **WHEN** a future client trains from camera, lidar, force, ROS, or simulator observations
- **THEN** the normal portable policy contract does not require those raw observations or trajectories to leave the client.
