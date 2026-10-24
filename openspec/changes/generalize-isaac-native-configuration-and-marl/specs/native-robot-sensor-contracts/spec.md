# Spec Delta

## Purpose

Preserve real sensor and physical control meaning from native simulation through learning, episode boundaries and portable policy interfaces.

## ADDED Requirements

### Requirement: SC1 Sensor compatibility includes meaning beyond shape

Policies SHALL support state, RGB, depth, segmentation and mixed observations with explicit layout, dtype, units, normalization, depth representation/invalid-value handling, segmentation labels and processor identity. Actor tensors SHALL be finite after declared processing. Missing, undeclared or semantically incompatible required inputs SHALL reject instead of being silently inferred or replaced.

#### Scenario: Equal-shaped depth or segmentation inputs differ
- **WHEN** input uses incompatible depth units/sentinels or a different segmentation vocabulary
- **THEN** compatibility validation identifies the semantic mismatch before inference or replay rather than accepting matching dimensions

#### Scenario: Actual mixed sensor observations are produced
- **WHEN** a declared native task supplies state and image observations
- **THEN** bindings expose actual qualified sensor values with recorded identities and no spectator-only camera or privileged ground truth added to the actor

### Requirement: SC2 Processing and temporal ownership are explicit

Every inference preprocessing/history operation SHALL be owned by the portable package or an external runtime with an identically defined tensor interface. Contracts SHALL retain processor assets/version, history ordering/initialization/reset, sensor/control rates and timing. When the package owns history, the existing stateless `CompiledPolicy` SHALL gain a **minimal opt-in, caller-owned processing-state path** with explicit initialization, observation/state-to-action/next-state stepping, and selected environment/group reset masks; it SHALL NOT rely on hidden global state or change ordinary stateless `forward()`. External runtime-owned history SHALL instead explicitly own its reset and timing semantics. Export SHALL NOT claim raw-frame input when external processing is required. No separate recurrent trainer, policy hierarchy or preprocessing framework SHALL be introduced.

#### Scenario: History is externally supplied
- **WHEN** a policy expects normalized stacked frames from an external runtime
- **THEN** the artifact declares that interface, its buffer ordering/reset and rates, and rejects raw single frames or incompatible processing

#### Scenario: Package-owned processing is reloaded
- **WHEN** representative RGB/depth/segmentation sequences pass through the exported package with a newly initialized caller-owned state
- **THEN** normalization, invalid-value treatment, state updates and decoded actions match the training reference, including reset boundaries

#### Scenario: One environment resets while another continues
- **WHEN** the caller applies a reset mask to a stateful package-owned history for one stream in a batch
- **THEN** only the selected history returns to its declared initial state, all other streams continue with unchanged histories and actions, and the existing stateless `forward()` results remain unchanged

### Requirement: SC3 Pre-reset sensor data belongs to the ending episode

Native transitions SHALL retain genuine pre-reset observations and required history/freshness for terminal and time-limit cases. Same-step autoresets SHALL NOT substitute next-episode images/state or trigger duplicate resets. Continuing environments/agents SHALL retain their temporal context while ended streams reset according to the declared contract.

#### Scenario: One stream ends with a rendered terminal observation
- **WHEN** the native simulator autoresets that stream in the same step
- **THEN** learning/bootstrap uses the ending episode's actual rendered observation and appropriate boundary mask, while cached reset data starts the next episode exactly once
- **AND** continuing streams retain their history

### Requirement: SC4 Action and sensor support requires real native updates

Continuous and scalar categorical controls SHALL preserve declared bounds, event dimensions, class mappings, physical scale/offset, joint ordering and period. Native sensor/control acceptance SHALL include real RGB/depth/segmentation, mixed inputs and categorical control with NexuML updates and complete export/replay. Viewer geometry and tensor-shape tests SHALL NOT substitute for sensor evidence.

#### Scenario: A categorical force controller is introduced
- **WHEN** no installed reference provides the required scalar categorical semantics
- **THEN** a distinctly registered reference maps integer classes to declared physical controls without relabeling the existing continuous task

#### Scenario: Sensor production is not qualified
- **WHEN** shapes are correct but native values, reset freshness or required graphics support have not been demonstrated
- **THEN** the affected sensor/control combination remains blocked or unverified and is not advertised as supported
