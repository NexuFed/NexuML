# Spec Delta

## Purpose

Preserve physical meaning and complete inference behavior across simulation, offline learning, exported policies and robot runtimes without creating a separate robotics model lifecycle.

## ADDED Requirements

### Requirement: Physical-system compatibility is machine readable

Each physical task and policy SHALL identify embodiment revision, asset identity, ordered joints, control mode, units, frames, bounds, control period, sensors and calibration requirements. Equal tensor shapes SHALL NOT establish compatibility. Definitions SHALL remain describable and serializable without native resources.

#### Scenario: A same-width incompatible controller is selected
- **WHEN** joint order, units, action scale, reference frame or control period differs from the exported policy
- **THEN** execution rejects the pairing before applying an action

### Requirement: Multimodal inputs retain preprocessing meaning

Policies SHALL declare required image, depth, proprioception, language and other sensor inputs with layout, dtype, normalization, timestamps, validity and processor identity. Raw instructions SHALL pass through the recorded tokenizer or processor rather than being silently replaced or coerced into numeric observations.

#### Scenario: A required camera or instruction is missing
- **WHEN** inference receives an incomplete or incompatible observation
- **THEN** validation reports the missing modality or processor without substituting privileged simulator data

### Requirement: Temporal execution is explicit and isolated

Policies SHALL declare history, recurrent or latent state, action-chunk horizon, execution stride and reset behavior. State SHALL be isolated per environment or agent, cleared on the appropriate episode boundary and owned by the caller or an explicitly resettable inference runtime.

#### Scenario: One row resets during chunked inference
- **WHEN** a batched policy has pending actions and only one episode ends
- **THEN** that row's pending actions and recurrent context are invalidated while continuing rows retain their state

### Requirement: Padding and time alignment do not become observations

Sequence validity, padding, sensor freshness and alignment policies SHALL be explicit. Invalid future targets SHALL be excluded from training loss; inference SHALL reject stale required inputs according to the task's recorded limits rather than silently reusing a prior episode's data.

#### Scenario: A training chunk crosses an episode boundary
- **WHEN** future actions are unavailable at the end of an episode
- **THEN** the chunk is padded and masked without borrowing actions from the next episode

### Requirement: Privileged training inputs cannot leak into actors

Training-only critic, teacher, contact or oracle observations SHALL be distinguished from deployable actor inputs. Export SHALL include only the inputs and components required by the selected deployable policy, with any remaining privileged dependency clearly disqualifying hardware-readiness claims.

#### Scenario: A teacher uses object ground truth
- **WHEN** a camera student is exported for deployment
- **THEN** the artifact's required inputs and parity test contain camera and measured state but no hidden object pose or teacher critic

### Requirement: Inference packages contain the full inference computation

Exported policies SHALL retain the trained inference graph, required processors and normalization, action decoding, inference-state initialization, compatibility metadata and integrity identities. Optimizers, training critics, replay, raw recordings and live simulator handles SHALL NOT enter an inference-only package.

#### Scenario: An ACT or Dreamer policy is loaded cleanly
- **WHEN** a trusted exported package is loaded without the simulator, collector and training learner
- **THEN** its complete required processor or latent-filter behavior executes with recorded dependencies and matches reference inference

### Requirement: Package trust and existing export semantics are preserved

Physical-policy packaging SHALL preserve existing general export and dependency-manifest behavior. External executable packages, reset datasets and remote model code SHALL require explicit trust and content verification; source provenance SHALL NOT be represented as cryptographic proof of safety.

#### Scenario: Remote model code is not approved
- **WHEN** a selected policy requires executing downloaded code
- **THEN** loading stops with a trust diagnostic rather than enabling remote execution automatically

### Requirement: Non-robot workflows retain their contracts

Physical-policy additions SHALL NOT require robot actions, episodes, ROS, language or simulation in ordinary supervised, self-supervised or industrial prediction scenarios. Existing valid configurations and portable policy packages SHALL remain supported or receive an explicit tested schema migration.

#### Scenario: An existing industrial classifier is restored
- **WHEN** it is compiled, trained and exported after the Physical AI extensions
- **THEN** it retains its normal dataset, model and inference lifecycle without constructing interaction resources
