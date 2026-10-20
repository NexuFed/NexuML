# Spec Delta

## Purpose

Export complete physically meaningful actors and preserve trusted local-learning compatibility without coupling inference to simulators or federation.

## ADDED Requirements

### Requirement: PD1 Actor-only packages preserve complete inference

Complete policy export SHALL preserve actors, sharing, processing/interface ownership, temporal reset and physical action decoding with reference parity. Clean inference SHALL require neither Isaac, TorchRL, Viser, custom simulator/config packages nor centralized critic inputs. Training-only state SHALL stay out of inference payloads. General train-export entries, sidecars and dependency manifests SHALL remain unchanged.

#### Scenario: A MAPPO multimodal actor is loaded cleanly
- **WHEN** simulator, learner, viewer and custom task imports are blocked and source-workspace paths are absent
- **THEN** the package runs from declared actor inputs, preserves shared/independent outputs and decoded actions over reset sequences, and does not reconstruct task factories or critics

#### Scenario: Native network export is limited by the pinned stack
- **WHEN** a training actor requires an unsupported graph-export path on PyTorch 2.11
- **THEN** the integration uses a qualified portable actor/weight transfer or rejects that path without upgrading dependencies or claiming a stateful network automatically removes TorchRL dependencies

### Requirement: PD2 Semantic compatibility and provenance are distinct

SO-101 validation SHALL separately record semantic task/data identity and implementation/capture provenance. Asset, task behavior, physical controls, sensors/processors, temporal rules and dataset meaning SHALL determine compatibility. Provenance/content identities SHALL remain verifiable and unchanged; tampering or semantic mismatch SHALL reject. Equal shapes or source hashes alone SHALL NOT establish compatibility or native authenticity.

#### Scenario: Equal-width actions have changed physical meaning
- **WHEN** joint order, units, scale, control period, asset or task/processor semantics differ
- **THEN** admission rejects before fit/replay/action and any new meaning requires matching newly captured data/evaluation rather than relabeled old samples

#### Scenario: Dataset or capture provenance is modified
- **WHEN** content checksums, required provenance, splits or physical validation fail
- **THEN** validation rejects and no implementation requalification waives the integrity failure

### Requirement: PD3 Implementation changes require proportionate requalification

An implementation-only change SHALL NOT inherently invalidate semantically identical current-schema demonstrations. Reuse SHALL require integrity and affected qualification tied to original data/capture identity and current implementation. Behavior-affecting or unknown-equivalence changes SHALL fail closed for reuse until qualified; changed semantics SHALL require new matching capture/evaluation. Historical data/evidence SHALL NOT be rewritten.

#### Scenario: Viewer initialization changes but task semantics do not
- **WHEN** physical/data semantics and integrity remain equal and affected runtime/SDK/native regressions pass
- **THEN** the original dataset remains reusable without automatic expert regeneration, original provenance is retained, and new qualification is attributed to the current implementation rather than rewriting prior evidence

#### Scenario: A physics or capture change has uncertain effects
- **WHEN** bounded native checks cannot establish equivalent task/data interpretation
- **THEN** reuse is blocked and new matching capture/evaluation is required if equivalence cannot be established, without adding legacy readers or hash-bypass allowlists

### Requirement: PD4 The working local-fit SDK retains its boundary

NEX-234 SDK entry points and validated requests SHALL preserve CPU/reset defaults, explicit approved CUDA, deterministic bounded fits, named parameters/scopes, finite updates and physical/schema checks. Raw private episodes and optimizer state SHALL remain local. Installed-wheel/offline imitation SHALL work without simulator/viewer/MARL requirements. Original SO-101 numerical/physical behavior SHALL remain unchanged.

#### Scenario: Two bounded fits use different initial states
- **WHEN** trusted callers run seeded CPU/reset fits and a separately approved CUDA/scoped fit
- **THEN** exact initial weights, scope/frozen tensors, counts and numerical outputs validate without leaking optimizer state, private paths or episodes

#### Scenario: Untrusted model or local-data selection is requested
- **WHEN** a remote request attempts executable loading, an unapproved dataset binding or incompatible manifest
- **THEN** the SDK rejects before fitting and produces no private payload or update

### Requirement: PD5 Consumer compatibility is explicit and independently owned

Changes to active SO-101 fixtures/manifests SHALL include producer tests and a compatibility check against the actual NEX-235 consumer revision. Exact source-pin, recipe/schema and numeric deltas SHALL be reported. NexuFL adjustments SHALL remain independent follow-ups with no source edits in this change; failed pins SHALL NOT be bypassed or represented as compatibility. NexuML SHALL NOT acquire federation/transport responsibilities.

#### Scenario: A consumer pins earlier SDK source
- **WHEN** the reviewed NEX-235 consumer rejects changed SDK/scenario files or a new semantic/recipe hash
- **THEN** the exact old/new contract and independent requalification work are recorded, consumer acceptance remains blocked, and its source pins are not patched to manufacture a pass

### Requirement: PD6 Completion states and evidence remain truthful

Each milestone SHALL separately report implementation, unit/API, actual native/browser acceptance and blocked/unverified states. Required gates SHALL pass before milestone completion; skips or mocks SHALL NOT count. Focused verified commits SHALL exclude raw execution evidence and unrelated work. Full completion SHALL retain Viser and both algorithms and SHALL NOT close the broader Physical AI roadmap.

#### Scenario: Software checks pass without required native evidence
- **WHEN** graphics, task assets, browser safety or consumer qualification is missing
- **THEN** the affected gates/milestones remain open with exact prerequisites, original historical evidence stays intact and no broader capability is marked complete
