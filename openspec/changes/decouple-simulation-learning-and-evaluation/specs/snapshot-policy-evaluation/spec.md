## ADDED Requirements

### Requirement: Evaluation has independent placement and immutable policy inputs

Evaluation SHALL support qualified separate process or Ray worker placement independently of training and collection. A consistent snapshot SHALL include complete deployable actor/processor/action-adapter state, manifest, digest and source version/frame. The evaluator SHALL NOT modify learner state or use a mutable live model.

#### Scenario: A periodic evaluation is requested during learning
- **GIVEN** A compatible separate evaluation resource is available
- **WHEN** the corresponding supported workflow is executed
- **THEN** A worker evaluates the identified immutable snapshot while learning can continue, and results retain the snapshot's source version rather than the latest learner frame.

### Requirement: Evaluation scheduling is bounded and observable

Independent evaluation SHALL use bounded concurrency/queueing, explicit periodic coalescing or blocking policy, visible failure/cancellation status and required final evaluation completion before successful finalization. Results SHALL reuse configured tracking and artifact publication, with explicit seed, task/suite identity and bounded media.

#### Scenario: Evaluation is slower than the training interval
- **GIVEN** A new periodic evaluation request arrives while one is running
- **WHEN** the corresponding supported workflow is executed
- **THEN** The configured bounded scheduling policy is applied and any skipped/coalesced request is recorded; final evaluation is not silently discarded.

### Requirement: Standalone and native evaluation semantics are preserved

Standalone checkpoint/package evaluation SHALL use existing public commands and the selected native or generic evaluator. Generic episode evaluation and upstream benchmarks SHALL preserve their own scoring and reset rules. Unsupported automatic in-process native evaluation SHALL continue to fail closed.

#### Scenario: An exported policy is evaluated through the existing CLI
- **GIVEN** The benchmark and policy package are compatible
- **WHEN** the corresponding supported workflow is executed
- **THEN** The evaluator runs without constructing a learner or modifying the artifact and publishes native outcomes.


### Requirement: Native fit evaluation is independently placed and rank-owned

Qualified Isaac periodic/final evaluation SHALL use an independently allocated compatible native process/worker and immutable snapshot; in-process or shared unsupported app/GPU evaluation remains rejected. Distributed learning SHALL have exactly one designated snapshot/report owner, coordinated finalization, explicit bounded scheduling and no rank-local duplication or stale source frame. Independent viewer/evaluator failure SHALL follow explicit policy and never mutate training.

#### Scenario: A distributed learner requests periodic Isaac evaluation
- **WHEN** a compatible separate native GPU resource is admitted
- **THEN** only the designated owner submits the complete snapshot at its global frame, the evaluator runs in its own app, and other ranks continue or synchronously wait according to declared scheduling without deadlock.
