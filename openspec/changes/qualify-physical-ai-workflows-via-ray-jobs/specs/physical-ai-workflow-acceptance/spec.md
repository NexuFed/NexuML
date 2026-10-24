# Spec Delta

## Purpose

Make release claims for the implemented Physical AI foundation depend on reproducible public CLI journeys and current artifact identities, independently of historical tests or broader roadmap completion.

## ADDED Requirements

### Requirement: W1 Advertised recipes have an explicit acceptance inventory

The release inventory SHALL enumerate every advertised runnable recipe, its exact task/backend/environment prerequisites and its required CLI, visual, learning and recovery gates. Source discovery SHALL be reconciled with that inventory. Missing or newly discovered recipes SHALL remain unverified until classified rather than silently omitted.

#### Scenario: A registered scenario has no acceptance row
- **WHEN** discovered reinforcement recipes are compared with the frozen inventory
- **THEN** the missing recipe is reported and inventory acceptance fails
- **AND** unavailable external integrations are identified as boundaries, not passed runnable examples

### Requirement: W2 CLI proof executes real public commands

A CLI acceptance pass SHALL execute the documented public commands against real selected environments and dependencies on remote workers. It SHALL verify actual optimizer progress, changed trainable parameters, finite outputs and identified artifacts. Static resolution, mocked submission, direct session calls and a successful CLI help command SHALL NOT substitute for this proof.

#### Scenario: A tiny native training command exits successfully
- **WHEN** its trained actor and checkpoint are inspected
- **THEN** recorded counters and parameter changes prove that updates occurred within the declared budget
- **AND** a zero-update run is rejected as training acceptance even if it printed completion

### Requirement: W3 Continuation and inference are separate journeys

Every supported learning recipe SHALL demonstrate interrupted-budget resume, checkpoint export and fresh-process evaluation. Resume SHALL restore the existing actor/critic/optimizer/counter semantics and advance the remaining budget. Exported-policy evaluation SHALL use the identified saved weights without constructing a learner or silently retraining a replacement model.

#### Scenario: A partial checkpoint resumes
- **WHEN** the remote resume command starts from the retained partial checkpoint
- **THEN** counters continue from the saved state and complete only the configured remaining budget
- **AND** the resulting policy is separately exported and evaluated in a fresh process

#### Scenario: The checkpoint already exhausted its budget
- **WHEN** continuation is requested for the completed checkpoint
- **THEN** no additional collection or native allocation is performed where the existing lifecycle forbids it
- **AND** completion is not represented as evidence of new optimizer updates

### Requirement: W4 Evidence is revision bound and multidimensional

Acceptance SHALL report CLI, visual, learning and recovery outcomes separately as pass, fail, blocked or unverified. Each result SHALL identify the source/artifact revision, effective configuration, runtime versions, case/attempt/job identities, commands, exit codes and retrievable evidence. Historical or changed-runtime results SHALL retain their original scope.

#### Scenario: Only native fixture tests passed
- **WHEN** no matching public CLI journey exists for the candidate
- **THEN** the CLI gate remains unverified
- **AND** the genuine native test result remains visible without being relabeled as CLI acceptance

#### Scenario: Code changes after qualification
- **WHEN** a subsequent patch affects a qualified path
- **THEN** its affected gates require fresh evidence on the new candidate
- **AND** prior results are neither deleted nor silently attributed to the new revision

### Requirement: W5 Learning showcases use frozen outcome protocols

Examples advertised as demonstrating learning SHALL compare identified untrained/reference and trained policies under a frozen held-out protocol. Budgets, seeds, native horizons, success predicates, thresholds and denominators SHALL be recorded before fitting. Finite updates, attractive videos and smoke-only shortened horizons SHALL NOT establish learning quality or hardware safety.

#### Scenario: The learned policy is scored against a baseline
- **WHEN** the frozen showcase protocol runs
- **THEN** both policies use matching reset distributions, horizons and outcome rules
- **AND** failures, action rejections, timeouts and training-seed limitations remain explicit

#### Scenario: Only one training seed is qualified
- **WHEN** the showcased run passes its held-out reset episodes
- **THEN** the report claims only that recorded training-seed scope
- **AND** it does not claim multiseed robustness or unseen-task generalization

### Requirement: W6 Installed distributions preserve deployment boundaries

Acceptance SHALL exercise the candidate's built distributions with editable source roots excluded. Complete actor inference SHALL remain independent of native simulators, training-only algorithms and recording tools where the existing package contract permits. At least one ordinary supervised model and the existing SO-101 local-fit workflow SHALL retain their documented behavior.

#### Scenario: A policy loads from the installed candidate
- **WHEN** training, native simulator and recording imports are unavailable
- **THEN** declared actor inputs still produce the qualified outputs
- **AND** the package is not silently replaced with an editable checkout or a different build

### Requirement: W7 Release qualification follows branch integration

The candidate SHALL be reconciled with the actual PR base before final acceptance. Required static, declared-Python, regression, documentation and distribution checks SHALL pass for the identified final candidate. Known failures SHALL be diagnosed explicitly; weakening selectors, suppressing failures or using old successful runs SHALL NOT manufacture a green release gate.

#### Scenario: The published head differs from the tested head
- **WHEN** PR readiness is reported
- **THEN** the mismatch is visible and affected required checks remain pending
- **AND** readiness is not inferred from older check runs or local-only commits

### Requirement: W8 Prerequisite failures do not erase required coverage

Native GPU, viewer, middleware and remote-browser gates SHALL name their exact prerequisites and report absent resources as blocked. Required live-view journeys SHALL not become passed through fake transport, virtual-display startup alone or static contracts. Any decision to ship without a required family SHALL require an explicit maintainer scope decision and revised claims.

#### Scenario: Gazebo is not prepared on a remote worker
- **WHEN** the live ROS/Gazebo viewer journey is selected
- **THEN** that gate reports the missing world/controller/reset/display prerequisites
- **AND** it neither executes on the local GPU nor treats fake-transport tests as the live result
