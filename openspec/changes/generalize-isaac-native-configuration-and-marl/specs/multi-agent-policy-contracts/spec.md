# Spec Delta

## Purpose

Represent fixed multi-agent policies and transitions with portable ordered groups, explicit physical meaning and truthful environment accounting.

## ADDED Requirements

### Requirement: MG1 Fixed group identity and compatibility are explicit

Contracts SHALL persist fixed ordered agent/group identities and their observation/action semantics. Heterogeneous roles SHALL retain separate spaces and distributions without artificial padding. Compatible groups SHALL support shared actor parameters or separate actors; equal shape alone SHALL NOT authorize sharing. Dynamic membership SHALL remain unsupported rather than silently remapped.

#### Scenario: Roles have different observations or actions
- **WHEN** groups use different feature widths or continuous versus scalar categorical distributions
- **THEN** each retains its declared actor/bindings and ordered identity through save/reload and replay, and incompatible sharing requests reject

### Requirement: MG2 Tensor axes preserve environment time and agents

Grouped live tensors SHALL have root batch `[N]` and group `[N,A_g]`; rollouts SHALL have root `[N,T]` and group `[N,T,A_g]`. Per-agent actions, rewards, values, advantages and validity SHALL remain grouped. Action log probabilities SHALL reduce event dimensions only. GAE SHALL precede environment/time minibatch flattening; physical frame accounting SHALL count `N*T`, never agent samples.

#### Scenario: A heterogeneous rollout is minibatched
- **WHEN** native GAE finishes on `[N,T]` and the batch is flattened
- **THEN** root minibatches are `[M]`, group children remain `[M,A_g]`, value-like fields retain a trailing singleton and scalar categorical actions/log probabilities retain one entry per agent
- **AND** counters, budgets and resume advance by physical environment transitions only

### Requirement: MG3 Agent endings and environment resets stay distinct

Transitions SHALL retain per-agent termination/truncation and the actual native environment boundary. Terminal samples SHALL remain valid; inactive suffixes SHALL be invalid until native reset. Continuing agents SHALL keep their trajectories. Effective estimator boundaries SHALL reflect native task semantics, not invented any/all rules. Time-limit bootstrap SHALL use true pre-reset actor and critic data.

#### Scenario: One agent ends before its peers
- **WHEN** its terminal transition is followed by an inactive suffix while other agents continue
- **THEN** its terminal sample contributes correctly, later learning samples do not, and peers do not acquire a fabricated terminal/reset boundary

#### Scenario: Privileged final state is unavailable
- **WHEN** a centralized binding requires state not preserved by the public native final-observation interface
- **THEN** validation rejects that binding rather than bootstrapping from reset state or inventing privileged observations

### Requirement: MG4 Learning validity is not physical-state validity

Actor-visible observations SHALL be separate from declared training-only critic inputs. Inactive learning samples SHALL NOT erase still-valid world state needed by another agent's centralized critic. IPPO values SHALL depend on local agent inputs; MAPPO actors SHALL remain decentralized while critics can consume declared centralized state. Exported actors SHALL NOT require privileged inputs.

#### Scenario: A finished agent remains physically present
- **WHEN** its learning sample is invalid but its body is part of a continuing agent's declared world state
- **THEN** loss validity masks the inactive sample without deleting that genuine critic feature or adding it to actor inputs

#### Scenario: Another role's privileged observation is perturbed
- **WHEN** local actors/IPPO values and centralized MAPPO values are evaluated under that perturbation
- **THEN** local outputs remain independent while the qualified centralized critic responds through its declared input dependency

### Requirement: MG5 Rewards and outcomes retain their actual meaning

Per-agent rewards SHALL remain per-agent. Only genuine task-declared team rewards SHALL be broadcast to agent values. Reporting aggregates SHALL be explicitly named and SHALL NOT replace learning rewards or manufacture global success. Environment completion/success SHALL come from native declared outcomes, separately from agent trajectories and optimization metrics.

#### Scenario: Agents receive unequal rewards without a team signal
- **WHEN** learning and reporting consume the transition
- **THEN** each agent learns from its own signal, named aggregate metrics remain reporting-only, and no root team reward or global success is inferred from their mean
