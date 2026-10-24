# Spec Delta

## Purpose

Use native Isaac experience in NexuML's existing TorchRL learning lifecycle with correct algorithm semantics, checkpoint ownership and measurable update evidence.

## ADDED Requirements

### Requirement: One NexuML learning lifecycle owns each run

Isaac learning SHALL use the ordinary NexuML configuration, session, learning, logging, checkpoint and export workflow. Upstream training runners SHALL remain independent baselines rather than hidden substitute trainers. Simulation choice SHALL NOT become a second execution or session architecture.

#### Scenario: A native upstream baseline succeeds first
- **WHEN** NexuML support for that task is assessed
- **THEN** independent NexuML collection and optimizer updates are still required before its training capability is complete

### Requirement: Device and batch accounting are truthful

Environment, time and agent axes SHALL remain distinct until the selected learner's reduction. Actual physical transitions SHALL determine frame budgets; imagined samples and optimization steps SHALL be counted separately. Supported GPU hot paths SHALL avoid CPU/NumPy action and observation round trips.

#### Scenario: Eight environments produce sixteen time steps
- **WHEN** the rollout reaches the learner
- **THEN** the physical frame count increases by 128 and temporal estimation preserves each environment's episode identity

### Requirement: PPO targets and behavior policies remain consistent

PPO SHALL compute time-limit values from real final observations, distinguish true termination and preserve action sampling/log-probability semantics. Updates and rollout-policy synchronization SHALL occur in a tested on-policy order without undeclared clipping or reuse of stale behavior-policy data.

#### Scenario: A controlled timeout has a known final-state value
- **WHEN** advantages and returns are computed
- **THEN** numerical targets match the final-state oracle rather than reset-state values and the next rollout uses the updated policy

### Requirement: Off-policy and offline learning use their own data semantics

Supported off-policy and offline workflows SHALL declare replay ownership, sampling, warm-up, update ratio, target updates and dataset restrictions. They SHALL NOT inherit PPO-only assumptions or silently collect new physical experience during an offline-only run.

#### Scenario: An offline learner is run on recorded episodes
- **WHEN** its configured training budget executes
- **THEN** actual parameter updates and held-out evaluation use only authorized data and no simulator or hardware collection is started

### Requirement: Recurrent and asymmetric policies preserve deployability

Recurrent policies SHALL reset state per episode and carry valid sequence masks through learning. Asymmetric critics SHALL consume only explicitly training-only inputs; exported actors SHALL remain executable from their declared deployable observations.

#### Scenario: The critic uses privileged contact measurements
- **WHEN** the trained actor is exported and replayed
- **THEN** its inference succeeds without the critic or contact oracle and partial resets clear only affected recurrent streams

### Requirement: Multi-agent learning retains agent identity

Multi-agent workflows SHALL preserve named agent groups, per-agent actions, shared or independent policy identity, reward reduction and agent/environment completion semantics. A native multi-agent task SHALL have a real learning and replay proof; flattening agents into unrelated environments SHALL NOT count.

#### Scenario: One agent ends before the environment episode
- **WHEN** experience is stored and recurrent state is reset
- **THEN** the configured agent termination rule is respected without corrupting continuing agents or the global task boundary

### Requirement: Resume restores algorithm-specific state

Resume SHALL restore trainable and target networks, optimizers, normalization, counters, update scheduling and required replay/state according to the selected algorithm. Restarted physics SHALL be fresh. Omitted replay SHALL follow an explicit refill policy, not fabricate trajectory continuity or owed updates.

#### Scenario: A replay-based run resumes without saved replay
- **WHEN** the checkpoint's declared refill policy is used
- **THEN** learning waits for valid data and resets incompatible pending update accounting before new optimization

### Requirement: Each learning family has independent acceptance

On-policy, off-policy, offline, recurrent/asymmetric and multi-agent support SHALL each include focused numerical tests, changed-parameter evidence, checkpoint resume, export and fresh evaluation. A finite loss or another algorithm's successful run SHALL NOT establish support.

#### Scenario: Only the PPO reference has been exercised
- **WHEN** a capability report is generated
- **THEN** unexecuted off-policy, offline and multi-agent workflows remain visibly incomplete

### Requirement: Offline-to-online and corrective RL retain algorithm ownership

Simulation-first RLPD and HIL-SERL workflows SHALL reuse the existing algorithm-owned replay/optimizer/session boundaries and separately track old demonstrations, newly sampled simulator interactions, reward classifier inputs and corrected/executed actions. A native JAX/MJX upstream PPO trainer SHALL NOT be mislabeled TorchRL learning even when its simulator uses MJWarp.

#### Scenario: An upstream Playground JAX PPO run succeeds
- **WHEN** NexuML-owned GPU learning is reported
- **THEN** that upstream result remains an external benchmark until real NexuML actor/optimizer updates, checkpoint resume and portable inference have run
