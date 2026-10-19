# Spec Delta

## Purpose

Evaluate Physical AI policies reproducibly across Isaac tasks, Arena variations and recorded experience, producing reviewable comparisons without equating simulation results with safety certification.

## ADDED Requirements

### Requirement: Policy evaluation is independent of its training mode

Exported offline-trained policies SHALL carry a validated evaluation environment and trainer-independent episode/media configuration. The ordinary evaluation command SHALL accept them without fabricating RL training settings or constructing a learner. Existing RL evaluation configuration and overrides SHALL retain a documented compatible path.

#### Scenario: An offline ACT or VLA package is evaluated
- **WHEN** the ordinary evaluation command loads its original non-RL scenario and complete evaluation configuration
- **THEN** fresh simulation evaluation and requested recording run without a training session, optimizer, collector or upstream training runner
- **AND** the package's original training mode is not rewritten as reinforcement learning

#### Scenario: An offline package has no evaluation environment or configuration
- **WHEN** closed-loop replay is requested
- **THEN** the command reports the missing evaluation input without inventing RL algorithm settings or starting a learner

### Requirement: Release evaluation is frozen and traceable

An evaluation suite SHALL identify task/asset versions, policy and processor hashes, embodiment/calibration, observation/action contract, distributions, seeds, horizons, sample counts, metrics and reference policy. Changed protocols SHALL receive distinct identities rather than silently replace prior results.

#### Scenario: A candidate uses a different reset distribution
- **WHEN** its result is compared with a reference run
- **THEN** the mismatch is exposed and direct promotion comparison is withheld until equivalent conditions are evaluated

### Requirement: Task outcomes are independent of reward

Evaluation SHALL validate task success and available task-specific errors without requiring fabricated metrics. Success accumulation SHALL follow the frozen task predicate, and outcomes SHALL retain episode identity across reset, batching and recording.

#### Scenario: Vial placement is rewarded near the rack but never released
- **WHEN** the episode is scored
- **THEN** shaped return is recorded but physical placement success remains false under the task's seating/release/hold predicate

### Requirement: Batched evaluation counts completed episodes correctly

Supported GPU evaluation SHALL track complete episodes per row, recurrent/chunk resets, requested episode budgets and native final states without mixing episode returns or counting partial episodes as complete. Scalar evaluation SHALL retain its existing behavior.

#### Scenario: Rows finish at different times
- **WHEN** the requested complete-episode budget is reached
- **THEN** results identify the exact counted episodes and any discarded partial work without double-counting autoreset transitions

### Requirement: Arena integration reuses maintained task composition

The backend SHALL evaluate at least one compatible NexuML policy on a pinned Isaac Lab-Arena suite using maintained task/embodiment composition. Suite support SHALL require native compatibility and policy-interface proofs, not copying an obsolete installation recipe or rebuilding Arena inside NexuML.

#### Scenario: An Arena recipe requires incompatible Python or Isaac versions
- **WHEN** the shared stack is resolved
- **THEN** the conflict is reported and Arena remains incomplete until a compatible maintained integration is validated

### Requirement: Robustness and forgetting are measured explicitly

Evaluation SHALL include held-out conditions, multiple training seeds where applicable, old/new-task regression and task-relevant failure categories. Success uncertainty, worst-condition results, latency tails, dropped inputs and constraint violations SHALL remain distinct from mean return.

#### Scenario: Fine-tuning improves the mean while failing a held-out condition
- **WHEN** candidate-versus-reference evidence is assembled
- **THEN** the failing condition and forgetting results remain visible rather than being hidden by the average

### Requirement: Promotion decisions represent evidence sufficiency

Candidate comparison SHALL produce pass, fail or insufficient evidence against recorded thresholds and minimum sample requirements. Model-release decisions SHALL remain distinct from operator authorization for hardware use and SHALL NOT claim certified robot safety.

#### Scenario: Only a tiny optimizer smoke was evaluated
- **WHEN** a deployment-quality decision is requested
- **THEN** the report returns insufficient evidence instead of treating finite losses as release readiness

### Requirement: Offline shadow evaluation precedes new physical use

Recorded-observation evaluation SHALL exercise processor, state-reset and action parity without publishing physical commands. Its limitations SHALL distinguish action agreement from counterfactual task success, which requires an appropriate simulator or supervised physical trial.

#### Scenario: A candidate predicts plausible actions on a recording
- **WHEN** no closed-loop trial has been performed
- **THEN** the report does not infer physical task success from those predictions alone

### Requirement: Existing logs and evidence remain the publication path

Evaluation SHALL reuse configured NexuML scalar/media loggers and publish machine-readable trials, provenance, failures and comparison reports. Video SHALL be bounded and preserve terminal-scene identity; new evaluation services or duplicate logger runs SHALL NOT be required.

#### Scenario: Recorded evaluation is cancelled
- **WHEN** cancellation occurs before media finalization
- **THEN** cancellation remains visible, native loggers finalize truthfully and incomplete media is not published as a successful artifact
