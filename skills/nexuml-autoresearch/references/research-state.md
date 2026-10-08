# Small, recoverable research state

Use the project's existing record format. Otherwise one Markdown ledger with
links to immutable run artifacts is enough; do not install a tracking service.
The following fields are a checklist, not four mandatory files or a new schema.

## Contract and continuation

Record the objective/direction, metric implementation, cohort/partition identity,
baseline and source/environment identity. Include allowed changes, hard resource
limits, total/per-run budget, repeat/admission rules, stopping conditions,
final-test gate, and active user steering. Give a resumed reader the next action
and the evidence or unresolved question that makes it useful.

Distinguish total, spent, reserved and remaining capacity. A run starts consuming
an attempt when the agreed accounting rule says so. Execution failures/retries
consume real resources; a proven pre-execution dispatch failure may release its
reservation but does not erase upload/API/time costs. Never manufacture capacity
by changing accounting after a failure. Reserve before launch when concurrent
work could otherwise overspend; use the existing job mechanism, not a scheduler.

## Hypotheses, not a list of arbitrary configurations

For each queued idea, retain:

| Field | Purpose |
| --- | --- |
| ID/parent and exact change | Distinguish the candidate from its baseline and siblings. |
| Mechanism and evidence | Explain why this could improve the objective or resolve uncertainty. |
| Rival explanation / cheap discriminator | State what would undermine the idea before a costly run. |
| Required metric/cohort and estimated cost | Check feasibility and comparison validity. |
| Priority and keep/reject/investigate rule | Explain why this is the next useful use of budget. |

Prefer a routing/gradient/scorer check over a large sweep when implementation
validity is uncertain. Prefer confirmation when a promising result is too noisy
for admission. Otherwise choose a feasible experiment that distinguishes plausible
mechanisms. Joint changes are legitimate when declared as such; do not attribute
their outcome to one factor. Drop duplicates and ideas whose premise was disproved.

## Attempts and lessons

For every attempt link its command, actual source/config/environment identities,
partition manifest, seed, resource use, raw metrics, logs and output artifacts.
Keep execution state (`planned`, `running`, `completed`, `failed`, `interrupted`,
`unknown`) separate from the scientific verdict (`keep`, `reject`, `investigate`).
Process exit or stage completion does not prove the metric/constraint checks passed.

A useful lesson is scoped: “Under cohort C and protocol P, run R did not support
hypothesis H; evidence E suggests checking X next.” Preserve negative/inconclusive
results. Add contrary evidence rather than quietly overwriting the history. Do
not generalize one seed, one dataset, or an agent's explanation into a fact.

## Comparable evidence and optional compact manifest

Before calling a candidate an improvement, verify that baseline and candidate
use the same research cohort and scorer, training/validation boundaries,
preprocessing, and permitted changes. Report meaningful differences in effective
batch size, optimizer/scheduler, update count, training examples, precision,
hardware, parameter count, wall time and compute/communication costs when those
affect the claim. A matched number of epochs is insufficient if data exposure
or work per epoch differs. Identify exact baselines and ablations for a new
method, including a cheap falsifier for its proposed mechanism.

Keep the project's record format. If absent, a single ledger linking immutable
per-run artifacts is enough. A useful **optional** per-run record, not a new
NexuML schema or mandatory second ledger, contains:

- ID/parent/hypothesis and changed parameters/components; exact runnable command.
- Git commit plus dirty source identity, installed package versions, resolved
  scenario and environment/hardware identity.
- Dataset manifest, split IDs or hash, seed(s), metric implementation and
  selection cohort/direction.
- Updates/epochs, examples seen, elapsed/resource costs and budget accounting.
- Raw results/artifact paths, actual completion state, validity gate,
  comparison verdict and remaining uncertainty.

Never replace source identity with the scenario-file snapshot alone. If work is
stopped or unavailable, record unknown fields honestly instead of inserting
zeros, stale estimates or implied success.

## Resume checklist

1. Read the contract, active steering, attempted runs, lessons and pending ideas.
2. Inspect any live job through the host's supported status/completion mechanism;
   use its identity and artifacts to prevent duplicate dispatch. Missing evidence
   stays unknown until reconciled; do not poll or cancel unrelated work.
3. Verify current source, data, scorer and environment against recorded identities.
   If incompatible, preserve the old study and identify a new comparison within
   authorized scope; ask only if it changes a consequential boundary.
4. Reconcile spent/reserved/remaining capacity from actual attempts. Keep failures
   and unspent capacity visible. Do not restart a verified baseline merely because
   the session is new.
5. Continue the highest-priority feasible next step under the standing contract,
   or stop on an explicit pause, completed target/budget, or unresolved blocker.

## Design sources and limits

These procedures adapt the procedural-knowledge and selective-loading principles
of [Kassis et al. (2026)](https://doi.org/10.48550/arXiv.2609.00065), read as v2,
and the hypothesis/lesson/continuation ideas in
[OpenScience's autoresearch documentation](https://github.com/synthetic-sciences/openscience/blob/847f7378e0099c52ef44f7b6526835b1bf726125/frontend/docs/src/content/openscience/autoresearch.mdx).
They do not copy its runtime, approval settings, tracking shim, database, fixed
review frequencies or four-file ledger. The paper describes its skill library;
it does not establish task-level efficacy of this NexuML workflow.
