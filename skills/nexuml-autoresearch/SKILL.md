---
name: nexuml-autoresearch
description: >-
  Run bounded, evidence-driven research with installed NexuML in a user's
  external library: search architecture and parameters for a specification,
  compare algorithm variants, implement missing reusable components and promote
  verified candidates. Use for NexuML autoresearch, autonomous experiments,
  architecture/hyperparameter search, improving a scenario metric or exploring
  new algorithms. Preserve baselines and select on research/validation data,
  never adapt candidates using final-test results.
compatibility: Compatible installed NexuML, Python >=3.12 and filesystem/shell access; Optuna only for native tuning.
---

# Bounded NexuML research

Aim for the best **verified feasible candidate within the authorized budget**,
not a promise of the globally best architecture. Evidence decides the next step.

## Establish the research contract

Locate the user's library, scenarios, prior evidence and selected environment.
Inspect installed NexuML's version/help/schemas and discovery diagnostics. Work
in that project, not the framework's base-library tree. Use an editable user
library or a deliberately configured local root; its dependencies must be in the
selected runtime. No NexuML checkout, MCP or API server is required.

Establish the contract once from the user's request and existing project policy:

- Objective metric, exact definition/direction and research cohort/partition.
- Hard requirements: data/protocol fidelity, latency/memory/size, hardware,
  reproducibility and allowed architecture/parameter/component changes.
- Budget: trials/experiments, per-run epochs/time, total compute/time and any
  permitted parallelism. Ask for missing consequential bounds; do not infer
  unlimited compute from “find the best”.
- Baseline identity, repeat/selection rules, required evidence and stop/admission
  gates, including when final-test evaluation is allowed.

## Continue autonomously within the contract

A bounded research request authorizes the experiment loop, not just its first
run. Choose, execute, inspect and continue experiments/confirmation runs within
that scope without asking for approval after each plan, result or routine fix.
Announce consequential decisions and record them; an update is not an approval gate.

Fix recoverable user-library/configuration problems and perform required local
checks within scope, counting retries against the budget. Ask only when proceeding
needs a real scope/resource/data-access change or a decision the contract cannot
resolve. Do not ask merely because the next hypothesis differs from the last one.

Honor explicit pauses/read-only requests. Remaining capacity is not exhausted
just because execution is paused, and a normal pause is not a permanent new
approval requirement after the user resumes the agreed loop.

Reuse the project's ledger/artifact convention. If absent, use one small ledger
and unique run folders; do not introduce a database, scheduler or fixed set of
four competing ledgers. Record assumptions before looking at results.

## Keep selection and final evaluation separate

Use validation/research metrics for proposals, tuning, pruning, checkpoint
selection, threshold selection and candidate admission. Keep official final-test
sample identities/data out of research runs where possible, and verify actual
loader membership—not just split ratios or a renamed `test/` metric.

Native training/tuning runs the configured lifecycle; optimizing `val/loss`
alone does not prevent a configured test phase. For ordinary random-split data,
set `test_split=0` and adjust train/val fractions during selection. For datasets
with fixed assignments inspect the resulting loaders and exclude final-test
records explicitly. A post-training evaluator may require a separately named
research-scoring cohort: it must contain **research data**, not the final-test
cohort, even if the framework exposes it through a test-stage interface.

Freeze the candidate, protocol and selection decision before final-test access.
Once test feedback guides a change, that cohort is no longer an independent
final test; disclose it and stop treating its score as unbiased acceptance.

## Preserve and verify the baseline

Read earlier source/configs, commands, raw scores, metrics and failed attempts.
Do not compare different partitions, scorers or resource protocols as if only
the architecture changed. Preserve historical evidence and correct any invalid
metric claims explicitly.

If no valid baseline exists, run a cheap sanity check followed by the agreed
baseline. Check data/routes, numerical/gradient behavior, portable configuration
and actual lifecycle before spending on search. A smoke pass is not convergence.

## Choose an experiment that answers a question

Before each run state: ID/parent, hypothesis, exact changes, cohort/scorer,
expected discriminator, budget and keep/reject/investigate rule. A controlled
single-variable comparison is useful for attribution, but do not force it when
a declared joint search or an algorithm change requires coupled parameters.

Reuse existing components/recipes first. Use native `nexuml tune` for supported
parameter/factory search, or explicit scenario variants for architectural and
algorithm hypotheses. Read [tuning boundaries](references/tuning.md) before
writing search spaces. Do not mandate tuning before every architecture change;
choose what addresses the observed bottleneck. Never rewrite the training loop.

When a real gap requires new code, add the minimal component to the user's
importable library, with a focused numerical/contract check. Use
`nexuml-library` guidance when available. Treat fitted/scoring state and ordinary
trainable parameters differently, and preserve train-only fitting.

## Execute and record actual evidence

Use exactly one source per CLI call and a fresh scenario/run identity. Scenario
files expose `scenario() -> ScenarioSpec`; `HYPOTHESIS`, `PARENT`, `TAGS` and
tuning exports are optional. Keep reusable classes/factories in importable
library modules rather than defining them in a dynamically loaded scratch file.

Use a fresh `--artifact-dir` for scenario-file runs. Record the actual command,
overrides, imported library source identity (including dirty changes), resolved
environment, data/partition manifest, metric/scorer, raw evidence as needed,
resource observations and result. The built-in snapshot is not a complete
snapshot of imported code/data, nor a substitute for real metric/log evidence.

Use the host's supported background completion mechanism for long jobs. Do not
silently start clusters or parallel workers, detach jobs without a way to learn
their outcome, or equate submission/process exit with remote completion.

After each run update the ledger with running/completed/failed/interrupted state,
metrics, constraints, decision and evidence paths. A framework failure is a
blocker/reproducer, not permission for site-packages patches or diagnostic
monkeypatches disguised as candidate implementations. Record a user-code fix as
a new identified attempt; do not quietly spend extra retries beyond the budget.

## Select, verify and stop

Admit only candidates meeting hard constraints and evidence gates. Confirm a
promising candidate under the agreed repetition budget; one lucky score or
unequal training budget does not establish an improvement. Report uncertainty,
infeasible candidates and unavailable resource measurements honestly.

Stop at budget/target/stop-rule completion, user pause/cancellation or a genuinely
blocking environment/protocol failure that cannot be corrected within scope.
Promote verified reusable code and ordinary
recipes/configs into the user's library without overwriting the baseline or
discarding failed-run evidence. Final-test execution remains separately gated.

Report best feasible candidate, selection metric/cohort, comparisons, compute
spent, exact reproduction command, evidence and remaining uncertainty. If none
qualified, say so rather than declaring the least broken run a winner.
