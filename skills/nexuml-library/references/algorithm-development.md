# Develop a novel algorithm with NexuML

Use this workflow when the goal is an **algorithmic research contribution**, not
merely another hyperparameter value. It combines `nexuml-library` for reusable
components, `nexuml-reproduce-paper` for faithful reference baselines when
applicable, and `nexuml-autoresearch` for bounded experiments. Only load the
procedural guidance relevant to the current task.

## 1. State the method contract before coding

Record the task, metric and direction, intended user/data domain, allowed
changes, data access and compute budget. Formulate a falsifiable hypothesis:
what mechanism should improve what failure mode, compared with which baselines?
Name a rival explanation and the cheapest observation that would refute the
claim. Use credible, cited prior methods instead of relying on an agent's
novelty assertion; novelty ultimately requires a literature review.

Specify the protocol once: dataset release and sample/group identities,
preprocessing, split/adaptation boundaries, loss/scorer mathematics, fixed
training and evaluation settings, seeds, repetition/confirmation rules and
admission criteria. Explicitly separate known details from assumptions.

## 2. Use the smallest existing NexuML contracts

Inspect the selected installed runtime, the user's package, registered
components and existing scenarios before creating new code.

- Ordinary compatible tensor operations use importable PyTorch factories
  (`nn_module`) with explicit `LayerSpec` key routing.
- NexuML-specific behavior uses typed, immutable definitions with private
  runtime objects, correct TensorDict inputs/outputs and train-only state.
- Put reusable models, losses, datasets, fitting and evaluators in the
  user-owned importable library. Put composition and experimental choices in
  ordinary scenarios and portable `ResolvedConfig` snapshots.
- Keep reference method, new variant and ablations under separate identities.
  Do not edit the framework core or invent another training loop, registry or
  research scheduler to accommodate the method.

Preserve the existing package layout; do not copy the demo package wholesale.

## 3. Gate correctness before spending training budget

Use a small fixed input/output example and verify each applicable contract:

1. Parameter bounds, actual dataset membership, labels, TensorDict keys/shapes
   and layer ordering.
2. Numerically correct equations and reductions, finite outputs, expected
   gradients and optimizer update behavior. Compare against a tiny independent
   reference when possible.
3. Train-only fitting, state save/restore, score direction, exact metric
   aggregation and exclusion of final-test sample IDs.
4. Fresh-process discovery, portable configuration round-trip, compiler/dummy
   forward, then one authorized small native-lifecycle smoke.

A check of finite loss or successful compilation does not prove that a new
algorithm is correct, effective, or novel. Investigate detached gradients,
ignored search parameters, wrong metrics and split leakage before large runs.

## 4. Compare fairly and pursue hypotheses

Run a verified baseline under the declared protocol first. Each experiment
should record a parent, mechanism, specific changes, expected discriminator,
cohort/scorer, cost, acceptance rule and evidence paths.

Start with an ablation that can isolate the contribution. Keep data processing,
training selection and resource budgets comparable, or explicitly report
differences. Equal epoch counts may hide different update counts and compute
costs. Vary coupled factors together only when scientifically justified, without
claiming to attribute the joint effect to one factor.

Selection and iteration use research/validation data only. Hold the final test
behind a frozen candidate/protocol and separately authorized evaluation.
Treat promising single-seed results as provisional; confirm under a predeclared
repetition budget. Preserve negative attempts, contradictions and uncertainty.

Use native NexuML train/tune and an existing project ledger. The autoresearch
skill can schedule *which* authorized experiment to run next, but should not
replace NexuML's execution lifecycle. Resume recorded jobs and budgets rather
than restart a study because the agent context changed.

## 5. Promote only verified reusable work

Deliver only what the project needs:

- Importable and discoverable components with stable identities/dependencies.
- Ordinary reproducible scenarios/configs for baseline and accepted variants.
- Focused checks for the method's actual numerical/data/runtime risks.
- An evidence ledger linking source/data/protocol identities, exact commands,
  raw metrics, seeds, resources, successful and failed attempts.
- A report separating implemented, smoke-verified, numerically verified,
  reproducibly improved and externally validated claims.

Keep experimental/failed artifacts out of the stable library API; do not
overwrite historical baselines. If checks reveal a framework defect, provide a
minimal reproducer and file it separately instead of hiding a core patch inside
the user library.
