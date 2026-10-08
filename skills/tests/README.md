# Skill evaluation boundaries

Run `python skills/tests/check_skills.py` for install-payload/frontmatter/link
checks. With compatible installed NexuML, run `python skills/tests/check_runtime.py`
for routing, fitted-state and native tuning-contract checks.

`evals/nexuml*.json` holds read-only response cases; `evals/workflows.json` holds
bounded external-project execution cases; `evals/routing.json` holds near misses.
These files and graders are not part
of the skill installation. Keep generated artifacts in ignored `*-workspace/`
directories, preserving previous versions and failures.

## Paired workflow protocol

- Snapshot the previous skill directories before edits. Give revised and previous
  versions identical copies of the fixture and installed NexuML distributions.
  Keep model/agent/tools/permission policy/budgets fixed and record their identities.
- Install the four skills in the host's normal project-local discovery path.
  Give the task prompt without naming a skill or pointing at its instructions.
  Record actual skill loads separately from execution success. Forced skill loads
  test adherence, not natural selection; keep those results separate.
- Preflight the host-reported tool directory and effective permissions. A child
  process's OS working directory alone may not set its tool location. Allow
  equivalent safe Python flags/environment prefixes consistently across pairs;
  preserve harness-blocked attempts and label any corrected-policy recheck.
- Execute in disposable external projects, not a framework checkout. Pin the
  runtime and dataset/partition identities. Do not expose graders, expected
  outputs or final-test feedback to the agent. Installed public API/source
  inspection and the skill's runnable examples are legitimate inputs.
- Record whether the user package is actually source-linked/editable or detached;
  resolve imported module paths and symlinks rather than infer this from metadata.
- Use existing native NexuML build/train/session operations, not substitute loops.
  Inspect the resulting code, resolved configs, numeric/gradient checks, actual
  loaders and independently recomputed metrics. A claimed pass is not a pass.
- Include failed/interrupted attempts and all consumed resources. Keep missing
  counters unavailable, not zero. Host tool permissions are not an OS sandbox.
- Repeat paired trials when drawing performance conclusions. A single pair is
  diagnostic; synthetic reference parity is not reproduction of a real paper,
  and a one-epoch training run is lifecycle evidence, not convergence.
- Generate the standard skill-creator review with outputs, per-assertion grades,
  benchmark data and limitations. Preserve the user feedback for the next iteration.

## Compare agent performance, not just the skill text

The read-only cases in `evals/` are task prompts and expected evidence, not
automatically executed benchmark scores. For a meaningful skill change, run
matched trials on the **same external NexuML task** in three conditions:

| Arm | Skills available | Purpose |
| --- | --- | --- |
| A | None | Capability without procedural guidance. |
| B | Previous four-skill snapshot | Regression/control against the current release. |
| C | Proposed four-skill snapshot | Measure whether the revision improves outcomes. |

Keep agent/model/tool permissions, installed NexuML and dependencies, user
fixture, compute/data, budget, prompt and grading criteria fixed across arms.
Use fresh identical fixture copies and record model randomness/repetitions; one
paired run is diagnostic, not evidence of general superiority. Include the
`routing.json` near-misses where **no** NexuML skill should activate and
multi-skill tasks where companion guidance is justified.

Measure separately: correct natural skill selection/load, working code and
numerical/data/protocol correctness under independent grading, unnecessary
new components or core modifications, reproducible artifacts, final-test
contamination, attempts/retries, wall/compute cost and observed model token cost
when available. Missing cost or load evidence is *unavailable*, never zero.
For complicated methods report both functional completion and whether the
algorithm's scientific claim was actually supported.

Retain transcripts, grader output and failed attempts outside the agent input.
Publish a small per-case A/B/C table with successes, failures and uncertainty,
not a single unsupported aggregate score. Do not ascribe an outcome to skills
when the runtime, permissions or model changed between arms. The automated
offline checks below cover structural/API contracts; full agent-run comparisons
require the selected agent host and are **not** part of routine CI.

## Executable fixtures and independent artifact checks

Use separate external copies of `nexuml-library/assets/minimal-library` with its
package discoverable in each selected runtime. Provide `fixtures/methods.md` only
for the reference case. For the resumed-research case, copy `fixtures/run_candidate.py`
to the external project and execute its baseline once:

```bash
.venv/bin/python -I run_candidate.py baseline 0.001
```

Clone that completed baseline evidence into both paired contexts. The initial
project ledger must record the protocol, completed baseline and total/spent/
reserved/remaining budget of 3/1/0/2. Keep helper/source/baseline immutable; no
jobs are initially running. The helper permits three run folders total, rejects
existing IDs, saves failures as consumed attempts, and uses native `fit()` and
`validate()` without invoking `test()`. Its limit is a fixture guard, not an OS
sandbox or production research scheduler.

After the agent finishes, independently run the applicable check from its project
with the grader file **outside the agent's available inputs**:

```bash
.venv/bin/python -I /path/to/skills/tests/check_workflow.py component
.venv/bin/python -I /path/to/skills/tests/check_workflow.py paper
.venv/bin/python -I /path/to/skills/tests/check_workflow.py research
```

The research check restores all saved models and recomputes validation scores;
it does not train or evaluate test data. Separately inspect transcript skill loads,
commands, unchanged source/helper/baseline hashes, the agent's executed checks,
summary selection and scoped report/ledger. Numeric artifact checks alone do not
prove natural routing, authorized execution or sound research decisions.

The paper [Scientific Agent Skills](https://doi.org/10.48550/arXiv.2609.00065)
(Kassis et al., 2026; v2) motivates separating description routing, context cost
and task outcomes; it reports no task-level efficacy test.
[OpenScience's native benchmark protocol](https://github.com/synthetic-sciences/openscience/blob/847f7378e0099c52ef44f7b6526835b1bf726125/evals/science-harness/README.md)
is a reference for pinned whole-agent execution and labeled ablations, not a
benchmark score for NexuML. No OpenScience runtime or dependency is required.
