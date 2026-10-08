# Autoresearch

Use an agent to compare NexuML configurations, architectures and algorithm
variants under an explicit research budget. Work in your own library/project
with installed NexuML; a framework checkout, API server and MCP are not required.

## Agent skills

The repository's `skills/` directory contains independently installable skills:
`nexuml`, `nexuml-library`, `nexuml-reproduce-paper` and `nexuml-autoresearch`.
Install the selected directories through your agent's supported skill mechanism;
installing the Python package does not install agent skills. See the repository's
`skills/README.md` for installation and validation details.

Open your external library project with the skills available. For example:

```text
Use NexuML to improve experiments/baseline.py.
Select by validation loss under a 2 ms latency limit.
Budget: 5 experiments, 3 epochs each, CPU only, no downloads.
Preserve the baseline and exclude official final-test records from all search runs.
```

For a novel research method rather than a configuration search, first follow
the [algorithm development workflow](../../skills/nexuml-library/references/algorithm-development.md):
formulate a falsifiable mechanism, implement numerical checks and establish fair
baselines/ablations. Do not infer novelty from a validation-score improvement.

The agent should establish the runtime, dataset/partition identities, scorer,
hard constraints and allowed changes; inspect prior evidence; propose a bounded
experiment; then record actual results and keep/reject decisions. Reuse existing
components first. Tuning is one option, not a mandatory prerequisite to every
architecture change. Missing components belong in your user-owned library.
Once the scope and budget are established, the agent continues the loop without
per-experiment approvals. It asks only for consequential boundary changes or
unresolved blockers, and still honors explicit pauses and read-only requests.

## Scenario files and execution

Scenario files expose `scenario() -> ScenarioSpec` and may include `HYPOTHESIS`,
`PARENT` and `TAGS`. They execute trusted Python, not sandboxed configuration.
Keep reusable components/factories in the importable library. Select one runtime
explicitly; a global uv-tool NexuML cannot see dependencies installed only in
your project's `.venv`.

Using that project's interpreter, for example:

```bash
.venv/bin/python -m nexuml.cli.main train \
  --scenario-file experiments/baseline.py \
  --artifact-dir experiments/run-001 --max-epochs 3

.venv/bin/python -m nexuml.cli.main tune \
  --scenario-file experiments/search.py \
  --artifact-dir experiments/search-001 --n-trials 3
```

Check the installed CLI help. Set tuning epoch limits in the scenario/factory;
do not assume training command options also exist on `tune`. Tuning files may
export `SEARCH_SPACE`, `TUNING_SPEC` and a `build(**params)` factory; see the
[tuning-file reference](../reference/tuning-file.md).

## Selection and evidence

Select and debug on validation/research data, never final-test feedback.
Optimizing `val/loss` does not itself prevent the normal lifecycle from executing
a configured test phase. Exclude actual final-test sample identities and inspect
loader membership, especially with fixed metadata assignments. Freeze the
candidate/protocol before separately authorized final-test evaluation.

Use fresh scenario/run identities and preserve source, commands, resolved
configuration, data/environment identities, metrics and failures. Scenario-file
`--artifact-dir` snapshots help, but do not capture all imported code/data or
necessarily the winning trial's resolved configuration. Preserve those separately.

Promote verified reusable components and ordinary recipes into your library,
without overwriting the baseline or discarding failed-run evidence. Report the
best feasible candidate found within budget and its uncertainty; a smoke pass
does not prove convergence or paper parity. Framework failures need a minimal
reproducer/report, not site-packages edits or hidden monkeypatches.

For trained-model export, see [Export a model package](export.md).
