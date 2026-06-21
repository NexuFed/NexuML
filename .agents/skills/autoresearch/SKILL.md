---
name: autoresearch
description: >-
  Run iterative AI-assisted experiments for NexuModular using trusted Python
  scenario files. Use when asked to perform autonomous research, improve a
  metric, iterate on a scenario, or run follow-up experiments.
---

# Autoresearch for NexuModular

Run a disciplined, evidence-driven experiment loop in the style of Karpathy's
nanogpt research notes: one variable changed per step, clear hypothesis,
honest recording of results, no surprises.

The loop always targets a **test metric** (not validation). When no separate
test set exists, validation substitutes.

---

## Safety and scope

- Scenario files are trusted project code, not sandboxed.
- Start with the narrowest change that tests the hypothesis.
- **Hyperparameter tuning first.** Architecture changes only after tuning
  confirms the baseline is not just under-tuned.
- Adding new library components (`src/nexumodular/library/layers/`) is allowed
  when a gap is clearly identified and the component is self-contained. Report
  it explicitly when you do.
- If core code appears broken, stop and report — do not work around it.
- Never delete prior experiment artifacts.

---

## Repo contract

```text
.experiments/agent/scenarios/exp_NNN_short_name.py   ← gitignored scratch
.experiments/agent/exp_NNN_short_name/        ← gitignored artifacts
.experiments/agent/research_log.md            ← gitignored log
src/nexumodular/library/                      ← versioned library code
```

`.experiments/agent/scenarios/` lives under the already-gitignored `.experiments/`. It is research scratch.
When an experiment wins, note it as a candidate for promotion to
`src/nexumodular/library/scenarios/` and tell the user.

---

## Workflow

### 1. Parse the invocation

The user invokes `/autoresearch <scenario-name-or-file>` with an optional
objective. Extract:

| Field | Default |
|---|---|
| scenario | required — built-in name or file path |
| metric | from `TUNING_SPEC.metric_key` in scenario, else `val/loss` |
| direction | from `TUNING_SPEC.directions`, else `minimize` |
| budget | 3 experiments unless specified |
| epoch budget | cheapest first: 3 → 10 → 25 → longer only if improving |
| allowed changes | all (tuning, architecture, new layers) unless restricted |

If `<scenario-name>` maps to a built-in, import it. If it is a file path,
load it. If ambiguous, ask once.

### 2. Read prior evidence

Check in order:

1. `.experiments/agent/research_log.md`
2. `.experiments/agent/*/` artifact dirs
3. `.experiments/agent/scenarios/*.py`
4. The scenario source in `src/nexumodular/library/scenarios/`

Summarise: current best, failed attempts, open questions.

### 3. Plan the next experiment

Write a one-paragraph plan before touching any file:

```
Experiment: exp_NNN_name
Parent:     exp_MMM_name or None
Hypothesis: one clear claim
Change:     exactly what is different from parent
Metric:     key + direction
Budget:     N epochs [+ M tuning trials if tuning]
Stop rule:  keep if metric X, reject if metric Y, investigate if Z
```

**Progression heuristic:**

```
Phase 1 — Sanity (1–3 epochs)
  Confirm the pipeline runs end-to-end. Fix any errors before proceeding.

Phase 2 — Tuning (use nexumodular tune)
  Search lr, batch_size, weight_decay. Use the scenario's TUNING_SPEC.
  Accept the best config before architecture changes.

Phase 3 — Architecture (one change at a time)
  width → depth → bottleneck → skip connections → normalization → ...
  Each change trained with the tuned config from Phase 2.

Phase 4 — New components (if Phase 3 plateaus)
  Identify the gap (e.g. "no per-frequency attention").
  Write a minimal new layer in src/nexumodular/library/layers/.
  Test it in one experiment. If it helps, note it for graduation.
```

### 4. Write the scenario file

Use the smallest possible scenario file. Import an existing factory and patch
only what changes:

```python
HYPOTHESIS = "Wider latent reduces compression ratio and improves AUC."
PARENT = "exp_002_baseline_tuned"
TAGS = ["architecture"]

from nexumodular.core.types import TuningSpec
from nexumodular.library.scenarios.composed.my_scenario import my_scenario

TUNING_SPEC = TuningSpec(
    metric_key="test/anomaly_detector_0/auc",
    directions=["maximize"],
    n_trials=15,
)

def scenario():
    s = my_scenario(latent_dim=256, max_epochs=25)
    s.name = "exp_003_wider_latent"
    return s
```

Optionally expose `SEARCH_SPACE` for tuning:

```python
SEARCH_SPACE = {
    "training.lr":         {"type": "float",       "low": 1e-5, "high": 1e-2, "log": True},
    "training.batch_size": {"type": "categorical",  "choices": [8, 16, 32]},
}
```

### 5. Run

```bash
# Train
nexumodular train \
  --scenario-file .experiments/agent/scenarios/exp_NNN_name.py \
  --artifact-dir .experiments/agent/exp_NNN_name \
  --max-epochs N

# Tune (metric and direction from TUNING_SPEC; --metric/--direction are overrides)
nexumodular tune \
  --scenario-file .experiments/agent/scenarios/exp_NNN_name.py \
  --artifact-dir .experiments/agent/exp_NNN_name \
  --n-trials M
```

On failure: read the error, fix scenario-file mistakes only, retry once. If
it still fails and the root cause is library or environment code, stop and
report.

### 6. Extract and record results

After each run inspect artifacts and output for:

- final train/val/test metrics
- best trial and params (if tuned)
- resolved config, run duration, warnings

Then append to `.experiments/agent/research_log.md`:

```markdown
## exp_NNN_name

- Parent: ...
- Hypothesis: ...
- Command: `...`
- Result: success / failure
- Metrics: `val/loss=...`, `test/metric=...`
- Decision: keep / reject / investigate
- Next: ...
```

### 7. Iterate

Continue when the next step follows from evidence.

Stop when:
- budget exhausted
- metric has not improved for 2 consecutive experiments
- failures indicate a library or environment problem
- user-requested target is reached

Final answer:

```
Best: exp_NNN_name  (test/metric=X)
Tried: sanity → tuning → wider_latent → deeper_channels
Next: try skip connections OR graduate exp_NNN to library
Artifacts: .experiments/agent/
```

---

## Adding a new library layer

When no existing layer fits:

1. Write it in `src/nexumodular/library/layers/<category>/my_layer.py`
2. Register it: `@register_layer("my_layer")`
3. Export from `src/nexumodular/library/layers/<category>/__init__.py`
4. Use it in the experiment via `LayerSpec(type_key="my_layer", ...)`
5. Record in the research log: `New component: my_layer (src/...)`

Keep new layers minimal. If the experiment fails, the layer can be removed
without ceremony.

---

## Quality checklist

Before finishing:

- each scenario file imports successfully or was run by CLI
- each experiment has a unique `spec.name`
- every run has a matching `--artifact-dir`
- `research_log.md` is up to date
- new library components are noted explicitly
- failures are recorded, not hidden

---

## Workflow Optimization

- **Delegate execution to specialists**: Orchestrator should plan, delegate, and synthesize. Use @explorer for read-only analysis (artifact inspection, metric recomputation, code archaeology). Use @fixer for bounded file writes (ledger updates, scenario files, skill updates). Do not run long experiments or do detailed analysis as orchestrator.
- **Run experiments in parallel when possible**: If multiple hypotheses can be tested independently (e.g., tuning vs architecture), run them in parallel to save time. Use the research log to track which experiments are running and their results. Use cheap subagents for running one experiment locally and the others on a remote kubernetes cluster.
- **Use cached features for offline scoring**: When features are expensive to extract (e.g., AST forward pass), cache them once and recompute metrics offline. Avoid re-running full framework pipelines for metric-only comparisons.
- **Verify metric definitions before comparing**: Scratch scripts may use non-official metric variants (e.g., same-domain anomalies for source/target AUC). Always recompute official metrics from raw scores before claiming parity or improvement.

## Diagnostic Patterns

- **Scenario-local monkeypatches for diagnostics**: When testing a hypothesis (e.g., "does shuffle affect GMM init?"), use scenario-local monkeypatches first. If confirmed, promote to core patch via @fixer.
- **Feature parity checks**: Before debugging metrics, verify feature extraction parity on a small batch (e.g., first 32 train files). If features match exactly, the issue is in fitting/scoring, not extraction.
- **Batch size sensitivity**: Test whether batch size affects results (e.g., padding in feature extractor). If negligible, focus on other factors.

## Ledger Discipline

- **Update ledgers after each experiment**: Append to research_log.md, leaderboard.md, experiments.jsonl, candidate_registry.md immediately after result. Do not batch updates.
- **Correct optimistic metrics**: If a scratch script reports higher metrics than framework, recompute both with official scorer before declaring winner.
- **Mark monkeypatch vs clean**: In candidate registry, distinguish scenario-local hacks from core patches. Only clean patches are lockable.

## Cost Control

- **Use cheap models for long-horizon loops**: Orchestrator should be cheap (e.g., Qwen3.7-class). Specialists can be more expensive for bounded tasks. Models running experiments should be as cheap as possible and reporting back.
- **Compress handoffs**: Specialists return paths, line refs, and short summaries. Do not paste full logs/artifacts into orchestrator context.
- **Batch artifact analysis**: One explorer analyzes multiple artifacts rather than one agent per run.
