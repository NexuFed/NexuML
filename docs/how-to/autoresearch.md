# Autoresearch

Autoresearch lets a Claude agent run an iterative experiment loop in the style of Karpathy's research notes: one variable changed per step, clear hypothesis, honest recording of results. The agent handles hyperparameter tuning, architecture search, and — when warranted — writing new library components.

## Start

Open Claude Code in this repository and run:

```
/autoresearch <scenario-name-or-file>
```

Examples:

```
/autoresearch synthetic-linear-ae-reconstruction

/autoresearch dcase-conv-ae  Maximize test/anomaly_detector_0/auc on ToyCar.
                             Budget: 5 experiments, 25 epochs each.

/autoresearch .experiments/agent/scenarios/exp_003_baseline.py
```

The agent will:

1. Read `.experiments/agent/research_log.md` for prior results
2. Run a sanity check (few epochs) if no baseline exists
3. Tune hyperparameters (`nexuml tune`) before changing architecture
4. Iterate architecture one variable at a time, recording every result
5. Add new library layers if no existing component fits a clearly identified gap

## Agent-authored scenario files

External agents write trusted Python scenario or tuning files, invoke existing commands, read metrics/artifacts, and iterate. NexuML does not sandbox these files; treat them as project code.

```bash
nexuml train \
  --scenario-file .experiments/agent/scenarios/exp_001_baseline.py \
  --artifact-dir .experiments/agent/exp_001

nexuml tune \
  --scenario-file .experiments/agent/scenarios/exp_002_tuning.py \
  --artifact-dir .experiments/agent/exp_002
```

Scenario files expose `scenario() -> ScenarioSpec` plus optional metadata:

```python
HYPOTHESIS = "Try a wider projection head."
PARENT = "exp_001_baseline"
TAGS = ["architecture"]

def scenario():
    ...
```

Tuning files may also export `SEARCH_SPACE` or `search_space()` and `TUNING_SPEC` or `tuning_spec()`. Without an explicit search space, `tune` uses the built-in default.

`--artifact-dir` snapshots the source file, resolved scenario YAML, metadata, git state, command context, and tune summaries when available.

## Artifacts

| Path | Contents | Tracked by git |
|------|----------|----------------|
| `.experiments/agent/scenarios/` | Agent scratch scenario files | No |
| `.experiments/agent/` | Run artifacts, resolved configs, metrics | No |
| `.experiments/agent/research_log.md` | Full experiment history | No |
| `library/src/nexuml_library/` | Proven, promoted code | Yes |

## Graduating a winner

When an experiment reaches its target, promote it:

1. Copy the winning factory into `library/src/nexuml_library/scenarios/<category>/` (for example, `vision/`, `asd/`, or `tune/`)
2. Give it a stable name (not `exp_NNN_`)
3. New layers written during the run live permanently in `library/src/nexuml_library/layers/`

## Export a trained model

```bash
nexuml export <scenario> --checkpoint logs/<scenario>/checkpoints/best.ckpt
```

See [Export a model package](export.md) for the full export and reload workflow.
