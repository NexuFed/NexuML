# Tune with Optuna

Install the optional integration:

```bash
uv pip install "nexuml[tuning]"
```

`nexuml tune` accepts either a registered scenario or a trusted Python scenario file. It does not tune from resolved YAML because structural/Python search-space logic is not represented by that persistence format.

## Registered scenario

```bash
nexuml tune my-scenario --n-trials 30
```

`ScenarioSpec.tuning` can provide defaults such as trial count, direction, metric key, storage, and pruning policy:

```python
from nexuml.core.types import TuningSpec

TuningSpec(
    n_trials=30,
    directions=["minimize"],
    metric_key="val/loss",
    storage="sqlite:///.experiments/optuna/study.db",
    prune=False,
)
```

CLI options can override those run-level values. Use the generated [CLI reference](../reference/cli.md) for exact flags.

## Trusted file and custom search space

```bash
nexuml tune --scenario-file experiment.py --n-trials 20
```

A trusted file can expose `SEARCH_SPACE`, `TUNING_SPEC`, and a structural `build(**params)` factory. Conditional and derived search-space entries are also Python-only.

Use [Tuning file reference](../reference/tuning-file.md) for the exact format rather than duplicating that schema here.

## Metric selection

The optimized metric must actually be emitted by the run. Typical pipeline validation metrics use keys such as `val/loss`.

If the objective comes from a post-training evaluation algorithm, configure `evaluation.test_result_metrics` so the selected evaluation scalar is mirrored into the test result surface consumed by the tuning workflow.

## Tracking

When MLflow is configured, NexuML can record study/trial runs through the tracking integration. MLflow itself is optional; install/configure it separately from Optuna.

## See also

- [Trusted scenario files](scenario-file.md)
- [Tuning file reference](../reference/tuning-file.md)
- [Evaluation](evaluate.md)
- [Tracking](tracking.md)
