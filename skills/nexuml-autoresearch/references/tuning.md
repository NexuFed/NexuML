# Native tuning boundaries

Inspect the selected installed CLI/signatures. These examples target typed
NexuML 0.2 and do not require a framework checkout. Native tuning needs the
`tuning` extra/Optuna in that same environment.

## Scenario file

```python
from nexuml.core.types import TuningSpec
from my_library.scenarios.baseline import baseline

HYPOTHESIS = "A smaller learning rate improves held-out research loss."
PARENT = "baseline-001"
TAGS = ["learning-rate"]

def scenario():
    spec = baseline()
    spec.name = "research-lr-001"
    # Appropriate only when the dataset uses these random-split fractions.
    # Fixed metadata assignments need explicit final-test exclusion as well.
    spec.data.train_split = 0.8
    spec.data.val_split = 0.2
    spec.data.test_split = 0.0
    spec.training.max_epochs = 3
    return spec

SEARCH_SPACE = {
    "training.lr": {"type": "float", "low": 1e-5, "high": 1e-3, "log": True},
}
TUNING_SPEC = TuningSpec(
    n_trials=3,
    metric_key="val/loss",
    directions=["minimize"],
    storage="experiments/research-lr-001/study.log",
)
```

The metric must really be emitted. Inspect actual loader membership to verify
that the final-test cohort is absent. Native tuning calls normal training;
there is no automatic final-test isolation just from using a validation metric.

```sh
"$PYTHON" -m nexuml.cli.main tune --scenario-file experiments/lr.py --artifact-dir experiments/research-lr-001/provenance --n-trials 3 --metric val/loss --direction minimize
```

There is no `tune --max-epochs` option in this interface: set epochs in the
factory. `--override training.max_epochs=...` exists, but applies to the initial
scenario; a trial rebuilt by `build` can replace that override. Keep fixed epoch,
cohort and resource constraints in every factory-returned scenario. Do not invent
CLI options from the training command. Native tuning accepts a name or a scenario
file, not a YAML `--config` in this version.

## Structural parameters

Export `build(**params) -> ScenarioSpec` alongside `scenario()`. Non-dotted
parameter names are passed to the factory; simple dotted scalar paths are
applied as overrides. For example:

```python
SEARCH_SPACE = {"hidden_dim": {"type": "int", "low": 8, "high": 32}}

def build(hidden_dim=16):
    return baseline(hidden_dim=hidden_dim)  # user's actual factory signature

def scenario():
    return build()
```

Categorical entries containing `choices` are treated as factory parameters by
this implementation, including dotted names. Dotted factory names become nested
dicts: `training.batch_size` becomes `build(training={"batch_size": value})`,
not `build(**{"training.batch_size": value})`. A matching signature is needed.
Without a factory these parameters are currently sampled but ignored; a study's
recorded params therefore do not prove that the model/configuration changed.
Prefer simple structural names and inspect the actual scenario for every trial:

```python
SEARCH_SPACE = {
    "batch_size": {"type": "categorical", "choices": [8, 16]},
    "hidden_dim": {"type": "int", "low": 8, "high": 32},
}

def build(batch_size=8, hidden_dim=16):
    spec = baseline(hidden_dim=hidden_dim)
    spec.name = "research-structure-001"
    spec.training.batch_size = batch_size
    spec.training.max_epochs = 3
    spec.data.train_split, spec.data.val_split, spec.data.test_split = 0.8, 0.2, 0.0
    return spec  # also exclude fixed final-test IDs in the user's data factory

def scenario():
    return build()
```

Immutable component fields also require rebuilding definitions/scenarios, not
assigning sampled values into their fields. Keep complex structural names simple
until the installed behavior is verified. Conditional/derived entries are
Python-only; do not assume arbitrary string arithmetic expressions are supported.

## Evidence and reproducibility limits

The study uses persistent storage and can resume an existing scenario-named
study. Use a unique scenario/storage identity for a new protocol; deliberately
document continuation rather than mixing earlier results silently. Confirm
budget accounting when continuing: the current `study.optimize(n_trials=N)`
adds N trial attempts; it does not cap the total stored trials at N. Storage is
resolved as a filesystem path with a `.db` suffix, not an arbitrary Optuna URL.

A scenario-file `--artifact-dir` snapshot records the source, a resolved initial
scenario, command context and best-trial information after tuning. It is not
necessarily the winning trial's fully resolved configuration. Reconstruct the
winner from its actual sampled params/factory and preserve that configuration
and the study/trial evidence separately.

Record actual training/sampler seeds and repeat conditions. Native tuning does
not guarantee a seeded search simply because a dataset has a seed field.
Do not advertise pruning, multi-objective selection or reproducibility features
without checking that the installed objective/callback path actually implements
them. Use explicit variants when they provide clearer controlled evidence.

These details were checked against the CLI `tune` function and
`nexuml.tuning.optuna_tuner` (`_resolve_search_space`, `_set_param_dict`,
`build_objective`, `tune`). Inspect the installed versions; reference examples
can lag implementation. Training-free checks are in `skills/tests/check_runtime.py`
in the skill-kit repository; they are verification evidence, not a required runtime
dependency for users installing just this skill.
