# Run scenarios

NexuML executes a scenario from one of three sources. This page is the source of truth for how each command accepts scenarios.

## Scenario sources

| Source | `resolve` | `build` | `train` | `tune` | `export` |
|---|---|---|---|---|---|
| Registered scenario name | `nexuml resolve <name>` | — | `nexuml train <name>` | `nexuml tune <name>` | `nexuml export <name>` |
| Resolved YAML config (`-c`) | — | `nexuml build <path>` | `nexuml train -c <path>` | Not supported | — |
| Trusted Python file (`--scenario-file`) | — | — | `nexuml train --scenario-file <path>` | `nexuml tune --scenario-file <path>` | — |

Only one source may be provided per command.

## 1. Registered scenario name

Scenarios decorated with `@scenario` and discovered from `nexuml_library`, entry-point packages, or local library roots are available by name.

```bash
nexuml registry list scenarios
nexuml resolve my-scenario
nexuml train my-scenario --max-epochs 10
nexuml tune my-scenario --n-trials 20
nexuml export my-scenario
```

## 2. Resolved YAML config (`--config` / `-c`)

`nexuml resolve` writes a reproducible YAML config. That config can be passed to `train` or `build` with `--config` / `-c`:

```bash
nexuml resolve my-scenario -o configs/my-scenario.yaml
nexuml build configs/my-scenario.yaml
nexuml train -c configs/my-scenario.yaml --max-epochs 10
```

`build` requires a resolved YAML path as its positional argument.

## 3. Trusted Python scenario file (`--scenario-file`)

A plain Python file that defines `scenario() -> ScenarioSpec` can be executed directly:

```bash
nexuml train --scenario-file my_experiment.py
nexuml tune --scenario-file my_experiment.py --n-trials 20
```

Use `--artifact-dir` to save a provenance snapshot:

```bash
nexuml train --scenario-file my_experiment.py --artifact-dir ./artifacts/exp-001/
```

!!! warning "Trusted execution"
    Scenario files are executed with `exec()`. Only load files from trusted sources.

## `tune` limitations

`nexuml tune` supports a registered scenario name **or** `--scenario-file`, but **not** resolved YAML via `--config` / `-c`. This is because tuning relies on Python-side `SEARCH_SPACE`, `TUNING_SPEC`, or `build(**params)` definitions that are not part of the resolved YAML format.

## Common command matrix

```bash
# Resolve a registered scenario to YAML
nexuml resolve synthetic-linear-ae-reconstruction

# Inspect the resolved config
nexuml build configs/synthetic-linear-ae-reconstruction.yaml

# Train by name
nexuml train synthetic-linear-ae-reconstruction --max-epochs 5

# Train from YAML
nexuml train -c configs/synthetic-linear-ae-reconstruction.yaml --max-epochs 5

# Train from a trusted Python file
nexuml train --scenario-file my_experiment.py

# Tune a registered scenario
nexuml tune synthetic-linear-ae-reconstruction --n-trials 10

# Tune from a trusted Python file
nexuml tune --scenario-file my_experiment.py --n-trials 10
```

## See also

- [Define a scenario](define-scenario.md)
- [Trusted scenario files](scenario-file.md)
- [Tuning file reference](../reference/tuning-file.md)
- [Optuna tuning](tune.md)
- [ScenarioSpec reference](../reference/scenario-spec.md)
