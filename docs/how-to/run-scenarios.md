# Run scenarios

NexuML accepts different scenario sources at different boundaries. Use this page to choose the source; use the generated [CLI reference](../reference/cli.md) for exact flags.

## Supported sources

| Command | Registered scenario | Resolved YAML | Trusted Python file |
| --- | --- | --- | --- |
| `resolve` | yes | — | — |
| `build` | — | yes | — |
| `train` | yes | yes | yes |
| `export-dataset` | yes | yes | — |
| `tune` | yes | — | yes |
| `export` | yes | — | — |
| `smoke` | yes | — | — |

Only one scenario source may be supplied to a command.

## Registered scenario

Libraries expose scenario recipes with `@scenario`:

```bash
nexuml registry list scenarios
nexuml resolve my-scenario
nexuml train my-scenario
```

Use this for stable, reusable experiment recipes.

## Resolved YAML

`resolve` persists the scenario's typed definitions and surrounding specs:

```bash
nexuml resolve my-scenario -o configs/my-scenario.yaml
nexuml build configs/my-scenario.yaml
nexuml train -c configs/my-scenario.yaml
```

Use YAML when you want an inspectable/versionable frozen configuration rather than re-evaluating the scenario recipe.

## Trusted Python file

For local experiments, a file can expose `scenario() -> ScenarioSpec`:

```bash
nexuml train --scenario-file experiment.py
nexuml tune --scenario-file experiment.py --n-trials 20
```

Use `--artifact-dir` when you want NexuML to snapshot the trusted file and provenance alongside the run.

!!! warning "Trusted execution"
    Scenario files are executed as Python code. Only run files you trust.

## Checkpoint-only resume

`train` also accepts a Lightning trainer checkpoint without a separate scenario source. NexuML can recover the persisted scenario information from a compatible checkpoint. See [Checkpoints](checkpoints.md).

## See also

- [Define a scenario](define-scenario.md)
- [Trusted scenario files](scenario-file.md)
- [CLI reference](../reference/cli.md)
