# CLI lifecycle

NexuML exposes a Typer CLI (`nexuml`) with ten top-level commands covering the full pipeline lifecycle.

## Prerequisites

- NexuML installed (`uv sync`)
- Optional: `NEXUML_DATA_ROOT` and `NEXUML_LOGS_ROOT` set (see [Environment roots](../reference/environment.md))

## Commands at a glance

| Command | Role |
|---|---|
| `resolve` | Compile a Python scenario to a reproducible YAML config |
| `build` | Validate a YAML config and export a Mermaid diagram |
| `train` | Train a scenario or config via Lightning |
| `export-dataset` | Export a dataset view to disk |
| `export` | Export a trained pipeline package |
| `smoke` | Run the full resolve → build → train → export → reload → infer cycle |
| `tune` | Run Optuna hyperparameter search |
| `registry` | Inspect discovered layers, data sources, scenarios, and eval algorithms |
| `backend` | List available backend implementations |
| `library` | Manage local library roots |

## Normal workflow

### 1. resolve — compile scenario to YAML

```bash
nexuml resolve <scenario-name>
# Writes: configs/<scenario-name>.yaml

# Custom output path
nexuml resolve <scenario-name> -o my-config.yaml
```

`resolve` runs the compiler: resolves layer keys, validates `keys_in`/`keys_out` contracts, and writes the canonical config.

### 2. build — validate and diagram

```bash
nexuml build configs/<scenario-name>.yaml
```

Prints layer order, tensor shapes, and parameter counts to stdout. If `logging.diagram.enabled=true`, also writes a Mermaid `.md` file to `logging.diagram.output_dir`.

### 3. train — run Lightning training

```bash
nexuml train <scenario-name>

# From a config file
nexuml train -c configs/my-scenario.yaml

# Field overrides (repeatable)
nexuml train my-scenario -O training.max_epochs=20 -O training.lr=1e-4

# Resume from a Lightning checkpoint
nexuml train my-scenario --trainer-checkpoint .experiments/checkpoints/last.ckpt

# Trusted Python scenario file
nexuml train --scenario-file my_experiment.py --artifact-dir ./artifacts/exp-001/
```

Full `train` options:

| Option | Description |
|---|---|
| `SCENARIO_NAME` | Registered scenario name |
| `--config` / `-c PATH` | Config YAML path (alternative to scenario name) |
| `--scenario-file PATH` | Trusted Python file defining `scenario() -> ScenarioSpec` |
| `--artifact-dir PATH` | Directory for provenance snapshots (used with `--scenario-file`) |
| `--max-epochs N` | Override `training.max_epochs` |
| `--trainer-checkpoint PATH` | Lightning checkpoint to resume from |
| `--override` / `-O key=value` | Override any spec field (repeatable) |

### 4. export — save a portable package

```bash
nexuml export <scenario-name>

# Specify a checkpoint
nexuml export my-scenario --checkpoint .experiments/checkpoints/best.ckpt

# Custom output directory
nexuml export my-scenario -o packages/my-scenario/
```

See [Model export and reload](export.md) for package layout and Python reload API.

### 5. smoke — end-to-end sanity check

```bash
nexuml smoke                        # uses synthetic-linear-ae-reconstruction by default
nexuml smoke my-scenario --max-epochs 2
```

Runs resolve → build → train → export → reload → infer in one shot. Use this to verify a scenario works before a full training run.

## Tuning

```bash
nexuml tune my-scenario --n-trials 30
```

See [Optuna tuning](tune.md) for the full guide.

## Dataset export

```bash
nexuml export-dataset my-scenario -o ./output/ --backend numpy
```

See [Dataset export](export-dataset.md).

## Registry and backend inspection

```bash
# List what is registered in this environment
nexuml registry list layers
nexuml registry list scenarios
nexuml registry list data
nexuml registry list eval

# List all backend implementations
nexuml backend list
```

See [Registry inspection](../reference/registry.md) and [Backends](../reference/backends.md).

## Expected output

After `nexuml train`, the following are written (paths configurable via `LoggingSpec`):

| Artifact | Default location |
|---|---|
| Lightning checkpoints | `.experiments/checkpoints/<scenario>/` |
| TensorBoard logs | `.experiments/tensorboard/` |
| MLflow runs | `.experiments/mlflow.db` |
| DVCLive | `.experiments/dvclive/` |
| Mermaid diagram | `.experiments/diagrams/<scenario>.md` |

## Implementation map

- `src/nexuml/cli/main.py` — all Typer command definitions
- `src/nexuml/core/types.py` — `ScenarioSpec`, `TrainingSpec`, `TuningSpec`
- `src/nexuml/core/scenario_loader.py` — `--scenario-file` loading
- `src/nexuml/training/lightning.py` — `NexuSession` Lightning wrapper
- `src/nexuml/core/export.py` — `export_package`, `load_package`

## See also

- [Trusted scenario files](scenario-file.md)
- [Optuna tuning](tune.md)
- [Dataset export](export-dataset.md)
- [Model export and reload](export.md)
