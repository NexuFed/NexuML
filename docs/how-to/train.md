# Train a model

## Basic training

```bash
nexuml train <scenario-name>
```

## Options

| Option | Description |
|---|---|
| `--config` / `-c PATH` | Config YAML path (alternative to scenario name) |
| `--scenario-file PATH` | Trusted Python file defining `scenario() -> ScenarioSpec` |
| `--artifact-dir PATH` | Directory for provenance snapshots (used with `--scenario-file`) |
| `--max-epochs N` | Override `training.max_epochs` |
| `--trainer-checkpoint PATH` | Lightning checkpoint to resume from |
| `--override` / `-O key=value` | Override any spec field (repeatable) |

## Examples

```bash
# Train by scenario name
nexuml train my-scenario

# From a YAML config
nexuml train -c configs/my-scenario.yaml

# Override fields
nexuml train my-scenario -O training.max_epochs=20 -O training.lr=1e-4

# Resume from a checkpoint
nexuml train my-scenario --trainer-checkpoint .experiments/checkpoints/my-scenario/last.ckpt

# Trusted Python file
nexuml train --scenario-file my_experiment.py
```

## Outputs

| Artifact | Default location |
|---|---|
| Checkpoints | `.experiments/checkpoints/<scenario>/` |
| TensorBoard | `.experiments/tensorboard/` |
| MLflow | `.experiments/mlflow.db` |
| Mermaid diagram | `.experiments/diagrams/<scenario>.md` |

## Automatic batch size

Set `training.batch_size` to an `AutoBatchSizeSpec` to let NexuML probe the largest batch that fits in GPU memory. See [Automatic batch size](auto-batch-size.md).

## See also

- [CLI lifecycle](cli-lifecycle.md) — all commands
- [Checkpoints](checkpoints.md) — resume and selective loading
- [Tracking and logging](tracking.md)
- [Automatic batch size](auto-batch-size.md)
- [Export a model package](export.md)
