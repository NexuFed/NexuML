# Train a model

## Train a registered scenario

```bash
nexuml train my-scenario
```

## Train resolved YAML

```bash
nexuml train -c configs/my-scenario.yaml
```

## Train a trusted local experiment

```bash
nexuml train --scenario-file experiment.py --artifact-dir artifacts/exp-001
```

See [Run scenarios](run-scenarios.md) for when to use each source.

## Common overrides

```bash
nexuml train my-scenario --max-epochs 20
nexuml train my-scenario -O training.lr=0.0001 -O training.precision=bf16-mixed
```

Overrides change the scenario used for that command. For reusable experiment changes, prefer expressing the value in the Python scenario and resolving a new configuration.

## Resume the Lightning trainer

```bash
nexuml train my-scenario --trainer-checkpoint PATH
```

This is a **full Lightning resume**: model, optimizer/scheduler state, epoch/global step, and compatible callback state come from the trainer checkpoint.

Do not use `CheckpointLoadSpec` for the same purpose. `CheckpointLoadSpec` is the separate selective/pretrained-weight workflow described in [Checkpoints](checkpoints.md).

## Execution placement

The command stays the same when the scenario selects Ray execution:

```bash
nexuml train my-scenario
```

`ScenarioSpec.execution` controls whether the canonical session runs locally or through Ray. See [Execution modes](training-backends/index.md).

## Outputs

NexuML does not require one hard-coded output directory layout for every project. Outputs are driven by the scenario:

- checkpoint callbacks decide checkpoint creation and location;
- `LoggingSpec` controls TensorBoard/MLflow/DVCLive/diagram outputs;
- `ExportSpec(kind="train_package")` can package the live trained model after a local run;
- `--artifact-dir` stores provenance for trusted scenario-file runs.

Use [Environment roots](../reference/environment.md) for default root resolution and the relevant guide for each output type.

## Exact options

The CLI is generated from the implementation. Use [CLI reference](../reference/cli.md) instead of copied option tables.
