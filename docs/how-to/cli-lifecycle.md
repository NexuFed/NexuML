# CLI lifecycle

The CLI exposes the main NexuML boundaries without requiring a project-specific training script. Use the generated [CLI reference](../reference/cli.md) for the exact flags; this page explains when each command belongs in the workflow.

## Inspect the environment

```bash
nexuml registry list scenarios
nexuml registry list layers
nexuml backend list
```

The registry shows discovered semantic components/scenarios. `backend list` shows runtime/export backends available in the current installation.

## Resolve: Python → persisted configuration

```bash
nexuml resolve my-scenario
```

A registered Python scenario is evaluated and compiled, and its resolved configuration is written to `configs/my-scenario.yaml` by default.

## Build: configuration → runtime pipeline

```bash
nexuml build configs/my-scenario.yaml
```

`build` restores typed definitions from YAML, materializes the pipeline, propagates shapes, and reports the compiled stages. When diagram output is enabled, the configured Mermaid file is written as part of the build.

## Train: run the canonical lifecycle

```bash
nexuml train my-scenario
```

Training can also start from resolved YAML or a trusted scenario file; see [Run scenarios](run-scenarios.md).

The local `NexuSession` lifecycle is:

```text
fit → validate → fit post-train pipeline layers → test
```

If the scenario contains `ExportSpec(kind="train_package")`, local training exports the trained package after the run. This is the simplest way to package the exact live trained model without locating a checkpoint manually.

## Export an existing checkpoint

```bash
nexuml export my-scenario --checkpoint PATH -o packages/my-scenario
```

`--checkpoint` is optional syntactically, but omitting it does **not** search for the latest checkpoint. Supply the checkpoint explicitly when the command should package previously trained weights. See [Model export](export.md).

## Other workflows

```bash
nexuml export-dataset my-scenario -o exported-data
nexuml tune my-scenario --n-trials 20
nexuml smoke my-scenario --max-epochs 1
```

- `export-dataset` persists raw or partially processed dataset views.
- `tune` runs Optuna search from a registered scenario or trusted scenario file.
- `smoke` exercises resolve → build → train → export → reload → inference for a **registered scenario**.

## See also

- [CLI reference](../reference/cli.md)
- [Run scenarios](run-scenarios.md)
- [Train](train.md)
- [Dataset export](export-dataset.md)
- [Model export](export.md)
