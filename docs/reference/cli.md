# CLI reference

This page is generated from the current Typer application, so it is the source of truth for command names, arguments, and options. Task guides intentionally do not copy every flag.

The CLI covers scenario resolution/build/training, dataset export, model export, smoke tests, tuning, registry inspection, backend inspection, and local-library management. `evaluate <model-or-policy-package>` evaluates a trusted model on its `NexuDataset` test split or a complete policy in its environment, without training. Both `train` and `evaluate` accept trusted Python `--scenario-file` definitions; evaluation retains the exported model's computation and weights. See [Evaluate a model](../how-to/evaluate.md), [Reinforcement learning](../how-to/reinforcement-learning.md) for policy/media settings, and [Checkpoints](../how-to/checkpoints.md) for the separate training-resume path.

Execution commands share `nexuml.execution` and need no API extra:

```sh
nexuml backend catalog
nexuml backend discover --config scenario.yaml --timeout 10
nexuml backend preflight --config scenario.yaml
nexuml train --config scenario.yaml
nexuml backend inspect reference.json
```

Catalog does not probe targets; discover/preflight do not submit training. Native jobs
require an infrastructure template and explicit config handoff. See
[execution modes](../how-to/training-backends/index.md) for current registered syntax,
unsupported distributed phases and honest observation/cancellation limits.

::: mkdocs-click
    :module: nexuml.cli.main
    :command: click_app
    :prog_name: nexuml
    :depth: 2
