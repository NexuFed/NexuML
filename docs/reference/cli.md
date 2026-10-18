# CLI reference

This page is generated from the current Typer application, so it is the source of truth for command names, arguments, and options. Task guides intentionally do not copy every flag.

The CLI covers scenario resolution/build/training, dataset export, model export, smoke tests, tuning, registry inspection, backend inspection, and local-library management. `evaluate <policy-package>` replays a trusted complete RL policy without training; see [Reinforcement learning](../how-to/reinforcement-learning.md) for evaluation/media overrides and [Checkpoints](../how-to/checkpoints.md) for the separate training-resume path.

::: mkdocs-click
    :module: nexuml.cli.main
    :command: click_app
    :prog_name: nexuml
    :depth: 2
