# Reference

Compact, information-oriented reference for NexuML. These pages document interfaces, commands, and configurations without tutorials or extended explanation.

## CLI reference

- [CLI reference](cli.md) — all `nexuml` commands and flags, generated from the current implementation.

## Python API reference

- [API reference](api/) — auto-generated from `nexuml` and `nexuml_library` docstrings.

## Spec and config reference

- [ScenarioSpec](scenario-spec.md) — all fields of `ScenarioSpec` and nested spec types.
- [Tuning file](tuning-file.md) — Optuna search space file format.

## Registry and extension reference

- [Decorators](decorators.md) — `@scenario`, `@layer`, `@data_source`, `@eval_algorithm`.
- [Registry inspection](registry.md) — `nexuml registry list` commands and output.
- [Backends](backends.md) — available backend implementations and configuration.
- [Environment variables](environment.md) — `NEXUML_DATA_ROOT`, `NEXUML_LOGS_ROOT`, and other roots.
