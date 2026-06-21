# Specification: CLI Override System

## Purpose

Provide a repeatable `--override` flag on `train` and `tune` CLI commands for nested config field overrides, with console logging of applied changes and relative warmup notation support.

## Requirements

### Requirement: CLI override flag
The `train` and `tune` CLI commands SHALL accept a repeatable `--override key.path=value` option for nested config field overrides. Dot-separated paths SHALL traverse the Pydantic model hierarchy.

#### Scenario: Simple override
- **WHEN** `nexuml train my-scenario --override training.max_epochs=10` is executed
- **THEN** the scenario's `training.max_epochs` SHALL be set to 10 before compilation

#### Scenario: Nested override
- **WHEN** `nexuml train my-scenario --override training.scheduler.params.warmup_epochs=5` is executed
- **THEN** the scheduler's `warmup_epochs` param SHALL be set to 5

#### Scenario: Multiple overrides
- **WHEN** multiple `--override` flags are provided
- **THEN** all overrides SHALL be applied in order

#### Scenario: Invalid override path
- **WHEN** `--override nonexistent.field=value` is provided
- **THEN** the CLI SHALL print an error message identifying the invalid path and exit with non-zero status

### Requirement: Override logging
All applied overrides SHALL be logged in a Rich table showing the key path, old value, and new value.

#### Scenario: Override summary displayed
- **WHEN** one or more overrides are applied
- **THEN** a table SHALL be printed to console with columns: Key, Old Value, New Value

### Requirement: Relative warmup notation
`SchedulerSpec` SHALL accept a string warmup value with `%` suffix (e.g., `"5%"`) representing a fraction of `max_epochs`. This SHALL be resolved to an integer at compile time.

#### Scenario: Percentage warmup
- **WHEN** `SchedulerSpec(warmup="5%")` is resolved with `max_epochs=100`
- **THEN** the effective warmup SHALL be 5 epochs

#### Scenario: Percentage warmup with small max_epochs
- **WHEN** `SchedulerSpec(warmup="10%")` is resolved with `max_epochs=3`
- **THEN** the effective warmup SHALL be at least 1 epoch (never 0 from rounding)

#### Scenario: Absolute warmup still works
- **WHEN** `SchedulerSpec(warmup=5)` is used
- **THEN** the effective warmup SHALL be 5 epochs (no change from current behavior)

### Requirement: Remove hardcoded warmup clamping
The hardcoded warmup clamping logic in `cli/main.py` (lines 195-201) SHALL be removed. Warmup validation SHALL be handled by `SchedulerSpec.resolve_warmup()`.

#### Scenario: No warmup clamping in CLI
- **WHEN** `nexuml train my-scenario --max-epochs 5` is executed with a scenario that has `warmup_epochs=10`
- **THEN** the scheduler resolution SHALL handle the invalid warmup via `SchedulerSpec` validation, not CLI-level clamping
