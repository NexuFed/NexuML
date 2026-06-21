# Specification: Metadata Contract

## Purpose

Document the eval metadata contract for `_attach_eval_metadata` and `AxisKeySpec.source`, making the metadata lifecycle and column conventions explicit for developers.

## Requirements

### Requirement: Metadata contract documentation
`_attach_eval_metadata` in `src/nexuml/training/lightning.py` SHALL have a comprehensive docstring documenting the metadata contract:
- Metadata columns are string-valued (basename, machine, domain, target, section)
- Resolved from the dataloader's dataset via `sample_index` lookup
- Available only during eval/test phases (not training)
- `AxisKeySpec.source = "metadata"` declares which axis keys resolve from metadata

#### Scenario: Docstring explains metadata source
- **WHEN** a developer reads `_attach_eval_metadata` docstring
- **THEN** it SHALL explain that metadata comes from the dataset's pandas DataFrame, looked up via `sample_index` in the input TensorDict

#### Scenario: Docstring lists typical metadata columns
- **WHEN** a developer reads the docstring
- **THEN** it SHALL list the typical DCASE metadata columns: basename, machine, domain, target, section

#### Scenario: Docstring explains phase availability
- **WHEN** a developer reads the docstring
- **THEN** it SHALL note that metadata is only attached during test_step (eval/test phases), not during training

### Requirement: AxisKeySpec metadata source documented
`AxisKeySpec` in `src/nexuml/core/types.py` SHALL have an expanded docstring for the `source` field explaining when `"metadata"` is appropriate.

#### Scenario: AxisKeySpec documents metadata usage
- **WHEN** a developer reads `AxisKeySpec.source` documentation
- **THEN** it SHALL explain that `"metadata"` is for non-tensor columns from the dataset metadata DataFrame (e.g., machine name, domain label) that are needed for grouped evaluation
