# agent-authored-research-scenarios

## Purpose

TBD

## Requirements

### Requirement: Scenario-file search-space and tuning-spec contract

A Python scenario file SHALL be able to declare `SEARCH_SPACE`, `TUNING_SPEC`,
and an optional `build(**params) -> ScenarioSpec` factory. `SEARCH_SPACE` SHALL
support scalar dotted-path entries (applied to the built scenario), categorical
architecture knobs with conditional `when` sub-spaces, and `derived` entries.
The loader SHALL classify each entry as a scalar override (dotted attribute path
with no `choices` / `when` / `derived`) or an architecture knob, and SHALL
validate the declarations with clear errors. The scalar portion SHALL remain
exportable to YAML.

#### Scenario: Architecture knobs map to the factory

- **WHEN** a scenario file declares architecture knobs and a `build` factory
- **THEN** sampled architecture values SHALL be passed to `build(**params)` and
  scalar entries SHALL be applied to the resulting scenario

#### Scenario: Conditional and derived entries are accepted

- **WHEN** `SEARCH_SPACE` includes a categorical entry with a `when` map and a
  `derived` entry
- **THEN** the loader SHALL accept them and expose them to the tuner's
  define-by-run resolver

#### Scenario: Scalar-only declaration stays YAML-exportable

- **WHEN** a scenario declares only scalar dotted-path entries
- **THEN** the declaration SHALL remain exportable to YAML and load identically
  to the existing behavior
