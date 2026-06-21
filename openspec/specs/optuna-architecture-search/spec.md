# optuna-architecture-search

## Purpose

TBD

## Requirements

### Requirement: Per-trial scenario rebuild from a factory

The tuner SHALL build a fresh `ScenarioSpec` for each Optuna trial by invoking
the scenario factory `build(**params)` with the sampled parameters, instead of
mutating a previously built `ScenarioSpec`.

#### Scenario: Architecture knob changes pipeline structure

- **WHEN** a trial samples `detector="knn"` with `metric="euclidean"` and the
  derived `normalize="none"`
- **THEN** the rebuilt `ScenarioSpec` SHALL omit the L2Normalize stage and
  configure the kNN detector with Euclidean distance

#### Scenario: Detector swap with distinct parameter schema

- **WHEN** a trial samples `detector="gmm"`
- **THEN** the rebuilt `ScenarioSpec` SHALL configure a GMM detector using only
  GMM parameters (`n_components`, `covariance`, `reg_covar`) and SHALL NOT
  require kNN parameters

### Requirement: Declarative conditional and structural search space

The tuner SHALL accept a declarative `SEARCH_SPACE` whose entries are scalar
dotted-paths, categorical knobs with conditional `when` sub-spaces, or `derived`
values, and SHALL translate it into Optuna define-by-run sampling using
`TPESampler` with the `group` option enabled.

#### Scenario: Conditional sub-space is sampled per choice

- **WHEN** `detector` is a categorical entry with a `when` map providing
  detector-specific sub-spaces
- **THEN** only the sub-space for the chosen detector SHALL be sampled in that
  trial

#### Scenario: Derived value is computed, not sampled

- **WHEN** an entry is marked `derived` with a rule referencing other sampled
  values
- **THEN** its value SHALL be computed from those values and SHALL NOT create a
  new Optuna parameter

#### Scenario: Scalar-only space is backward compatible

- **WHEN** a scenario's `SEARCH_SPACE` contains only scalar dotted-path entries
- **THEN** tuning SHALL behave identically to the current `_set_nested`-based
  tuner

### Requirement: Arbitrary evaluator-metric objective

The objective SHALL optimize any metric named by `TUNING_SPEC.metric_key`,
resolved against the merged set of training logged metrics and evaluator results,
in the direction(s) given by `TUNING_SPEC.directions`.

#### Scenario: Optimize the omega evaluator metric

- **WHEN** `TUNING_SPEC.metric_key="omega"` and `directions=["maximize"]` and
  `omega` is permitted by the evaluation `test_result_metrics`
- **THEN** the objective SHALL read `omega` from the merged evaluator results and
  the study SHALL maximize it

#### Scenario: Missing metric fails loudly

- **WHEN** `metric_key` is not present in the merged metrics
- **THEN** the objective SHALL raise an error that lists the available metric
  keys

### Requirement: Tuning behavior has fast regression coverage

The test suite SHALL cover core Optuna architecture-search behavior with small
deterministic tests that do not run expensive studies.

#### Scenario: Search space sampling is exercised with fake trial

- **WHEN** a test passes a representative search space with scalar,
  categorical, conditional, and derived entries to the tuning code
- **THEN** sampled parameters SHALL match the expected structure without
  launching a long optimization run

#### Scenario: Per-trial scenario rebuild is verified

- **WHEN** tuning evaluates multiple fake trials with different sampled
  architecture choices
- **THEN** each trial SHALL build a fresh `ScenarioSpec` from the factory
  rather than mutating a reused spec

#### Scenario: Missing objective metric fails loudly

- **WHEN** tuning completes a fake trial whose merged metrics do not include
  the configured objective metric key
- **THEN** the objective SHALL raise an error that lists available metric keys

#### Scenario: Existing optuna behavior remains fast-suite compatible

- **WHEN** the default fast pytest suite runs
- **THEN** tuning regression tests SHALL use fakes, monkeypatches, or tiny
  in-memory objects and SHALL NOT require real training, real datasets, or
  many Optuna trials
