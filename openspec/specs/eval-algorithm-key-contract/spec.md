## Purpose

Define the key-contract system for evaluation algorithms: declared tensor inputs/outputs, axis provenance, group/normalization/reduction keys, fit keys/masks, and key-agnostic reusable primitives.

## Requirements

### Requirement: Evaluation algorithms declare tensor inputs and outputs
The system SHALL allow (but not require) key-contract evaluation algorithms to declare the tensor keys they read from `x`. The `x_keys`, `output_keys`, `fit_keys`, `staged` fields are removed. Tensor contracts are now enforced at the pipeline layer level via `LayerSpec.keys_in` / `keys_out`.

#### Scenario: Declared tensor key is available
- **WHEN** an evaluation algorithm declares `axis_keys` for grouping and the compiled evaluation input provides those keys
- **THEN** the algorithm can group results by those keys

#### Scenario: Declared tensor key is missing
- **WHEN** an evaluation algorithm declares `axis_keys` referencing a key not present in the pipeline output
- **THEN** the algorithm logs a warning and skips that axis rather than failing at spec-parse time

### Requirement: Evaluation algorithms declare axis inputs with provenance
The system SHALL support declared axis keys that identify the axis name and where it is sourced from, such as `x`, `y`, or metadata supplied by the scenario. Key-contract algorithms SHALL resolve axes only through shared accessors that enforce the declared source exactly.

#### Scenario: Axis key resolves from labels
- **WHEN** an evaluation algorithm declares `{key: "machine", source: "y"}`
- **THEN** the shared accessor resolves `machine` from labels only and fails if labels do not provide it

#### Scenario: Axis key resolves from transformed data
- **WHEN** a multiview frontend writes a `view` key into transformed data and an evaluation algorithm declares `{key: "view", source: "x"}`
- **THEN** the shared accessor resolves `view` from transformed data only without assuming DCASE-specific vocabulary

#### Scenario: Axis source fallback is rejected
- **WHEN** an algorithm declares `{key: "domain", source: "y"}` and `domain` exists only in `x`
- **THEN** validation or runtime access fails instead of silently falling back to `x`

### Requirement: Group, normalization, and reduction keys reference declared axes
The system SHALL validate that `group_keys`, `normalize_keys`, and `reduce_keys` only reference axes declared in the evaluation algorithm contract.

#### Scenario: Group key is declared
- **WHEN** an algorithm declares axis `machine` and uses `group_keys: ["machine"]`
- **THEN** validation accepts the grouping configuration

#### Scenario: Group key is undeclared
- **WHEN** an algorithm uses `group_keys: ["domain"]` without declaring a `domain` axis key
- **THEN** validation fails before the algorithm is instantiated for execution

### Requirement: EvalAlgorithmSpec exposes feature_key and label_key at top level
`EvalAlgorithmSpec` SHALL expose `feature_key: str | None = None` and `label_key: str | None = None` as top-level typed fields. These fields define the primary input tensor key and target label key used by the evaluation algorithm.

When `feature_key` or `label_key` is None, algorithm implementations SHALL fall back to `params.get("feature_key")` / `params.get("label_key")` for backward compatibility.

#### Scenario: feature_key at top level
- **WHEN** an `EvalAlgorithmSpec` is constructed with `feature_key="latent_asd"`
- **THEN** `spec.feature_key == "latent_asd"` is accessible without reading `spec.params`

#### Scenario: Backward-compatible fallback
- **WHEN** an `EvalAlgorithmSpec` has `feature_key=None` but `params={"feature_key": "latent"}`
- **THEN** algorithm implementations that check `spec.feature_key or spec.params.get("feature_key")` resolve to `"latent"`

### Requirement: Reusable evaluation primitives are key-agnostic
Reusable anomaly-detection evaluation algorithms SHALL operate on declared generic keys and SHALL NOT require hard-coded DCASE vocabulary such as machine, section, domain, or view unless those names are supplied by the scenario contract.

#### Scenario: Same primitive uses different axis names
- **WHEN** two scenarios provide different grouping axis names through their contracts
- **THEN** the same reusable distance estimator can group by either scenario's declared axes without code changes

#### Scenario: DCASE axis names have no intrinsic meaning in primitives
- **WHEN** a scenario declares axes named `machine`, `section`, or `domain`
- **THEN** reusable score-path primitives treat those as ordinary declared axes and do not select DCASE-specific code paths based on the names