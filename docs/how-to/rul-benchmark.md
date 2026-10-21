# NEX-233 — Supervised remaining useful life baseline suite

This benchmark is **RUL regression**, not abnormal-sound detection or
healthy/faulty Flow Matching. Train only through **NexuML's conventional
`nexuml build` and `nexuml train --config` commands**. Neither preparation
nor resolved-config generation trains a model.

## Implemented scope

- **Datasets:** NASA C-MAPSS (FD001–FD004, official test RUL offsets),
  XJTU-SY (fifteen bearing runs, stratified 3/1/1 asset split per condition).
- **Backbones:** feature-window MLP; LSTM; causal dilated residual TCN;
  causal Transformer encoder; TSMixer-like temporal/channel MLP.
- **Objectives:** MSE, MAE, smooth-L1/Huber, multi-quantile pinball.
- **Evaluation:** native NexuML train/validation/test, sample-weighted test MAE
  and RMSE, late-prediction bias, prediction-interval coverage, interval
  width and quantile-crossing rate for quantile models.
- **Reproducibility:** asset-disjoint train/val/test, training-asset-only
  global standardization, window ordering and label provenance, immutable
  prepared tensors and YAML, deterministic model initialization and
  train-start seed callback.

These are **competent baseline implementations**, **not** certified
reproductions of one particular paper's SOTA architecture or reported
benchmark result. No accuracy claims are made until actual experiments
have run. The former Flow Matching experiment remains separate.

### A. C-MAPSS dataset preparation

Obtain the original **NASA C-MAPSS turbofan degradation simulation**
files using the official NASA data repository. The raw directory must contain,
for each selected subset, e.g. `train_FD001.txt`, `test_FD001.txt`,
`RUL_FD001.txt`. This adapter deliberately **does not** download from
unverified mirrors.

```bash
python -m nexuml_library.evaluation.maintenance.rul_prepare \
  --dataset cmapss --root data/cmapss --subset FD001 \
  --output-file .experiments/rul/prepared/cmapss-FD001-seed42.pt \
  --window 30 --stride 1 --seed 42 --rul-cap 125
```

The official training engines are split **by engine ID** into model-training
and validation engines (default 80:20). The complete official test-engine
prefixes are *never* used for model fitting, normalization or early stopping.
By default, evaluation includes the **final window** of each truncated test
engine, scored against the official terminal RUL offset.

The 24 features are operating settings (3) plus sensors (21), retaining
the full channels, without using future cycles. RUL targets represent
`min(remaining_cycles,125) / 125`. This is a capped supervised task:
test targets are also capped for mathematical consistency; compare only to
published results using the same target-label convention.

### B. XJTU-SY dataset preparation

Obtain the original **XJTU-SY run-to-failure bearing vibration recordings**,
with directories `Bearing1_1` through `Bearing3_5` under three working
conditions. Each directory holds chronologically numbered CSV recordings.
We do **not** fabricate timestamps or infer a hard failure threshold.

```bash
python -m nexuml_library.evaluation.maintenance.rul_prepare \
  --dataset xjtu_sy --root data/xjtu_sy \
  --output-file .experiments/rul/prepared/xjtu-sy-seed42.pt \
  --window 20 --stride 1 --seed 42
```

The feature vector has 12 dimensions: RMS, standard deviation, maximum
absolute amplitude, skewness, kurtosis and frequency-bin centroid, each on
two vibration axes. The split is 9 train / 3 val / 3 test bearings, with
one val and one test bearing per operating condition. **It is an explicit
research split rather than the official protocol of any external paper.**
Labels are **number of recordings remaining**, rescaled by the maximum
observed **training** lifetime. To report minutes, you must verify the
source's acquisition interval and multiply by it. Never claim that
`RUL=0` means a physical failure occurred precisely at the file boundary.

### C. Native scenario resolution and training

**Use the conventional NexuML CLI.** Every RUL backbone is registered as a
normal scenario, so no custom configuration generator, Python training script,
or direct compiler call is required.

After preparing an artifact, set its absolute path for the **resolve step**:

```bash
export NEXUML_RUL_PREPARED_FILE="$(realpath .experiments/rul/prepared/cmapss-FD001-seed42.pt)"

nexuml resolve maintenance-rul-tcn \
  --output .experiments/rul/configs/cmapss-FD001-tcn-seed42.yaml

nexuml build .experiments/rul/configs/cmapss-FD001-tcn-seed42.yaml

nexuml train --config .experiments/rul/configs/cmapss-FD001-tcn-seed42.yaml
```

The resolved YAML captures the prepared artifact's path, dimensions, target
scale, split seed and model definition. Training loads the exact same
prepared dataset; it does **not** re-split assets or refit normalization.
The environment variable is needed for resolving the scenario, not for
running the saved YAML. The default `maintenance-rul-benchmark` scenario
is equivalent to the TCN baseline and has a known C-MAPSS FD001 fallback
shape so generic registry compilation succeeds before the local dataset
is available; **training still requires the actual prepared artifact**.
An explicitly selected missing artifact causes an immediate error.

Registered names for the backbone comparison:

| Scenario | Architecture |
| --- | --- |
| `maintenance-rul-feature-mlp` | Feature-window MLP |
| `maintenance-rul-lstm` | LSTM |
| `maintenance-rul-tcn` | Causal TCN |
| `maintenance-rul-transformer` | Causal Transformer |
| `maintenance-rul-tsmixer` | TSMixer |
| `maintenance-rul-quantile-tcn` | TCN with 0.1/0.5/0.9 pinball loss |

For XJTU-SY, point `NEXUML_RUL_PREPARED_FILE` to the prepared XJTU-SY
artifact and resolve the desired scenario name to a different output YAML.
The scenario derives `[window, feature_channels]`, target scale, and seed
from the validated prepared artifact, so the same registered architectures
work for both datasets. Each resolved configuration uses the standard
NexuML Lightning training, validation, checkpoint, and test lifecycle.

### One native NexuML scenario for all six RUL models

`maintenance-rul-all` is a **registered NexuML scenario** containing
the five point-regression models and the quantile TCN as separate branches.
No external suite script, custom trainer, or direct compiler invocation is
required. Use the **same two commands** as any other NexuML scenario:

```bash
export NEXUML_RUL_PREPARED_FILE="$(realpath .experiments/rul/prepared/cmapss-FD001-seed42.pt)"

nexuml resolve maintenance-rul-all --output .experiments/rul-all.yaml
nexuml train --config .experiments/rul-all.yaml
```

Each model has its own trainable parameters, prediction tensor, loss, metrics,
and named held-out RUL evaluator. The scenario has one datasource, one shared
optimizer and training schedule, and one aggregate validation-loss checkpoint.
Therefore this is a **jointly executed multi-branch comparison**, not six
independent training/early-stopping runs. To benchmark fully independent model
selection, use the six individual registered scenario names listed above and
invoke the same native `nexuml resolve` and `nexuml train --config` commands
for each. Different datasets (C-MAPSS versus XJTU-SY) also require their
own resolved scenario, because their sensor-window shapes and RUL units differ.

### D. Quantile predictions

Use `maintenance-rul-quantile-tcn` with the same native
`nexuml resolve` → `nexuml train --config` sequence.
The 0.1/0.5/0.9 pinball model outputs conditional lower, median and upper
RUL estimates. Its NexuML test evaluator reports actual coverage, width
and quantile-crossing rate. This is **not conformal calibration**: coverage
must be measured on held-out assets before making reliable-interval claims.

### E. Multi-seed evaluation

Re-run preparation with `--seed 43`, `44`, etc., giving each run a new
artifact and output YAML path. Set `NEXUML_RUL_PREPARED_FILE` to each artifact
when invoking `nexuml resolve`; the scenario derives that artifact's seed.
All windows from one physical asset must remain in one split.
These seeds resample the same equipment; do not treat repeated seed runs as
independent new machines or claim confidence from more window counts.

Validation uses `val/loss` for early stopping and best-checkpoint selection.
The standard NexuML session runs fit → validate → post-train fit → test;
the RUL evaluator computes exact **sample-weighted** held-out test statistics.
The native batch metric layers also log approximate batch-aggregated
MAE/RMSE during optimization; the exact final test evaluator is authoritative.

## Explicitly not yet part of this implementation

The broader research roadmap also includes:

- LightGBM feature regression (requires a **correct native fit/eval
  component lifecycle** rather than an external bespoke training script).
- PRONOSTIA/FEMTO-ST, IMS, and higher-fidelity N-CMAPSS dataset adapters.
- Publication-specific SOTA reproductions, ablation tuning, cross-domain
  transfer learning and masked self-supervised pretraining.
- Conformal coverage calibration, censored survival regression and
  Flow Matching as an optional post-baseline extension.

Those remain **future work**. Do not interpret the five implemented
backbones as a complete leaderboard or a universal SOTA claim.

## Quality checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_benchmarks.py -v
pytest tests/library/test_rul_native_cli.py -v
pytest tests/_registry -v
```
