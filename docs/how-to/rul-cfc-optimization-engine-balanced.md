# NEX-233: CfC RUL generalization — AdamW and engine-balanced supervision

This is the next controlled study after the three-seed readout-capacity
ablation. The primary question is whether RUL generalization improves
when optimizing a small, frozen-feature temporal readout more
conservatively and reducing the impact of highly overlapping windows
from long-lived engines.

The **same FD001 seed-42 frozen DPAN and CfC Flow** provide features for
all experiments. Original engine-disjoint train/val/test boundaries and
train-only feature normalization are preserved; the official test set
is not used for hyperparameter selection.

## 1. Model and experimental factors

All runs use the original [30,128] DPAN sequence, optionally concatenated
with frozen CfC Flow displacement, and an identical new 64-wide temporal
attention readout. The original PatchFormer RUL regression head is unused.

The three optimization settings are:
- `adam_1e3`: Adam, learning rate 0.001, original reference
- `adamw_3e4`: AdamW, learning rate 0.0003, weight decay 0.01
- `adamw_1e4`: AdamW, learning rate 0.0001, weight decay 0.01

The dropout settings are 0.1 (`d010`) and 0.3 (`d030`).
The supervision settings are `window` (ordinary window MSE) and
`engine` (inverse-engine-window-frequency weighted MSE). Each setting
is paired with frozen `encoder` and frozen `flow` input arms.

`training.lr` is set explicitly alongside the optimizer's own
`lr`; NexuML's compiler uses `training.lr` when resolving the
optimizer, so changing only the optimizer kwargs would not work.

## 2. What engine-balanced supervision actually means

If engine `e` has `n_e` train windows, the original prepared
training split contains `N` total windows from `E` engines.
Each training window from engine `e` receives the weight

```text
w_i = N / (E * n_e)
L_train = mean_over_training_windows(w_i * (prediction_i - rul_i)^2)
```

Thus, for uniform window mini-batch sampling, the expected training
objective equals the unweighted mean of per-engine training MSEs.
All train windows, including every degradation stage, are retained.
A global inverse-count weight is used; weights are **not** normalized
within each mini-batch.

Validation and test windows always have weight **one**, and the native
unweighted `val/loss` remains the checkpoint-selection criterion.
The train-only weights are recomputed from metadata without using RUL
labels from validation or test.

## 3. Validation robustness (no upstream leakage)

The epoch diagnostic callback continues to calculate full-split,
dropout-disabled, sample-weighted train and validation MSE.

Additionally, it deterministically sorts the *existing* held-out
validation-engine IDs by SHA-256 and assigns them round-robin to four
disjoint groups. It reports:
- Each group's window-weighted validation MSE
- Each group's mean per-engine validation MSE
- The worst group's MSE and across-group standard deviation
- Per-engine validation MSE at the best fitting epoch

**These are diagnostic partitions, not four independent retrainings or
new upstream held-out sets.** Moving original DPAN/Flow training engines
into validation would invalidate the upstream checkpoint provenance.
A genuinely independent validation split would require new upstream
PatchFormer and Flow training on appropriately separated engines.

## 4. Running the native experiments

Default: the complete 24-run grid for head seed 42:

```bash
git pull
bash scripts/launch_cfc_gen2_fd001.sh
```

Add head-seed 43 and 44 repetitions while preserving the same frozen
DPAN/Flow weights (72 runs total):

```bash
HEAD_SEEDS="42 43 44" bash scripts/launch_cfc_gen2_fd001.sh
```

The launcher is fail-fast and uses only `nexuml resolve` and
`nexuml train`. Separate checkpoint/CSV/audit files prevent collisions.

You can restrict the dimensions with the environment variables
`ARMS`, `DROPOUTS`, `OPTIMIZERS`, and `WEIGHTINGS` to run a quick
smaller sample; the launcher only generates the full paired summary when
all 24 settings of each specified head seed have been run.

A single example:

```bash
nexuml resolve maintenance-rul-cfc-gen2-flow-d010-adamw_3e4-engine-head-s42 \
  --output .experiments/rul/gen2/example.yaml
nexuml train --config .experiments/rul/gen2/example.yaml
```

## 5. Comparison outputs

- `.experiments/rul/gen2/<run>.curves.csv`: each fitting epoch's
  unweighted train/val MSE and RMSE, gap, worst validation group
- `.experiments/rul/gen2/<run>.curves.validation_engines.json`:
  best-epoch per-engine and four validation-group error distributions
- `.experiments/rul/gen2/summary.json`: matched flow/encoder deltas,
  mean and standard deviation across requested head seeds, worst-group
  stress result, and validation-only ranking
- `.experiments/rul/audits/<run>.geometry.json`: frozen checkpoint
  hashes, asset counts, and engine-balanced objective flag

The native `nexuml train` lifecycle may evaluate on the official
test set after training. This study's ranking and summary **do not read**
those test scores.

Do not use the four validation groups as separate hyperparameter
selection attempts; they are a fixed robustness diagnostic of the same
validation pool.

## 6. Verification

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_cfc_gen2.py \
       tests/library/test_rul_cfc_generalization.py \
       tests/library/test_rul_sequence_flow.py \
       tests/library/test_rul_advanced_native_cli.py \
       tests/_registry -v
```

All additions are NexuML library components plus scenario, test,
documentation and launch-script artifacts. No NexuML core code is changed.
