# NEX-233: CfC sequence-RUL generalization and overfitting study

The earlier progress-bar `train/loss` is a *single training batch's*
loss; `val/loss` is a validation epoch aggregate. Comparing these two
numbers is not a meaningful generalization-gap measurement.

This experiment adds a **library-only Lightning callback** to measure
complete train and validation sets under **identical evaluation conditions**:

- Dropout disabled in both passes
- All windows evaluated, including the last partial mini-batch
- Each window weighted equally in overall MSE (not each loader batch)
- No shuffle, no train sampler, no augmentation and no gradients
- Asset-disjoint training and validation engines
- Per-engine validation errors measured for every epoch, with a best-epoch
  `.validation_engines.json` report (one MSE per validation engine)
- Official test labels not read in the diagnostic callback or summary

The callback reruns train and val evaluation each epoch, so it **increases
training time** compared with the previous RUL readout scenarios.

## Controlled 2x2 regularization experiment

Freeze the same existing **FD001 seed-42** PatchFormer DPAN and CfC flow
for all scenarios. Only train the full-token RUL readout, with separate
matching *encoder-only* and *CfC-flow-aware* arms:

| Variant | Head width | Dropout |
| --- | ---: | ---: |
| `w128_d010` | 128 | 0.1 |
| `w128_d030` | 128 | 0.3 |
| `w064_d010` | 64 | 0.1 |
| `w064_d030` | 64 | 0.3 |

Each variant uses the same native `Adam(lr=1e-3)`, batch size 64, max
150 epochs, minimum validation-loss checkpoints and early stopping
(patience 20). Trainable parameters differ only with head width.
Training uses the same original RUL target, with no former PatchFormer
regression/FAN head.

Two experimental arms use the same 256-channel input layout:

```text
Frozen DPAN Z=[30,128]
    |
    +-- Encoder control: [Z, 0]
    |
    +-- CfC-flow readout: [Z, phi_1(Z)-Z]
    |
Same temporal self-attention + learnable token pooling + RUL prediction
```

The validation curves and engine reports distinguish **underfitting**,
**overfitting** (training improves while validation worsens), and
systematically difficult validation engines. An absolute positive
train–validation gap alone does not establish overfitting.

## Single launcher (8 experiments)

The full factorial study for readout seed 42 runs with:

```bash
git pull
bash scripts/launch_cfc_generalization_fd001.sh
```

For matched readout initialization/mini-batch shuffling robustness across
three seeds, without changing the upstream frozen DPAN/Flow:

```bash
HEAD_SEEDS="42 43 44" bash scripts/launch_cfc_generalization_fd001.sh
```

This runs 24 head trainings in total. It does **not** retrain three
separately seeded upstream flows.

Example registered scenario:

```bash
nexuml resolve maintenance-rul-cfc-generalization-flow-w064_d030-head-s42 \
  --output .experiments/rul/generalization/cfc-compact.yaml
nexuml train --config .experiments/rul/generalization/cfc-compact.yaml
```

All scenarios run via native `nexuml resolve` and `nexuml train` on
the existing data pipeline. They create independent checkpoint and audit
directories and reuse the automatically discovered upstream weights.

### Outputs

- `.experiments/rul/generalization/<run>.curves.csv`: per-epoch
  `train_eval_mse`, `val_eval_mse`, both RMSE values, gap, window counts,
  number of validation engines
- `.experiments/rul/generalization/<run>.curves.validation_engines.json`:
  per-engine validation MSE/RMSE at the lowest diagnostic validation MSE
- `.experiments/rul/generalization/summary.json`: validation-only
  pairwise differences, mean and standard deviation across requested seeds,
  best validation configuration, and largest validation-engine errors
- `.experiments/rul/audits/<run>.geometry.json`: frozen upstream
  PatchFormer and Flow SHA-256 provenance
- Native validation-selected checkpoints and regular RUL auditing

The diagnostic best epoch uses **window-weighted evaluation-mode validation
MSE**; regular NexuML checkpoint restoration continues to use
`val/loss`, which may weight batches differently. These epochs can
therefore differ slightly.

**Important:** NexuML's normal train command currently runs its official
test lifecycle automatically at the end. The generalization *summary*
deliberately does not read those test results, and model selection must
be made only using train/validation statistics. FD001 has already
informed development, so these experiments remain exploratory.

### Validation-only interpretation

Look at the trend:
- Train MSE falls, val MSE rises after a minimum → evidence of overfitting
- Train and val MSE remain high → limited fitting or domain shift
- Train and val MSE fall similarly → no clear overfitting pattern
- A few validation engines dominate MSE → test sensitivity to
  engine composition, before increasing model complexity

Choose width/dropout by validation only; then replicate the chosen setting
on FD003 (with its own upstream flow) for less development-contaminated
confirmation.

## Checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_cfc_generalization.py \
  tests/library/test_rul_sequence_flow.py \
  tests/library/test_rul_advanced_native_cli.py \
  tests/_registry -v
```

The upstream full-sequence Flow transport/RUL experiments are unaffected.
