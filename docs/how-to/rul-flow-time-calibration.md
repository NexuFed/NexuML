# NEX-233 — Flow time to RUL calibration

This is the second phase after the PatchFormer latent transport experiment
(validation sliced-Wasserstein reduction ~83.6% for FD001 seed 42).
Transport quality alone is **not** remaining useful life prediction.

The experiments below test whether flow time carries independently
useful degradation information and whether it improves the already-trained
PatchFormer RUL predictor. All workflows use registered NexuML scenarios,
existing C-MAPSS loaders, and the standard `nexuml resolve/train` commands.

## Methods and validation

A trained PatchFormer provides frozen mean-pooled DPAN embeddings
`z(x)` in R^128. Standardization uses only the healthy/degraded
*training endpoint* population recorded by the flow run.

We select up to eight healthy reference embeddings **from training
engines only**. The learned and frozen NEX-196 flow transports each
reference `z0` along a 50-step Euler path `phi_t(z0)`.

For an unseen window `x`, define:

```text
(t_hat, k_hat) = argmin_{t in {0,1/50,...,1}, k in {1,...,K}}
                 || z(x) - phi_t(z0_k) ||^2
d_hat = min_{t,k} ||z(x) - phi_t(z0_k)|| / sqrt(128)
```

This produces a **projected flow time** `t_hat` in [0,1] and an
off-trajectory distance `d_hat`; no observed test label is involved.
This time is a **learned transport coordinate**, not guaranteed
physical age, hazard or remaining cycles.

The dataset also stores `p(x)`, the unchanged pretrained PatchFormer
RUL forecast. These three features are precomputed *once per source
window* using the frozen models. Training and validation RUL labels
remain exclusively in NexuML's regular target TensorDict.

Two registered calibrators implement different scientific hypotheses:

1. **Flow-time-only**: a small, 32-hidden-unit MLP receives `t_hat`
   alone and predicts `130 * sigmoid(MLP(t_hat))`, in cycles.
2. **Hybrid**: the same-sized MLP receives
   `[t_hat, d_hat, p(x)]` and emits a correction
   `p(x) + 30 * tanh(MLP(...))`. It can modify the supervised
   PatchFormer prediction by at most 30 cycles.

Both are trained using **capped training labels**, checkpointed by
minimum validation loss, and evaluated exactly once via existing
NexuML capped/raw RMSE, MAE, NASA score, and 100-engine CSV audits.
No official test RUL enters normalization, reference selection, fitting,
checkpoint selection, or geometry calibration.

### Degradation ordering diagnostics

`.experiments/rul/audits/<run>.geometry.json` records:

- Pearson correlation of projected flow time with training RUL.
- Pearson correlation of projected flow time with **validation** RUL.
- Mean within-validation-engine Pearson correlation of flow time and
  increasing cycle index, plus the fraction of engines with positive
  progression. Constant time is reported as unavailable, not as zero.
- Counts of training-only reference paths, upstream model SHA-256 digests,
  subset/seed provenance, and a test-leakage flag.

This is intentionally descriptive: the geometry measures are not used
to select or tune the downstream model.

## Execute the FD001 seed-42 experiment

Prerequisites are the already-trained `maintenance-rul-patchformer-adapted`
and `maintenance-rul-patchformer-flow-fd001-s42` experiments. The first
produces the supervised embedding and the second the vector field.
The new data source discovers both checkpoints automatically from the
native NexuML run directories and existing endpoint provenance. No
environment variables or manually assigned model paths are required.

```bash
git pull
nexuml resolve maintenance-rul-patchformer-flow-time-fd001-s42 \
  --output .experiments/flow-time-fd001-s42.yaml &&
nexuml train --config .experiments/flow-time-fd001-s42.yaml

nexuml resolve maintenance-rul-patchformer-hybrid-fd001-s42 \
  --output .experiments/flow-hybrid-fd001-s42.yaml &&
nexuml train --config .experiments/flow-hybrid-fd001-s42.yaml
```

If these fail because prerequisites are missing, first train
`maintenance-rul-controlled-fd001-patchformer-s42`, then train
`maintenance-rul-patchformer-flow-fd001-s42` with the normal CLI.

An available flow checkpoint must come from the **same** subset and
seed. The best validation-SWD checkpoint is selected from its native
run directory (excluding `last.ckpt`); ambiguous selections are
rejected rather than silently using random weights. The corresponding
endpoint artifact records which supervised PatchFormer checkpoint
produced those embeddings; its SHA-256 must still match.

For FD003 and multi-seed comparisons the native registrations are:

```text
maintenance-rul-patchformer-flow-time-fd001-s42/s43/s44
maintenance-rul-patchformer-hybrid-fd001-s42/s43/s44
maintenance-rul-patchformer-flow-time-fd003-s42/s43/s44
maintenance-rul-patchformer-hybrid-fd003-s42/s43/s44
```

Run matching controlled PatchFormer and pretrained flow scenarios first.

## Interpretation

Compare the corresponding frozen PatchFormer baseline against both
calibrations on identical test engines:

- Flow time only vs. PatchFormer baseline: does inferred transport
  position carry enough RUL information on its own?
- Hybrid vs. baseline: does transport geometry add predictive value
  beyond the supervised PatchFormer output?
- Hybrid vs. flow time only: does baseline supervision dominate?

Inspect train/validation time-RUL correlations and per-engine
progression *before* interpreting a good test score causally.
The FD001 test set already informed prior model development, so
claims remain exploratory. FD003 and multi-seed runs test robustness
without treating training seeds as independent held-out datasets.

## Implementation

- `layers/model/rul_flow_calibration.py`: native flow-time and hybrid MLP head
- `evaluation/maintenance/rul_flow_geometry.py`: strict frozen checkpoint
  restoration, native Euler trajectory integration, reference projection
- `data/maintenance/rul_flow_calibration.py`: prepared frozen features,
  validation ordering diagnostics, unchanged native RUL splits/labels
- `scenarios/maintenance/rul_flow_calibration.py`: 12 public scenarios
- `tests/library/test_rul_flow_calibration.py`: geometry and CLI-contract tests

## Checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_flow_calibration.py \
       tests/library/test_rul_patchformer_study.py \
       tests/library/test_rul_advanced_native_cli.py \
       tests/_registry -v
```

Passing tests and measured held-out gains are **not claimed** before
these commands and experiments are executed.
