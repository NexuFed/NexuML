# NEX-233: Sensor-aware, trend-aware and trajectory-aware CfC RUL ablation

This is a **new enhancement experiment**, not a rerun of the six frozen
RUL-head architecture comparisons. The goal is to compare the contribution
of published architectural ideas within a single **CfC-based RUL estimator**.

All experimental designs are **original NexuML adaptations**. They do not
claim exact reproduction of DMHA-ATCN, MSPA-GAT, DSNet or TATFA.

## Reference data and architecture

Every model consumes the same native FD003 train-normalized
30-cycle × 15-sensor windows, using the same official engine-disjoint
train/validation/test grouping as the previous FD003 experiments.
They start from the same FD003 seed-42 pretrained PatchFormer checkpoint
and identical frozen train-endpoint normalization. No Flow
displacement is supplied to the RUL estimator.

```text
Raw sensor window [B,30,15]
     │
     ├── Optional sensor-to-sensor graph attention (before PatchFormer)
     │
Pretrained PatchFormer / DPAN [B,30,128]
     │  optionally fine-tuned end-to-end for supervised RUL
     │
Train-endpoint feature normalization [B,30,128]
     │
Projection [B,30,64] + learned positional embeddings
     │
     ├── Optional first-difference degradation trend features
     ├── Optional multiscale temporal convolution (3,7,15)
     └── Optional differentiable soft segmentation
     │
One residual CfC sequence layer [B,30,64]
     │
Learned attention pooling → 64→32→1 → sigmoid × 130 cycles
     │
     └── Optional training-only real predecessor consistency loss
```

All controls use identical training settings: Adam at 0.001,
head dropout 0.1, batch size 64 for independent runs (16 for the
native simultaneous comparison), 150-epoch maximum,
20-epoch validation early stopping and **validation point MSE**
for checkpoint selection. All experiments use readout seeds 42–44,
while the pretrained upstream PatchFormer is fixed at seed 42.
The same pretrained checkpoint is loaded when fine-tuning is enabled,
so the finetune ablation tests optimization, not different initialization.

## Seven independently toggled mechanisms

| Toggle | When enabled | Control |
|---|---|---|
| `finetune` | Pretrained PatchFormer weights updated by RUL gradients | Frozen pretrained PatchFormer |
| `graph` | Content-adaptive attention between the original 15 sensor channels | No sensor interaction module |
| `trend` | Adjacent-cycle latent first differences projected into the CfC features | No explicit trend |
| `multiscale` | Depthwise convolutions with receptive fields of 3, 7 and 15 cycles | No explicit multiscale module |
| `segmentation` | Differentiable learned soft segment-boundary recurrence | Fixed individual temporal tokens |
| `trajectory` | Additional same-engine consecutive-cycle prediction consistency at train time | Ordinary window MSE |
| `uncertainty` | 20-pass post-fit MC dropout and descriptive absolute-residual conformal quantile | Deterministic point estimator only |

Unlike a fully dynamic DSNet hard segmentation procedure, `segmentation`
uses **differentiable soft segment boundaries**, not discrete change-point
labels or a separate entropy/proxy classification objective.

The sensor graph is learned over the actual selected 15 sensors, **before**
PatchFormer mixes them into latent channels. No physical adjacency or
hard-coded dependency is assumed.

For `trajectory`, each observation is paired with an **actual preceding
window from the same engine and the same split**, never a synthetically
shifted embedding. Only training uses the consistency penalty;
validation/checkpoint ranking and official test evaluation use the same
ordinary point-estimation MSE as every other variant. The expected
RUL increment is 1 cycle, except in the capped-RUL regime where it is 0.
Nonexistent preceding windows are excluded by a mask.

The `uncertainty` option **does not train a different point estimator**;
it calculates post-hoc MC-dropout spread and a 90% nominal
MC-variance-scaled absolute-residual calibration interval (with fixed\n1-cycle scale floor) using one predetermined hash-selected
window per validation engine. These validation engines were also used
for checkpoint selection. Therefore the interval is **descriptive only**
and has **no finite-sample coverage guarantee**. No official test
labels are read by this diagnostic. Rigorous prediction-interval
coverage would require a disjoint calibration split not used to train or
select either the encoder or RUL estimator.

## Ablation matrix: 16 configurations × 3 seeds = 48 model branches

A 7-binary-factor complete factorial would require 128 configurations
per seed (384 independent trainings). This experiment instead uses a compact,
**controlled 16-cell effect study** per seed (48 separately registered models
or three simultaneous native composite training runs):

- `baseline`: all seven options disabled.
- Seven **single-on** configurations, named `finetune`, `graph`,
  `trend`, `multiscale`, `segmentation`, `trajectory`, `uncertainty`.
- `all`: all seven enabled.
- Seven **leave-one-out** configurations:
  `all_no_finetune`, `all_no_graph`, `all_no_trend`,
  `all_no_multiscale`, `all_no_segmentation`,
  `all_no_trajectory`, `all_no_uncertainty`.

The single-on comparison measures improvements over the baseline.
The leave-one-out comparison measures each component's marginal value
*within the combined system*. These are different estimands; the
16-config design does not identify all high-order interactions.

## Native NexuML workflow: no runners or environment variables

The primary comparison is registered as a **NexuML composite scenario**,
following the existing `maintenance-rul-all` pattern. Every composite
contains 16 independent CfC heads with disjoint trainable parameters,
loss keys and per-head validation metrics, all in one typed
`ScenarioSpec`. There are three independently initialized composite
scenarios (seeds 42, 43 and 44).

**Prerequisites:** FD003 seed-42 pretrained PatchFormer checkpoint and
train-endpoint feature normalization must already be available. These
can be produced with their existing native NexuML scenarios. No
shell-runner script or environment-variable configuration is needed.

From the repository root, for a fresh installation:

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_enhanced_cfc.py -v

nexuml resolve maintenance-rul-controlled-fd003-patchformer-s42 \
  --output .experiments/rul/controlled-patchformer-fd003.yaml
nexuml train --config .experiments/rul/controlled-patchformer-fd003.yaml

nexuml resolve maintenance-rul-patchformer-sequence-cfc-fd003-s42 \
  --output .experiments/rul/sequence-cfc-fd003.yaml
nexuml train --config .experiments/rul/sequence-cfc-fd003.yaml
```

The second prerequisite currently produces the original train-endpoint
normalization artifact; **Flow displacement is never used by this CfC
RUL comparison**. If the upstream FD003 artifacts are already present,
these prerequisite runs can be omitted.

### One `resolve` + `train` pair to compare all toggles

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare \
  --output .experiments/rul/enhanced-cfc/fd003/compare-s42.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-s42.yaml
```

Repeat for the other two head seeds without any custom runner:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-s43 \
  --output .experiments/rul/enhanced-cfc/fd003/compare-s43.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-s43.yaml

nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-s44 \
  --output .experiments/rul/enhanced-cfc/fd003/compare-s44.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-s44.yaml
```

Every composite run automatically records its own validation-only
comparison (from each branch's native `val/rul_loss_<configuration>`
metric) in:

```text
.experiments/rul/enhanced-cfc/fd003/
    rul-FD003-enhanced-cfc-compare-s42.validation.csv
    rul-FD003-enhanced-cfc-compare-s42.validation.json
```

Corresponding paths exist for `s43` and `s44`. The JSON report
contains per-arm best-observed fitting-epoch validation MSE, best epoch,
single-on versus baseline and all-on versus leave-one-out comparisons.
Test predictions are not inspected by the comparison callback.

### Crucial difference: composite versus independent training

The composite scenario uses a **shared Adam optimizer**, an aggregate
validation loss and shared early stopping. Although the heads do not
share learned parameters, their gradients flow through a common optimizer
and they train for a common number of epochs. The comparator records
individual best *observed* validation scores, but does not restore an
individually optimal checkpoint for each arm. The composite sets
batch size 16 rather than 64 to limit memory use with 16 simultaneous
heads. Therefore it is an economical on/off **screening study**, **not**
the same experiment as 16 individually trained and independently
selected models. Its official test evaluations should not be treated as
per-arm optimal benchmark scores.

For exact, independently checkpoint-selected training, the 48 existing
registered scenarios remain available using the same NexuML interface,
without a script. For example:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-baseline-s42 \
  --output .experiments/rul/enhanced-cfc/fd003/baseline-s42.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/baseline-s42.yaml

nexuml resolve maintenance-rul-enhanced-cfc-fd003-all-s42 \
  --output .experiments/rul/enhanced-cfc/fd003/all-s42.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/all-s42.yaml
```

### Uncertainty is a post-fit diagnostic, not a point-estimation head

Enabling `uncertainty` in individual scenarios activates the MC-dropout
and descriptive conformal diagnostic. The composite scenario compares
all 16 configurations' point-estimation errors, including both
`uncertainty` on/off controls. Because this feature only affects
post-training intervals, it is expected to have no systematic effect on
training or validation point MSE; interval-quality analyses must use
the individual uncertainty-enabled scenarios.

The calibration data are drawn from validation engines also used for
checkpoint selection; coverage is descriptive only.

FD003 test results were already inspected in earlier experiments.
This is therefore exploratory architecture development, not an
untouched confirmatory test. A new held-out dataset or fully frozen
protocol is required for SOTA claims.
