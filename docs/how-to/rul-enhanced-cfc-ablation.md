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
head dropout 0.1, batch size 64, 150-epoch maximum,
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

## Ablation matrix: 16 configurations × 3 seeds = 48 trainings

A 7-binary-factor complete factorial would require 128 configurations
per seed (384 runs). This experiment instead uses a compact, **controlled
16-cell effect study** for each seed:

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

## Run the native NexuML study

From the NEX-233 branch:

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_enhanced_cfc.py -v

set -o pipefail
bash scripts/launch_enhanced_cfc_fd003.sh \
  2>&1 | tee .experiments-enhanced-cfc-fd003.log
```

The launcher uses only `nexuml resolve` and `nexuml train` for every
model. It prepares the FD003 seed-42 PatchFormer and CfC endpoint
normalizer via existing registered scenarios when absent. It requires
no ZIP downloads or external Python files.

You can first run a four-cell smoke test:

```bash
HEAD_SEEDS="42" VARIANTS="baseline trend graph all" \
  bash scripts/launch_enhanced_cfc_fd003.sh
```

Example native single scenario:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-all-s42 \
  --output .experiments/rul/enhanced-cfc/fd003/all-s42.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/all-s42.yaml
```

All complete-grid validation summaries are stored at:

```text
.experiments/rul/enhanced-cfc/fd003/summary.json
```

The output reports best fitting-epoch validation MSE per seed, head
variance, baseline-versus-single effects, and all-versus-leave-one-out
effects. Negative MSE differences mean enabling the technique helps.

MC/conformal reports are saved alongside the training curves as
`<run>.uncertainty.json` for configurations with `uncertainty`
enabled. **Do not interpret an in-sample empirical calibration coverage
number as held-out coverage**.

The native NexuML lifecycle may still write official FD003 test audits.
Do not use them to choose architecture settings. FD003 was already
inspected in the earlier readout comparison, so this ablation is now
an **exploratory architecture-development study** rather than an
untouched confirmatory experiment. An additional held-out dataset
or fully frozen evaluation protocol is required for new SOTA claims.
