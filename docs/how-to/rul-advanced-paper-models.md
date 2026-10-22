# NEX-233 — DMM-JA, PatchFormer and RUL-Mamba in NexuML

Three paper-structured backbones are implemented as native `@layer` NexuML
components with native `@scenario` recipes, no changes to NexuML core and
no custom trainer.

## Primary sources and correct benchmark domains

| Work | Original task | Paper architecture | NexuML implementation |
|---|---|---|---|
| Niu et al., **DMM-JA**, IEEE ICASSP 2026, DOI 10.1109/ICASSP55912.2026.11464170 | ETDataset equipment degradation, oil temperature + load | Bimodal dynamic fusion, parallel convolutions 3/7/11 (dilation 2 on 11), 2-layer BiLSTM, Mamba, positional encoding and jump-aware gate | `AdvancedRULForecaster(architecture="dmm_ja")` |
| Liu et al., **PatchFormer**, J. Power Sources 631 (2025), 236187, DOI 10.1016/j.jpowsour.2025.236187 | CALCE/NASA/TJU **battery capacity forecasting**, followed by RUL-to-EOL evaluation | Variable embedding, intra/inter-patch CSE-based DPAN, feature fusion, feature-wise multi-head attention (FAN) | `AdvancedRULForecaster(architecture="patchformer")` |
| Huang et al., **RUL-Mamba**, J. Energy Storage 120 (2025), 116376, DOI 10.1016/j.est.2025.116376 | NASA/Oxford/TJU **battery capacity forecasting**, followed by RUL-to-EOL evaluation | Variable embedding, feature-wise squeeze/excitation (FAN), selective SSM encoder, selective SSM decoder and gated residual (GRN) | `AdvancedRULForecaster(architecture="rul_mamba")` |

The two battery authors publish their own code:
[PatchFormer](https://github.com/USTC-AI4EEE/PatchFormer) and
[RUL-Mamba](https://github.com/USTC-AI4EEE/RUL-Mamba).
The implementations here are independent NexuML-native modules informed
by the PDFs and authors' reference implementations, not copied external
training frameworks.

**Fidelity limitations:** The PyTorch selective SSM is an unfused
Mamba-style input-selective recurrence, **not** the optimized
`mamba_ssm` CUDA/Triton implementation. Do not attribute the papers'
throughput numbers to it. PatchFormer uses CSE-style dual patch
weighting and feature-wise transformer attention, but does not duplicate
every external implementation detail. DMM-JA uses a configurable
jump-difference threshold of 2.5 on normalized sensor values; the
paper's exact threshold and ETDataset input preprocessing are not
established by the PDF alone. The C-MAPSS recipes below are **architecture
transfer experiments**, not comparable published-protocol reproductions.

## 1. C-MAPSS FD001 architecture comparisons

Each model ingests causal normalized turbine windows and emits a scalar
remaining useful life target. The data loader automatically downloads
official NASA C-MAPSS data if needed, uses disjoint engine
train/validation/test groups and retains Goel-aligned 14 sensors.

The architecture-specific windows follow the papers' respective
reported time horizons where applicable: DMM-JA 24, PatchFormer 30,
RUL-Mamba 16. The models optimize **direct RUL MSE** on C-MAPSS for a
clean target comparison. This is intentionally different from the
SMAPE-capacity prediction of the two battery papers.

```bash
nexuml resolve maintenance-rul-dmm-ja-adapted \
  --output .experiments/rul-dmm-ja.yaml
nexuml train --config .experiments/rul-dmm-ja.yaml

nexuml resolve maintenance-rul-patchformer-adapted \
  --output .experiments/rul-patchformer.yaml
nexuml train --config .experiments/rul-patchformer.yaml

nexuml resolve maintenance-rul-mamba-adapted \
  --output .experiments/rul-mamba.yaml
nexuml train --config .experiments/rul-mamba.yaml
```

As with Goel, the official 100 held-out FD001 engines are evaluated
separately against *capped* (130 cycles) and *raw* NASA labels. Each run
has independent checkpoints, reports exact held-out MAE/RMSE/NASA
penalties and exports a per-engine audit under
`.experiments/rul/audits/`. Checkpoints are selected on
**validation loss**, not the official test metric.

**DMM-JA adaptation:** C-MAPSS sensor channels are divided into two
groups for the dynamic fusion layer. These groups are NOT the ETDataset
oil-temperature and electrical-load modalities, so these C-MAPSS
scores cannot reproduce DMM-JA's published ETDataset numbers.

## 2. Genuine temperature/load DMM-JA inputs

The alternative `maintenance-rul-dmm-ja-bimodal` scenario takes the
**two input modalities specified by DMM-JA**, rather than arbitrary
halves of turbine sensor channels. It requires **real RUL annotations**.

Supply a CSV with one measurement per row and the following named
columns:

| Column | Meaning |
|---|---|
| `asset_id` | Equipment identity |
| `cycle` | Strictly increasing observed time/cycle identifier |
| `temperature` | Raw measured temperature |
| `load` | Raw measured electrical/mechanical load |
| `rul` | Actual observed remaining life, in consistent physical units |
| `split` | `train`, `val` or `test` assigned consistently per asset |

Train, validation and test assets must be **disjoint**, nonempty and
individually ordered; RUL labels are never inferred from unlabeled
ETDataset data. Input channels are standardized with **training
assets only**; the normalizer and split provenance are stored under
`.experiments/rul/prepared/`.

```bash
export NEXUML_RUL_BIMODAL_CSV=/path/to/labeled_temperature_load_rul.csv

nexuml resolve maintenance-rul-dmm-ja-bimodal \
  --output .experiments/dmm-ja-bimodal.yaml
nexuml train --config .experiments/dmm-ja-bimodal.yaml
```

The model uses a 24-step history, 64-dimensional representation, two
BiLSTM layers, Adam 2e-4, dropout 0.15, batch size 64 and up to
700 epochs. The jump threshold and plateau scheduler are explicit
implementation choices where source details are incomplete. These
results remain a DMM-JA **method adaptation** until the exact ETDataset
data/label protocol and optimized Mamba implementation are reproduced.

## 3. Native NASA battery capacity forecasting

The PatchFormer and RUL-Mamba models additionally support their
original **capacity prediction task** in two native NexuML recipes:

```bash
export NEXUML_BATTERY_CAPACITY_ROOT=/path/to/nasa-capacity-csvs

nexuml resolve maintenance-battery-patchformer-nasa \
  --output .experiments/battery-patchformer.yaml
nexuml train --config .experiments/battery-patchformer.yaml

nexuml resolve maintenance-battery-rul-mamba-nasa \
  --output .experiments/battery-rul-mamba.yaml
nexuml train --config .experiments/battery-rul-mamba.yaml
```

The source root must contain the **real measured** NASA lithium-ion
battery capacity traces exported as per-cell UTF-8 CSV files. Each
CSV must have exactly named `cycle` and `capacity` columns (other
columns are allowed), with monotonically increasing cycle identifiers.
The required filenames are `B0005.csv`, `B0006.csv`, `B0007.csv`
and `B0018.csv`. There is **no simulated data fallback** and this
adapter does **not** pretend to download/parse the original proprietary
MATLAB battery package; obtain and convert the authentic publicly
available NASA battery discharge-capacity records first.

- Train cells: B0006, B0007 and B0018.
- Test cell: B0005, held out before preprocessing and model selection.
- Validation: last 20% of each training cell's observations; training
  uses earlier observations. The temporal windows contain only past
  cycles. Training-only capacity min/max is saved to
  `.experiments/rul/prepared/<run>.normalizer.json`.
- Input: historical min/max-normalized measured capacity sequence.
- Target: **next observed capacity in Ah**, not a synthetic RUL.
- Objective: SMAPE on capacity, as in the battery publications.
- Evaluation: next-cycle capacity MAE/RMSE **in Ah** and SMAPE;
  these are not RUL errors and must not be compared to C-MAPSS scores.

The NASA end-of-life criterion is **1.40 Ah** as reported in both papers.
For a single historical capacity window, the utility
`nexuml_library.evaluation.maintenance.battery_eol_rollout.rollout_capacity_to_eol`
iterates the capacity forecaster, transforms forecasts with the saved
training-only offset/scale, and returns the first forecast crossing cycle.
A result of `-1` means no predicted EOL within the selected finite
horizon (right-censored). This helper does **not** inspect actual future
test measurements. Report EOL-RUL statistics separately from the
one-step capacity forecast metrics.

**Not yet an exact end-to-end battery paper reproduction:** The papers
also use publication-specific per-dataset cleaning (including outlier
removal/interpolation), distinct horizon and starting-point protocols,
and repeated experiments/hyperparameter tuning. The common CSV adapter
and single-step objective do not claim to have reproduced those
complete evaluation protocols.

## 4. Technical implementation

- `layers/model/rul_selective_ssm.py`: causal depthwise convolutions,
  input-dependent time-step/B/C projections, diagonal stable A,
  selective recurrent scan, output gate and residual. Pure PyTorch
  reference implementation.
- `layers/model/rul_advanced.py`: DMM-JA, PatchFormer DPAN/FAN and
  RUL-Mamba FAN/SSM/GRN backbones, wrapped in a typed NexuML
  `PipelineLayer` with `LayerDefinition`.
- `data/maintenance/bimodal_rul.py`: real dual-modality measurements
  with explicit RUL labels and asset-disjoint splits.
- `data/maintenance/battery_capacity.py`: real capacity CSV ingestion,
  chronological split and persistent train-only normalization.
- `evaluation/maintenance/battery_capacity.py`: native SMAPE layer
  and held-out capacity evaluator.
- `evaluation/maintenance/battery_eol_rollout.py`: inference-only,
  finite-horizon, censored EOL time calculation.
- `scenarios/maintenance/rul_advanced.py`,
  `scenarios/maintenance/rul_dmm_ja_bimodal.py` and
  `scenarios/maintenance/battery_capacity.py`: named native
  `nexuml resolve` / `nexuml train` scenarios.

## 5. Validation

```bash
ruff check .
ruff format --check .
ty check

pytest tests/library/test_rul_advanced.py \
       tests/library/test_rul_bimodal.py \
       tests/library/test_battery_capacity.py \
       tests/library/test_rul_benchmarks.py \
       tests/_registry -v
```

The new tests cover SSM causality, backward gradients, dual-patch
padding, stage shapes, configuration serialization, real capacity
labels/splits, SMAPE and censored battery EOL rollouts. **Performance
claims require completed experiments**; the architecture modules alone
do not establish that these models beat the current safety-aware LSTM.
