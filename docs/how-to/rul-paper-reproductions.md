# NEX-233 — Source-backed C-MAPSS paper settings (not exact reproductions)

This document records only parameters we could attribute to a published source.
**Matching data/training hyperparameters does not reproduce a paper's architecture
or its test evaluation protocol.** All scenarios use native
`nexuml resolve` and `nexuml train --config` with automatic NASA download.
No paper RMSE is claimed as reproduced.

## Available native literature-controlled experiments

### Lu et al. (2025), FD001: TCN parameter/protocol control

Source: [Lu et al., Scientific Reports 15, 19229 (2025)](https://doi.org/10.1038/s41598-025-00572-8),
[full text/Table 7](https://pmc.ncbi.nlm.nih.gov/articles/PMC12127457/).

| Field | Published FD001 | NexuML protocol |
| --- | --- | --- |
| Time window, stride | 40, 1 | 40, 1 |
| Input sensors | S2, S3, S4, S7, S8, S9, S11–S15, S17, S20, S21 | Same 14 |
| Feature normalization | Min–max | Train-asset-only min–max |
| Piecewise RUL cap | 125 cycles | 125 cycles |
| Batch | 256 | 256 |
| Optimizer / learning rate | Adam / 0.001 | Adam / 0.001 |
| Training epochs | 300 | Maximum 300; early stopping may terminate earlier |
| FD001 projected hidden width | 32 | 32 |
| Main architecture | AD-TCN + MSCDIM | **Generic causal TCN, not AD-TCN-MSCDIM** |
| Loss, kernel layout, depth, dropout | Not established for the TCN baseline by the cited table | MSE, NexuML default depth=3, dropout=0.1 |

The optimizer settings in Table 7 were reported for the **authors' model**;
they should not be misrepresented as a fully specified TCN ablation protocol.
Preprocessing is limited to the declared 14 sensors (no operational settings).
The published method has adaptive dilation and cross-dimension modules absent
from the current NexuML TCN. It also uses a separate 14-feature input
representation. Both method and architecture must be matched before claiming
replication of 12.54 (TCN) or 11.39 (AD-TCN-MSCDIM) RMSE.

```bash
nexuml resolve maintenance-rul-lu2025-tcn-protocol \
  --output .experiments/rul-lu2025-tcn-protocol.yaml
nexuml train --config .experiments/rul-lu2025-tcn-protocol.yaml
```

### Cen et al. (2024), FD001: MFSSCINet protocol on a TSMixer proxy

Source: [Cen et al., Engineering Applications of Artificial Intelligence 135,
108813 (2024)](https://doi.org/10.1016/j.engappai.2024.108813),
[article including Table 4](https://www.sciencedirect.com/science/article/pii/S0952197624009710).

| Field | Published FD001 | NexuML proxy |
| --- | --- | --- |
| Window | 28 | 28 |
| RUL cap | 120 | 120 |
| FD001 normalization | Min–max | Train-asset-only min–max |
| FD001 input | Remove constant sensor channels | Same 14 selected sensors |
| Batch | 256 | 256 |
| Optimizer / learning rate | Adam / 0.01 | Adam / 0.01 |
| Objective | MSE | MSE |
| Feature/sequence attention heads | 2 / 1 | **Not implemented** |
| FSSCI blocks / FSSCINet / prediction layers | 2 / 2 / 2 | **Not implemented** |
| Architecture | MFSSCINet | **TSMixer proxy, not MFSSCINet** |
| Epoch count, latent width and dropout | Not resolved from cited source | NexuML defaults, not publication settings |

The published 10.93 FD001 RMSE is for the actual MFSSCINet architecture.
A TSMixer, even with the same inputs and optimizer, is a different method.

```bash
nexuml resolve maintenance-rul-cen2024-tsmixer-proxy \
  --output .experiments/rul-cen2024-proxy.yaml
nexuml train --config .experiments/rul-cen2024-proxy.yaml
```

### Goel, Galchar & Kanu (2026), FD001: LSTM architecture and training protocol

Primary source: [arXiv:2604.27234, Section III, Algorithm 5, Training Protocol](https://arxiv.org/pdf/2604.27234).

| Field | Published FD001 | NexuML scenario |
| --- | --- | --- |
| Window / stride | 30 / 1 | 30 / 1 |
| Input sensors | 14 retained sensors | Same 14 |
| Feature normalization | Train-only z-score | Train-only z-score |
| RUL target cap | 130 cycles | 130 cycles, physical units |
| Split / seed | 80/20 asset-disjoint, seed 42 | Same |
| Short-test padding | Zero | Zero |
| Incomplete training windows | Use full 30-cycle windows | Same |
| LSTM | One layer, hidden dimension 32 | Same |
| Head | Dropout 0.5, linear 32→8→8→1 with ReLU | Same |
| Loss | MSE | MSE |
| Optimizer / LR | RMSprop, 0.001 | Same |
| Weight decay | 0.00001 | Same |
| Batch | 64 | 64 |
| Scheduler | ReduceLROnPlateau, factor 0.5, patience 5 | Library callback, same |
| Early stop / maximum epochs | Patience 20 / 200 epochs | Same |
| Test NASA score | Exponential asymmetric penalty | Same formula |

The paper-specific scenario now excludes incomplete training windows,
zero-pads short test histories, retains RUL labels in cycles (not
fractions), and reports the asymmetric NASA score. The generic NexuML
scenarios retain their original preprocessing defaults.
Two held-out evaluators report MAE, RMSE and NASA score against both
the 130-cycle capped labels and the **original uncapped NASA test RUL
offsets**, with explicit metric names `rul_test_*` and
`rul_official_raw/rul_test_*`. Training always uses capped labels and
never trains against the raw test offset. The exact published evaluation
label convention remains unresolved; report both sets rather than selecting
one solely because it matches the paper's RMSE. Published results (FD001 RMSE 14.93, MAE 11.73,
NASA score 410.97) are **reference numbers, not reproduced measurements**.

The paper's 1D CNN is also implemented with convolution channels
14→32→64→64, three same-padded kernel-3 convolutions, dropout 0.3,
and the published 64×30→128→1 regression head. It uses the same data
cache, RMSprop, scheduler and independent checkpointing as the LSTM.
The authors report FD001 CNN RMSE 16.97 and NASA score 365.

```bash
nexuml resolve maintenance-rul-goel2026-lstm-protocol \
  --output .experiments/rul-goel2026-lstm.yaml
nexuml train --config .experiments/rul-goel2026-lstm.yaml

nexuml resolve maintenance-rul-goel2026-cnn-protocol \
  --output .experiments/rul-goel2026-cnn.yaml
nexuml train --config .experiments/rul-goel2026-cnn.yaml
```

For three more recent paper-structured RUL architectures and battery forecasting
protocols, see [advanced DMM-JA / PatchFormer / RUL-Mamba models](rul-advanced-paper-models.md).
These transfer benchmarks are not Goel or original battery paper reproductions.

For a separate, **non-publication**, cost-sensitive RUL research track,
see [safety-aware RUL experiments](rul-safety-experiments.md).
These experiments do not replace the reference-paper scenarios.

## Goel FD001 methodology audit (October 2026)

The [published v1 PDF](https://arxiv.org/pdf/2604.27234) confirms:

- Training RUL uses a piecewise constant cap of **130 cycles**.
- FD001 removes sensors 1, 5, 6, 10, 16, 18, and 19, retaining 14.
- Z-score feature normalization fits on training data only.
- An 80/20 split is performed **by engine ID**; seed 42 is specified.
- Each model uses 30-cycle windows (stride 1). Short test histories are zero-padded.
- Model selection uses minimum validation loss and restores the best checkpoint.
- RMSprop, learning rate 0.001, weight decay 1e-5, batch 64,
  ReduceLROnPlateau(factor=0.5, patience=5), and early stopping
  (patience=20, maximum 200 epochs) are explicitly specified.

The published text does **not** define the exact engine-selection RNG
implementation, whether the feature scaler is fitted using just the 80
training engines or all 100 official training engines, or whether the
official held-out NASA RUL offsets are capped before reporting metrics.
Figure 3/4 visually suggests a capped target range but cannot prove it.
We do not silently substitute assumptions for these facts.

### Native evaluation and per-engine diagnostics

The Goel CNN and LSTM scenarios now restore the best `val/loss`
checkpoint weights after native NexuML fitting. The normal
`nexuml train --config` command then evaluates the restored weights.
This behavior differs from the earlier runs, where NexuML's session tested
the live final-epoch model.

On completion of the test stage, the `RULPerEngineAudit` evaluator writes
a deterministic CSV and JSON manifest, separately for the LSTM and CNN:

```text
.experiments/rul/audits/
  rul-cmapss-FD001-lstm-mse-seed42-goel2026-lstm-protocol-raw.csv
  rul-cmapss-FD001-lstm-mse-seed42-goel2026-lstm-protocol-raw.json
  rul-cmapss-FD001-lstm-mse-seed42-goel2026-lstm-protocol-raw.checkpoint.json
  rul-cmapss-FD001-cnn-mse-seed42-goel2026-cnn-protocol-raw.csv
  rul-cmapss-FD001-cnn-mse-seed42-goel2026-cnn-protocol-raw.json
  rul-cmapss-FD001-cnn-mse-seed42-goel2026-cnn-protocol-raw.checkpoint.json
```

Each CSV includes the test-engine ID, final-window index, model prediction,
capped and official raw targets, signed residuals under both targets,
individual asymmetric NASA penalties, and whether the official target
exceeded 130 cycles. The JSON records the split engine IDs, random seed,
normalization method, digest of fitted normalization parameters, train-only
normalization scope, and the versioned source artifact. The additional
`*.checkpoint.json` file records the best validation checkpoint location,
checkpoint epoch, saved step and monitored validation-loss value.

The native scalar metrics for the Goel scenarios continue to include
`rul_test_*` (capped) and `rul_official_raw/rul_test_*` (uncapped).
The per-engine audit makes it possible to determine which few engines
cause the scores to diverge.

**Re-resolve both scenarios before running them.** Existing resolved YAML
configurations do not automatically pick up newly added callbacks or
evaluation algorithms. Model weights from earlier runs cannot be reconstructed
from aggregate test logs; to generate the new CSVs, run native NexuML
evaluation on a trained checkpoint or retrain through the updated scenarios.

The audit cannot independently verify the unpublished per-engine
predictions, split assignments, or normalizer parameters of the paper.
Consequently, the published reference numbers remain targets for
methodological comparison rather than guaranteed bit-for-bit reproducible
outputs.

## Paper benchmarks not represented as exact NexuML scenarios

- **Li, Ding & Sun (2018), deep CNN:** the reported FD001 RMSE of 12.61
  involves their specific deep convolution architecture. NexuML's RUL
  forecaster currently offers an MLP, LSTM, TCN, Transformer and TSMixer,
  **not** that CNN. A hyperparameter-only rename would be misleading.
  Source: [Reliability Engineering & System Safety, 172, 1–11](https://doi.org/10.1016/j.ress.2017.11.021).
- **Hossain et al. (2026), stacking ensemble:** the published FD001
  RMSE 9.989 uses **four heterogeneous base learners** (LSTM, CNN, CNN-LSTM,
  CNN-GRU) and **out-of-fold XGBoost stacking**. Training six NexuML
  branches together is not the same algorithm. The stacked method requires
  an independently implemented out-of-fold/meta-learning protocol.
  Source: [arXiv:2608.27940](https://arxiv.org/abs/2608.27940).

## Remaining publication-fidelity gaps

1. Reimplement the missing architectures and their reported architecture
   hyperparameters as NexuML library components.
2. Select the published paper's official test-label convention after
   cross-checking its reported metric computations; both are now available.
3. Replicate reported train/validation selection, random seeds, number of
   experiments, best-checkpoint and preprocessing procedures.
4. Validate reported metrics only after native NexuML train/test runs.

The available protocol scenarios are **parameter-matched controls**, not
paper reproductions. Their output should not be used to claim published
performance has been replicated.
