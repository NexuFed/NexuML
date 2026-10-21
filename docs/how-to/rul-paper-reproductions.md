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

## Paper benchmarks not represented as exact NexuML scenarios

- **Li, Ding & Sun (2018), deep CNN:** the reported FD001 RMSE of 12.61
  involves their specific deep convolution architecture. NexuML's RUL
  forecaster currently offers an MLP, LSTM, TCN, Transformer and TSMixer,
  **not** that CNN. A hyperparameter-only rename would be misleading.
  Source: [Reliability Engineering & System Safety, 172, 1–11](https://doi.org/10.1016/j.ress.2017.11.021).
- **Goel, Galchar & Kanu (2026):** the paper reports a **single-layer LSTM**
  at FD001 RMSE 14.93, but a matching complete preprocessing/optimizer
  configuration is not established here. The existing `maintenance-rul-lstm`
  uses 3 layers, so it is **not** the same experiment.
  Source: [arXiv:2604.27234](https://arxiv.org/abs/2604.27234).
- **Hossain et al. (2026), stacking ensemble:** the published FD001
  RMSE 9.989 uses **four heterogeneous base learners** (LSTM, CNN, CNN-LSTM,
  CNN-GRU) and **out-of-fold XGBoost stacking**. Training six NexuML
  branches together is not the same algorithm. The stacked method requires
  an independently implemented out-of-fold/meta-learning protocol.
  Source: [arXiv:2608.27940](https://arxiv.org/abs/2608.27940).

## Remaining publication-fidelity gaps

1. Reimplement the missing architectures and their reported architecture
   hyperparameters as NexuML library components.
2. Preserve raw official NASA test RUL offsets separately from capped
   training targets. Current NexuML RUL preparation caps both.
3. Add the asymmetric NASA exponential score and document target conventions.
4. Replicate reported train/validation selection, random seeds, number of
   experiments, best-checkpoint and preprocessing procedures.
5. Validate reported metrics only after exact native NexuML train/test runs.

The available protocol scenarios are **parameter-matched controls**, not
paper reproductions. Their output should not be used to claim published
performance has been replicated.
