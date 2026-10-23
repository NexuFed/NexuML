# NEX-233 — Full-sequence Flow Matching for RUL estimation

This experiment tests whether retaining PatchFormer's complete temporal DPAN
sequence enables a trained **Transformer, Conformer or CfC latent flow** to
contribute predictive information about remaining useful life (RUL).

The previous flow-time and mean-pooled flow-state RUL experiments remain
unchanged. These are *new*, fully separate supervised readout scenarios.

## Architecture and controlled comparison

For a 30-cycle C-MAPSS history `x`, the frozen PatchFormer DPAN produces
`Z = DPAN(x) in R^[30,128]`. The same *per-position* standardization,
obtained from the matching sequence Flow's training healthy/faulty endpoints,
is applied to all samples without consulting official test labels.

For every pretrained sequence Flow architecture, two models receive a
**matched** 256-channel temporal input:

```text
                                  frozen PatchFormer DPAN
                                        [30,128]
                                            |
                                  train-endpoint normalizer
                                            |
                                standardized full sequence Z
                                       /       \
                       encoder-only           frozen SequenceFlow
                         [Z, 0]                phi_1(Z), Euler 50 steps
                            |                   |
                            |                   [Z, phi_1(Z)-Z]
                            |                          |
                            +-----------+--------------+
                                        |
                                 *same* trainable
                            Transformer temporal-attention
                             + learned attentive pooling
                                        |
                               130 * sigmoid(MLP)
                                        |
                                    RUL cycles
```

The new model **never calls PatchFormer's old FAN/RUL prediction head**.
The DPAN and all pretrained flow vector fields are frozen; only the new
RUL readout is trained. The encoder-only and flow-aware heads use the same
initialization seed, architecture, 256-input channel layout, optimizer,
training/validation splits, and checkpoint-selection protocol.

The encoder-only control uses exactly zero flow displacement channels.
It still requires the immutable full-sequence *endpoint artifact* to
guarantee identical encoder weights and train-only normalization; it does
**not** load a Flow checkpoint.

These controls isolate flow's additional information from improvements due
solely to retraining a new supervised temporal prediction head.

## Run FD001 seed 42

The original PatchFormer and all three full-sequence Flow checkpoints
already exist from the preceding experiments.

Run all six RUL experiments from the repository root:

```bash
git pull
bash scripts/launch_patchformer_sequence_rul_fd001_s42.sh
```

The launcher executes these registered `nexuml resolve` + `nexuml train`
scenarios sequentially:

```text
maintenance-rul-patchformer-sequence-transformer-encoder-fd001-s42
maintenance-rul-patchformer-sequence-transformer-flow-fd001-s42
maintenance-rul-patchformer-sequence-conformer-encoder-fd001-s42
maintenance-rul-patchformer-sequence-conformer-flow-fd001-s42
maintenance-rul-patchformer-sequence-cfc-encoder-fd001-s42
maintenance-rul-patchformer-sequence-cfc-flow-fd001-s42
```

Each scenario writes:
- The native best-validation-loss RUL checkpoint
- An unchanged 100-engine held-out CSV with capped/raw RUL metrics
- A `*.geometry.json` manifest containing the frozen encoder and, if
  applicable, flow checkpoint SHA-256 plus the upstream endpoint digest

On completion the launcher writes
`.experiments/rul/sequential/rul-comparison-fd001-s42.json`. To rerun
the report without retraining:

```bash
python -m nexuml_library.evaluation.maintenance.rul_sequence_flow_summary \
  --subset FD001 --seed 42 \
  --output .experiments/rul/sequential/rul-comparison-fd001-s42.json
```

The reporter calculates capped and official/raw RMSE, MAE, NASA score,
and positive-error bias. It compares each **flow-only temporal readout
against its matched encoder-only control** as well as the original
PatchFormer and counts which official engines have lower squared error.

If a sequence Flow directory contains several candidate checkpoints
from repeated runs, the source selects the checkpoint with the
lowest *saved Lightning validation transport SWD*. It rejects candidates
without a trustworthy score, does not pick `last.ckpt`, and never consults
RUL test labels. Do not manually mix checkpoints trained from different
encoder or endpoint preparation settings.

## FD003 and seeds 43/44

The repository registers 36 scenarios: three architectures × two
readout arms × FD001/FD003 × seeds 42, 43 and 44. All use the same
native RUL evaluation protocol and official engine-disjoint C-MAPSS
splits. The matching pretrained controlled PatchFormer and sequential
Flow must exist for the requested subset/seed.

For example:

```bash
nexuml resolve maintenance-rul-patchformer-sequence-cfc-flow-fd003-s43 \
  --output .experiments/rul/sequential/cfc-flow-fd003-s43.yaml

nexuml train --config .experiments/rul/sequential/cfc-flow-fd003-s43.yaml
```

FD001 held-out results have already informed development; treat further
FD001 comparisons as exploratory, and keep FD003 untouched during tuning.

## Source and verification

Library additions:

- `data/maintenance/rul_sequence_flow.py`: full frozen DPAN tokens and
  optional sequence displacement, no time pooling and no FAN prediction
- `layers/model/rul_sequence_flow.py`: identical learned-attention
  per-token RUL model for both arms
- `evaluation/maintenance/rul_sequence_flow_checkpoint.py`: strict
  trained Flow restore and checkpoint provenance
- `evaluation/maintenance/rul_sequence_flow_summary.py`: strict paired
  native engine-audit comparison
- `scenarios/maintenance/rul_sequence_flow.py`: 36 native scenarios
- `scripts/launch_patchformer_sequence_rul_fd001_s42.sh`: six-run
  launcher with fail-fast behavior
- `tests/library/test_rul_sequence_flow.py` and
  `test_rul_sequence_flow_summary.py`: shape, isolation, provenance,
  registration, end-to-end source and audit controls

Verify locally:

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_sequence_flow.py \
       tests/library/test_rul_sequence_flow_summary.py \
       tests/library/test_sequence_flow.py \
       tests/library/test_rul_advanced_native_cli.py \
       tests/_registry -v
```

Source changes are published on origin, but local Ruff/ty/pytest and
GPU training have not been executed by the assistant.
