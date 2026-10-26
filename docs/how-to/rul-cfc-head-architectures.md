# NEX-233: Transformer, Mamba-style SSM and CfC RUL readout comparison

This study isolates the **supervised RUL readout architecture**. All 18
experiments share the frozen **FD001 seed-42 PatchFormer** and, in the
Flow arm, the frozen **FD001 seed-42 CfC Flow**. The upstream encoder and
Flow are *not* trained as part of these runs.

The Flow variants are not three different Flow backbones: they are
three different **heads consuming the same CfC Flow representation**.

## Architecture controls

Each readout consumes the same `[batch,30,256]` features, formed from
the original 128-wide frozen PatchFormer embedding concatenated with
either the frozen CfC displacement or 128 zeros.

The shared components are:
- Linear input projection from 256 to 64.
- Learned 30-token positional embedding.
- One temporal backbone, selected per trial.
- Shared LayerNorm, token attention pooling, and `64 -> 32 -> 1`
  GELU/dropout regression MLP.
- `130 * sigmoid` output in cycles.

Compared temporal backbones:
- **Transformer**: the unmodified Gen2 one-layer 4-head Transformer
  encoder with 128-wide feedforward sublayer.
- **Mamba-style**: the repository's `SelectiveSSMBlock` with `state_dim=8`,
  `expand=2`, and causal depthwise convolution kernel 4. This is a
  *pure-PyTorch unfused Mamba-style implementation*, not the optimized
  `mamba_ssm` package or a claimed faithful throughput reproduction.
- **CfC**: the repository's `_CfCSequence`, one residual CfC layer with
  continuous-time gated interpolation. Elapsed token time is fixed at
  `1/29`, not inferred from engine-cycle timestamps.

The backbone parameter counts are **not identical**. The validation
summary records the complete trainable readout parameter count for
each architecture. The study controls width and shared readout stages,
but is not a parameter-matched or compute-matched comparison.

All trials use the preselected Gen2 settings: hidden width 64, dropout
0.1, **Adam at 0.001**, ordinary window-weighted MSE, batch size 64,
max 150 epochs and early stopping patience 20. Both train and validation
splits use the original engine-disjoint FD001 protocol; test labels
never affect readout fitting or architecture ranking.

## Experiments: 3 × 2 × 3 = 18

- Readout backbones: `transformer`, `mamba`, `cfc`.
- Feature arms: `encoder`, `flow`.
- Independent **readout** seeds: `42`, `43`, `44`.

All trials use the same frozen upstream checkpoints (upstream seed 42),
so only the supervised readout initialization and its batch RNG differ
across head seeds. This does not measure independent upstream
representation training variability.

## Run from repository root

First pull the NEX-233 branch and verify library tests:

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_head_architectures.py -v
```

Full 18-run study:

```bash
set -o pipefail
bash scripts/launch_cfc_rul_head_architectures_fd001.sh \
  2>&1 | tee .experiments-cfc-rul-heads.log
```

The launcher uses only native `nexuml resolve` and `nexuml train`
for training. It writes independent configurations, best checkpoints,
validation curves and engine audits for every cell. It does **not**
automatically pull, retrain upstream weights, or change NexuML core.

One example:

```bash
nexuml resolve maintenance-rul-cfc-head-mamba-flow-s42 \
  --output .experiments/rul/head-architectures/mamba-flow-s42.yaml
nexuml train --config .experiments/rul/head-architectures/mamba-flow-s42.yaml
```

To run the six architecture/arm cells for one seed:

```bash
HEAD_SEEDS="42" bash scripts/launch_cfc_rul_head_architectures_fd001.sh
```

Use `HEADS` and `ARMS` to restrict the grid, e.g.
`HEADS="mamba cfc"`. Partial architecture grids do not produce an
invalid complete summary.

When all architectures and arms for the requested seeds finish, the
launcher produces:

```text
.experiments/rul/head-architectures/summary.json
```

It can also be generated again without rerunning training:

```bash
python -m nexuml_library.evaluation.maintenance.rul_head_architecture_summary \
  --head-seeds 42 43 44 \
  --output .experiments/rul/head-architectures/summary.json
```

## Interpretation and safeguards

The summary ranks architectures by **mean best fitting-epoch
validation-window MSE** across head seeds. For each readout it reports:

- Validation Flow/encoder MSE and between-head-seed variation.
- Paired Flow-minus-encoder MSE reduction.
- Worst validation-engine stress group and engine-mean validation MSE.
- The train/validation generalization gap and post-best deterioration
  per run.
- Readout parameter count and exact frozen upstream SHA-256 checks.

The 4 stress groups are deterministic partitions of **one existing
validation split**, not 4 independently retrained folds.

The native NexuML train lifecycle currently also evaluates the official
test split after restoring best weights. The architecture summary
**does not read test predictions or scores**. Since FD001 test
failures were already inspected, results on this dataset must be
labeled exploratory; do not tune model configurations using its test
performance. Independent FD003 validation and final test evaluation
are needed to assess transferability.


## FD003 replication: fixed protocol, separate upstream weights

The FD001 architecture study selected CfC as the lowest-mean-validation-MSE
Flow readout, with Transformer close behind. FD003 repeats the **entire
predeclared three-backbone comparison**, rather than selecting only one
FD001 winner or changing its training settings.

It uses the identical head architectures, widths, dropout, Adam learning
rate, window MSE, epoch budget, patience and seeds. Differences from FD001
are only the **dataset-specific** sensor protocol, native FD003 asset splits
and correspondingly trained FD003 upstream weights. No FD001 pretrained
encoder/Flow checkpoints or feature-normalization statistics are carried
into FD003.

The FD003 run dependencies are:

1. The previously registered native
   `maintenance-rul-controlled-fd003-patchformer-s42` scenario, which
   trains the FD003 PatchFormer and writes its best checkpoint manifest.
2. The previously registered native
   `maintenance-rul-patchformer-sequence-cfc-fd003-s42` scenario, which
   prepares FD003 healthy/faulty sequence endpoints and learns the
   FD003 CfC Flow, selected using validation transport SWD.

The FD003 launcher checks for these artifacts and runs the corresponding
`nexuml resolve` / `nexuml train` commands **only when absent**. The
scenarios enforce checkpoint identity, train-only feature normalization
and official engine-disjoint splits. Existing valid upstream artifacts
are reused across all 18 supervised-head trials.

### Execute the FD003 comparison

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_head_architectures.py -v

set -o pipefail
bash scripts/launch_cfc_rul_head_architectures_fd003.sh \
  2>&1 | tee .experiments-cfc-rul-heads-fd003.log
```

All 18 registered FD003 scenarios use the form
`maintenance-rul-cfc-head-<transformer|mamba|cfc>-<encoder|flow>-fd003-s<42|43|44>`.
A single run can be invoked with the native CLI:

```bash
nexuml resolve maintenance-rul-cfc-head-cfc-flow-fd003-s42 \
  --output .experiments/rul/head-architectures/fd003/cfc-flow-s42.yaml
nexuml train --config .experiments/rul/head-architectures/fd003/cfc-flow-s42.yaml
```

The latter requires the two FD003 seed-42 upstream runs already described.

You can also restrict the full launcher with `HEAD_SEEDS`, `HEADS`, or
`ARMS`. On a complete grid, it writes a strictly **validation-only**
summary to:

```text
.experiments/rul/head-architectures/fd003/summary.json
```

To recalculate without training, use:

```bash
python -m nexuml_library.evaluation.maintenance.rul_head_architecture_summary \
  --subset FD003 --head-seeds 42 43 44
```

The report checks that every readout uses exactly the same FD003 frozen
encoder, endpoints and Flow checkpoint; it rejects mismatched validation
assets, missing run cells and best-checkpoint epoch differences.

**Test-set safeguard:** the summary never reads held-out test predictions.
Native NexuML `train` currently evaluates the official test split after
fitting. Treat those audits as unavailable for model selection; review
held-out FD003 performance only *after* the head configuration has been
selected using FD003 validation. FD001's already inspected test failures
are exploratory and cannot be used for selecting a FD003 checkpoint.

The FD001 summary and scenarios remain unchanged and accessible at
`.experiments/rul/head-architectures/summary.json`.
