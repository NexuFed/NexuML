# NEX-233: Standalone PatchFormer latent-Flow RUL estimation

## Motivation and scientific distinction

The old flow-time-only experiment inferred RUL from one projected transport
coordinate and reached approximately 26-cycle RMSE on FD001. The hybrid
instead improved the supervised PatchFormer by only 0.034 cycles in capped
RMSE versus a matched baseline-only calibrator (FD001 seed 42).

This new experiment asks a **different question**: can the trained flow
provide a richer representation that predicts RUL **without using any
output of PatchFormer's original RUL regression head**?

The encoder is still a **supervised-pretrained PatchFormer**, so this is
not an unsupervised-only or a flow-from-raw-sensors benchmark. We reuse
its learned *embeddings*, not its final RUL prediction.

## Architecture

```text
C-MAPSS sensors [30 cycles, 14 or 15 sensor channels]
                  |
      frozen PatchFormer DPAN
         [no FAN/RUL prediction]
                  |
       z = pooled embedding [128]
                  |
        train-only standardization
                  |
         frozen NEX-196 Flow
       phi_1(z) via 50 Euler steps
                  |
    [ z , phi_1(z)-z , t*(z), d*(z) ]
     128        128        1    1
                  |
        258-input RUL MLP
    [Linear(258,128), GELU, Dropout(0.1),
     Linear(128,64), GELU, Linear(64,1)]
                  |
         130 * sigmoid(output)
                  |
          Predicted RUL cycles
```

The flow-state model has **no supervised PatchFormer RUL prediction** as
an input, residual shortcut, teacher output or gating condition.
Only its new MLP is trainable. All upstream weights are frozen and
the source checks that their subset, seed and SHA-256 provenance match.

**Matched encoder-only control:** The same 258-input head receives
`[z, 0_128, 0, 0]`, isolating the value of flow displacement,
flow time and distance. If the matching flow endpoint feature artifact
exists, its *train-endpoint-only* normalization is reused exactly.
Otherwise, the encoder-only source computes z-score statistics from
training engines only and does **not** require any flow checkpoint.
This avoids unfairly comparing standardized flow states to raw encoder
features.

The registered implementations are:

- `AutoPatchFormerFlowRULFeatures`, mode `flow_only` or `encoder_only`
- `PatchFormerFlowStateRULHead`, a normal NexuML `LayerDefinition`
- `patchformer_flow_state_rul`, producing registered `ScenarioSpec`s

Both use native `nexuml resolve` and `nexuml train`, early stopping on
validation loss, and exactly the same official capped/raw per-engine RUL
evaluators/audits as the earlier PatchFormer experiments.

## FD001 seed 42 — train without manual checkpoint paths

The flow-state source discovers the existing trained PatchFormer and
latent Flow Matching models automatically; your earlier flow experiment
has already produced those prerequisites.

```bash
git pull

nexuml resolve maintenance-rul-patchformer-flow-state-fd001-s42 \
  --output .experiments/rul-flow-state-fd001-s42.yaml &&
nexuml train --config .experiments/rul-flow-state-fd001-s42.yaml

nexuml resolve maintenance-rul-patchformer-encoder-only-fd001-s42 \
  --output .experiments/rul-encoder-only-fd001-s42.yaml &&
nexuml train --config .experiments/rul-encoder-only-fd001-s42.yaml
```

Both return official capped and original/raw RMSE, MAE and NASA scores,
plus an exact 100-engine audit. The trained head checkpoints and geometric
provenance appear under `.experiments/rul/checkpoints/` and
`.experiments/rul/audits/`.

After both runs, generate the **paired, engine-level comparison**:

```bash
python -m nexuml_library.evaluation.maintenance.rul_flow_state_summary \
  --subset FD001 --seed 42 \
  --output .experiments/rul/audits/flow-state-fd001-s42.json
```

The comparison rejects mismatched test engines, capped/raw labels,
PatchFormer checkpoint identities, or accidentally using the pretrained
PatchFormer RUL head. It reports paired 95% engine-bootstrap intervals
for flow-state vs encoder-only and flow-state vs PatchFormer, alongside
per-engine differences in squared prediction error.

## Direct flow-feature permutation check on the *existing* hybrid

To see whether the previous hybrid actually **uses** its two learned
flow coordinates, restore the frozen trained hybrid and shuffle
`[t*,d*]` jointly across official test engines, keeping its
PatchFormer prediction unchanged:

```bash
python -m nexuml_library.evaluation.maintenance.rul_flow_permutation \
  --subset FD001 --seed 42 --repeats 100 \
  --output .experiments/rul/audits/hybrid-flow-shuffle-fd001-s42.json
```

This command checks that the original reconstructed hybrid predictions
reproduce the *existing* official test audit to within 0.005 cycles
before applying any permutations. It does not update the hybrid weights
or choose a different checkpoint. It reports RMSE/MAE/NASA scores
before and after the shuffled flow coordinates.

A negligible change after shuffling would indicate little dependence
on those features. An increase in error would indicate test-time
sensitivity, but not by itself prove causal physical degradation
ordering.

## FD003 and paired seeds 42–44

Run the matching supervised PatchFormer and latent Flow models before
the flow-state model for each new subset and seed. The encoder-only
control needs only the supervised encoder checkpoint.

The 12 registered scenarios follow these patterns:

```text
maintenance-rul-patchformer-flow-state-fd001-s42/s43/s44
maintenance-rul-patchformer-encoder-only-fd001-s42/s43/s44
maintenance-rul-patchformer-flow-state-fd003-s42/s43/s44
maintenance-rul-patchformer-encoder-only-fd003-s42/s43/s44
```

For a complete FD003 seed 42 example:

```bash
nexuml resolve maintenance-rul-controlled-fd003-patchformer-s42 \
  --output .experiments/fd003-patchformer-s42.yaml &&
nexuml train --config .experiments/fd003-patchformer-s42.yaml

nexuml resolve maintenance-rul-patchformer-flow-fd003-s42 \
  --output .experiments/fd003-latent-flow-s42.yaml &&
nexuml train --config .experiments/fd003-latent-flow-s42.yaml

nexuml resolve maintenance-rul-patchformer-flow-state-fd003-s42 \
  --output .experiments/fd003-flow-state-s42.yaml &&
nexuml train --config .experiments/fd003-flow-state-s42.yaml

nexuml resolve maintenance-rul-patchformer-encoder-only-fd003-s42 \
  --output .experiments/fd003-encoder-only-s42.yaml &&
nexuml train --config .experiments/fd003-encoder-only-s42.yaml
```

Replace `s42` with `s43` and `s44` to replicate. Do **not**
tune on FD003 official test scores and then claim its held-out
performance is an independent confirmation.

## Degradation ordering already recorded

Original `*.geometry.json` manifests include train/validation
flow-time-to-RUL Pearson correlation and within-validation-engine
progression statistics. Inspect these **without using official test
labels** before interpreting a flow coordinate as physical degradation.

Unlike a point estimate inferred from `t*` alone, the new head also
uses the full 128-dimensional flow displacement field, which may retain
degradation-relevant information lost by nearest-trajectory projection.

## Checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_flow_state.py \
       tests/library/test_rul_flow_calibration.py \
       tests/library/test_rul_flow_ablation.py \
       tests/library/test_rul_advanced_native_cli.py \
       tests/_registry -v
```

Tests ensure the old FAN/RUL regression head is never invoked while
building either flow-state or encoder-only features. The code is on
origin; checks and RUL training have not been run by the assistant.
