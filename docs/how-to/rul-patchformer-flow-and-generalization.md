# NEX-233: controlled PatchFormer generalization and latent Flow Matching

This extends the successful FD001 PatchFormer experiment with **paired,
test-safe replication** and a separate frozen-embedding Flow Matching
experiment. It uses only NexuML `LayerDefinition`, `DataSourceDefinition`,
`ScenarioSpec`, Lightning callbacks and `nexuml resolve/train`.

## Research questions

1. Does the PatchFormer architecture beat the safety-aware LSTM over three
   **matched training seeds** (42, 43, 44), on both FD001 and FD003?
2. Are gains robust to holding architecture comparisons to a **common 30-cycle
   window, optimizer and training configuration**?
3. Can NEX-196 Flow Matching transport **frozen PatchFormer latent features**
   from high-RUL to low-RUL regimes, as measured by validation sliced
   Wasserstein distance (SWD), without using official test RUL labels?

The FD001 official test residuals informed this research. Those observations
make new FD001 comparisons exploratory. FD003 must not be used to tune this
model and then presented as a new independent confirmation.

## Controlled benchmark protocol

Each subset × seed shares **one cached dataset preparation** among models:

- Official C-MAPSS FD001 uses 14 Goel-selected sensors; FD003 uses 15,
  including sensor 6. There is no cross-subset feature mixing.
- Window length 30 cycles, stride 1, zero-padding for short official test
  runs, training-only full windows, train-asset-only z-score and 130-cycle
  cap. Official test labels are also evaluated **uncapped**.
- The standard engine-disjoint 80/20 train/validation split, determined by
  the shared seed. The 100 official test engines are held out.
- Adam learning rate 0.001, batch 64, maximum 200 epochs, validation
  patience 20 and best-`val/loss` checkpoint restoration for all four.
- PatchFormer and DMM-JA share hidden width 64 and 0.1 dropout; the
  safety LSTM retains its paper-style 32-hidden-state/0.5-dropout
  architecture; RUL-Mamba uses width 64/0.1 dropout and a 30-cycle
  input rather than the original short-window configuration.
- The safety LSTM alone uses the **prespecified asymmetric squared
  loss** with positive-error weight 2; the other models use direct
  RUL MSE. This is a *controlled training protocol*, not an
  architecture-only ablation of equal objectives.

For each subset/seed, run these four variants:
`patchformer`, `dmm-ja`, `safety-lstm`, `rul-mamba`.
All variant checkpoint and audit destinations are distinct.

## Native run all 24 scenarios

```bash
set -euo pipefail
mkdir -p .experiments/rul/controlled

for subset in fd001 fd003; do
  for seed in 42 43 44; do
    for kind in patchformer dmm-ja safety-lstm rul-mamba; do
      name="maintenance-rul-controlled-$subset-$kind-s$seed"
      config=".experiments/rul/controlled/$name.yaml"
      nexuml resolve "$name" --output "$config"
      nexuml train --config "$config"
    done
  done
done
```

For just the first FD003 experiment:

```bash
nexuml resolve maintenance-rul-controlled-fd003-patchformer-s42 \
  --output .experiments/rul/controlled/patch-fd003-s42.yaml &&
nexuml train --config .experiments/rul/controlled/patch-fd003-s42.yaml
```

The C-MAPSS source auto-downloads (as in existing Goel scenarios), if
not cached. Native engine-audit CSVs and best-checkpoint provenance
are written under `.experiments/rul/audits/`.

**Pairwise aggregation after all runs finish** (no training/test
tuning in this utility):

```bash
python -m nexuml_library.evaluation.maintenance.rul_controlled_summary \
  --audit-root .experiments/rul/audits \
  --output .experiments/rul/controlled/study-summary.json
```

The reporter enforces the complete 100-engine holdout, matching capped
and raw test labels over **all models and seeds**, and reports per-seed
RMSE, MAE, mean positive RUL error, NASA score, means, and signed
differences versus the safety-aware LSTM. Three training seeds are
**not independent test populations**, so refrain from unqualified
significance claims.

## Native PatchFormer latent Flow Matching

The frozen embeddings come from the shared **supervised** PatchFormer
backbone after its variable-embedding and dual patch attention stages,
**before the final feature-wise attention/regression head**. Temporal
mean pooling gives one 128-dimensional latent vector per causal
30-cycle sensor window. The trained backbone is never updated by the
flow objective.

The newly registered dataset source
`AutoPatchFormerFlowEndpoints`:

1. Reuses the same C-MAPSS prepared data for the subset/seed;
   downloads data if missing.
2. Finds the best-validation checkpoint in native NexuML audit manifests
   for the matching subset and seed; requires no environment variable
   or manually selected checkpoint file. It validates every
   backbone tensor shape.
3. Samples training and validation endpoints separately:
   **healthy RUL ≥ 100 cycles** and **degraded RUL ≤ 30 cycles**.
   Pairing is deterministic and independent within each split.
4. Extracts frozen pre-regression feature vectors, then standardizes
   using the **training endpoint embeddings only**.
5. Saves an immutable `nex233-features-v1` [N,2,128] endpoint artifact
   containing split identifiers, actual source asset IDs, checkpoint
   SHA-256, training-only feature mean/std and threshold metadata.
   **No official test engine contributes features or labels** to
   endpoint preparation or validation.

The flow itself is the **existing NexuML**:
`PairedLinearFlowPath` → `Flow(TimeConditionedVectorField,
EulerIntegrator)` → `FlowMatchingLoss`.

The native `DistributionalTransportValidation` callback measures
**validation healthy-to-degraded SWD** before and after transport,
selecting the lowest `val/transport_swd` checkpoint. This follows
the earlier MIMII/FISHER feature-ablation measurement, using a
different frozen frontend and endpoint definition. Flow time is a
transport parameter, **not physical RUL**.

### Run FD001 seed 42 — no environment variables or manual checkpoint selection

The flow source resolves its upstream checkpoint automatically from
NexuML's own best-checkpoint audit manifests at training time:

1. Prefer `.experiments/rul/audits/rul-FD001-controlled-patchformer-seed42.checkpoint.json`.
2. If that model has not been trained, reuse the original already-trained
   FD001 seed-42 PatchFormer from
   `.experiments/rul/audits/rul-cmapss-FD001-patchformer-direct-rul-adaptation-seed42.checkpoint.json`.

Both model variants have compatible FD001 sensor preprocessing and
128-dimensional DPAN embeddings. The selected checkpoint must exist,
match the subset/seed run directory and contain all required backbone
weights. No checkpoint is needed for `nexuml resolve`; the data
source discovers and validates it during the ordinary `nexuml train`
preparation.

For an existing PatchFormer run, use only:

```bash
nexuml resolve maintenance-rul-patchformer-flow-fd001-s42 \
  --output .experiments/patch-flow-fd001-s42.yaml &&
nexuml train --config .experiments/patch-flow-fd001-s42.yaml
```

If neither best-checkpoint manifest exists, first train the matching
PatchFormer via NexuML (once), then rerun the flow commands:

```bash
nexuml resolve maintenance-rul-controlled-fd001-patchformer-s42 \
  --output .experiments/patch-fd001-s42.yaml &&
nexuml train --config .experiments/patch-fd001-s42.yaml
```

For other subsets and seeds, the corresponding
`maintenance-rul-controlled-<subset>-patchformer-s<seed>` run is
required. No shell-exported checkpoint path is necessary:

```text
maintenance-rul-patchformer-flow-fd001-s43
maintenance-rul-patchformer-flow-fd001-s44
maintenance-rul-patchformer-flow-fd003-s42
maintenance-rul-patchformer-flow-fd003-s43
maintenance-rul-patchformer-flow-fd003-s44
```

The data artifact records the automatically selected upstream run,
checkpoint path and SHA-256 digest for auditability.

**Trust boundary:** A selected checkpoint is a trusted local Lightning
artifact. It is deserialized in a non-weights-only mode because
Lightning callback/optimizer metadata may include non-tensor values.
Never load untrusted checkpoints.

### Limitations

These flow runs do **not yet** attach an RUL head to flow time or
show improvements to the supervised PatchFormer RUL score. The
primary endpoint metric is validation transport SWD. The checkpoint
matching and fitted embedding statistics are recorded to make
upstream-model provenance reproducible. Endpoint stages are based
on training/validation RUL labels, so the resulting flow is
supervised at the distribution-construction level, even though
the velocity objective is unconditional.

## Verification

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_patchformer_study.py \
       tests/library/test_rul_advanced.py \
       tests/library/test_rul_benchmarks.py \
       tests/_registry -v
```

The study tests verify paired dataset artifacts, FD003 sensor 6,
checkpoint independence, native YAML round-trips, original flow-path
gradients, split-safe frozen embeddings and the reporter's rejection
of mismatched official test targets.

The implementation does not modify NexuML's core structure.
