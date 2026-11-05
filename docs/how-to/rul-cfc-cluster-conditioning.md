# NEX-233: inferred working/failure-state conditioning for CfC RUL

## Scientific hypothesis

On NASA C-MAPSS FD003, operating-state and prospective failure-state
embeddings may contain useful regime-specific information not captured by
the unconditional CfC RUL regressor. At inference, **neither state cluster is
known**. Both conditioning signals therefore come from *predicted soft
memberships* of the observed 30-cycle sensor history. The model never
receives the oracle training-cluster ID as an input.

This is a **controlled follow-up** to the 41-job NEX-233 CfC campaign. It
does not modify the earlier runs or claim to identify physically annotated
HPC/fan failure modes. Clusters are unsupervised pseudo-labels, not NASA
fault-mode ground truth.

## Leakage-free cluster preparation

The native `AutoClusterConditionedCMapSSRULWindows` source starts from
the exact existing FD003 seed-42 controlled C-MAPSS prepared windows:

- 15 selected sensors, 30 cycles, stride one, capped RUL at 130 cycles;
- 80 *training* engines, 20 separate validation engines, official test engines
  remain untouched by fitting and model selection;
- the same trusted frozen seed-42 PatchFormer and its existing train-endpoint
  normalization artifact; source/checkpoint SHA-256 are checked;
- for each *training* engine, pool its first four and last four **complete**
  sensor windows into two 128-dimensional PatchFormer representations;
- standardized engine-balanced K-means with 3 healthy/working clusters and
  2 terminal/failure clusters, fitted **only on training engines**.

The cluster artifact records normalized centroids, statistics, train-engine
teacher IDs and provenance. Validation/test engines receive auxiliary labels
`-1` and are never used to fit clusters. The complete future trajectory of
an engine is used only to construct a **training-only** failure pseudo-label.

For conditioned models, the data source prepares the cluster artifact on
first use. For reproducibility and to avoid concurrent first-writer races,
prepare it **once before launching parallel training**. On the submitting
workstation, use the absolute Prisma experiment root; **do not assume** the
worktree's `.experiments` directory is symlinked to persistent storage:

```bash
cd /workspaces/NexuFed/NexuML-predictive
git pull --ff-only
export PYTHONPATH="$PWD/src:$PWD/library/src"

python - <<'PY'
from pathlib import Path
from nexuml_library.data.maintenance.rul_cluster_conditions import (
    AutoClusterConditionedCMapSSRULWindows,
)

experiments = Path("/mnt/juicefs_prisma/models/nexuml/.experiments")
prepared = experiments / "rul/prepared"
source = AutoClusterConditionedCMapSSRULWindows(
    subset="FD003",
    seed=42,
    root="/mnt/juicefs_prisma/data/cmapss",
    experiment_root=str(experiments),
    artifact_path=str(prepared / "cmapss-FD003-seed42-controlled-w30-s42.pt"),
    cluster_artifact=str(prepared / "cmapss-FD003-seed42-cfc-conditional-clusters-v1.pt"),
)
source.build()
assert Path(source.cluster_artifact).is_file()
print("Train-only FD003 cluster artifact prepared:", source.cluster_artifact)
PY
```

The explicit root is used for the prepared input, cluster output, frozen
PatchFormer audit/checkpoint, and sequence-flow endpoint normalizer. This
works with audit manifests created in Pods whose storage mountpoint differs
from the host; the selected checkpoint is mapped to the corresponding
run-specific checkpoint directory on Prisma.

The source requires existing FD003 PatchFormer checkpoint and matching
`rul-FD003-patchformer-sequence-cfc-flow-seed42-endpoints.pt`. It will fail
rather than silently substituting a different pretrained encoder.

## Four controlled experiments

Each arm has independent selected-checkpoint, curve and engine-audit paths.
All four arms share the *same* original `all_no_trend` CfC architecture,
upstream encoder, data split, optimizer, and paired seeds 42, 43 and 44.

| Arm | Condition | Registered scenario (append `-s42`, `-s43` or `-s44`) |
| --- | --- | --- |
| Reference | None | `maintenance-rul-cfc-cluster-fd003-reference` |
| Working | Predicted working-state soft probabilities | `maintenance-rul-cfc-cluster-fd003-working` |
| Failure | Predicted failure-state soft probabilities | `maintenance-rul-cfc-cluster-fd003-failure` |
| Both | Both predicted soft distributions | `maintenance-rul-cfc-cluster-fd003-both` |

The working and failure gates read only the *current* frozen/finetuned
PatchFormer sensor-history embeddings. They combine a learned predictor
with distances to the train-only cluster centroids and use probabilities
to condition the CfC sequence (not a hard cluster ID). The gate's
predicted logits receive auxiliary cross-entropy supervision from
train-engine pseudo-labels; the RUL regression objective and validation
checkpoint selection remain based on RUL loss. The reference has no
gate or pseudo-label supervision.

The current study uses working-cluster CE weight 5 and failure-cluster
CE weight 5 when enabled. These are exploratory, prespecified research
hyperparameters and should not be tuned against official test engines.

## Running native NexuML scenarios

```bash
ruff check .
ruff format --check .
ty check
PYTHONPATH="$PWD/src:$PWD/library/src" pytest \
  tests/library/test_rul_cluster_conditioning.py -v

# Example: run paired seed 42. Repeat for s43 and s44.
for ARM in reference working failure both; do
  NAME="maintenance-rul-cfc-cluster-fd003-${ARM}-s42"
  CONFIG=".experiments/rul/enhanced-cfc/cluster-conditioned/fd003/${ARM}-s42.yaml"
  nexuml resolve "$NAME" --output "$CONFIG"
  nexuml train --config "$CONFIG"
done
```

For GPU-parallel experiments on the original Kueue pool, use the dedicated
immutable-source launcher **instead of** running all twelve local commands:

```bash
# After preparing the cluster artifact on Prisma:
python .k8s/scenarios/submit_rul_cfc_cluster_conditioning.py --dry-run
python .k8s/scenarios/submit_rul_cfc_cluster_conditioning.py
kubectl -n lbecker get jobs -l nexuml-study=rul-cfc-cluster
```

The launcher creates twelve separate Jobs, preserves all prior 41-run results,
uses the previous fixed GHCR CUDA image, excludes GH200 and
`student-worker-1`, requests one GPU per Job, and stages one shared,
content-addressed source snapshot. A missing cluster artifact prevents
submission; it is **never** fitted independently by twelve Kubernetes Jobs.

## Evaluation requirements

Compare validation **evaluation-mode per-window MSE** at each run's own
best `val/loss` checkpoint, paired by readout seed. Use the existing
`SequenceRULGeneralizationDiagnostics` curves at

` .experiments/rul/enhanced-cfc/cluster-conditioned/fd003/ `

and the corresponding audit manifests at
`.experiments/rul/audits/rul-FD003-enhanced-cfc-cluster-<arm>-s<seed>.checkpoint.json`.

Report paired improvements, seed variance, generalization gaps and
worst-validation-engine-group MSE. Inspect whether the gates learned
informative memberships (entropy, cluster counts, calibration), not just
lower RUL loss.

**Do not interpret cluster IDs as physical fault modes** without external
labels. FD003 contains two nominal fault modes but that does not force the
unsupervised clusters to recover those modes. This FD003 validation study
does not establish independent test-set superiority or transfer to FD001.


## Completed-campaign checkpoint gate audit (no retraining)

The gate diagnostic audit operates on the **existing 12 validation-selected
checkpoints** after the training campaign has completed. The three reference
runs supply validation-MSE context; the nine conditioned checkpoints produce
12 gate/split-family reports (working, failure, and both arms). The audit
does not launch Jobs, update model weights, use official test observations,
or compute metrics on the official test split.

Run from the NEX-233 feature worktree with direct Prisma paths (no worktree
`.experiments` symlink required):

```bash
export PYTHONPATH="$PWD/src:$PWD/library/src"
python -m nexuml_library.evaluation.maintenance.rul_cluster_gate_audit \
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \
  --device auto \
  --windows-per-engine 64 \
  --batch-size 64
```

By default the audit deterministically samples up to 64 windows **per
engine**, spread over each engine's history; it checks the original 80
training / 20 validation engine separation and never selects test
windows. A CUDA GPU is used when available; CPU is supported but slower.
The checkpoint loader accepts **only trusted Lightning checkpoints**
generated by the NEX-233 Jobs.

For each enabled gate and split, the report records:

- Engine-balanced soft cluster occupancy, hard predicted engine/window
  occupancy, top-probability confidence and (normalized) entropy.
- Trained additive gate-projection L2 magnitude and its ratio to the
  unconditioned CfC input representation L2 magnitude.
- **Training only**: accuracy, negative log-likelihood and 10-bin expected
  calibration error against train-engine pseudo-labels, label occupancy
  and majority-label baseline.
  This agreement is not physical failure-mode classification accuracy.
- For all 12 runs: the selected-checkpoint epoch, existing validation
  MSE/gap/worst-group metrics, selected checkpoint SHA-256, prepared
  dataset/cluster SHA-256, and explicit test-exclusion provenance.

Outputs are written under:

```text
/mnt/juicefs_prisma/models/nexuml/.experiments/rul/enhanced-cfc/cluster-conditioned/fd003/gate-audit-v1/
  reference-s42.json ... reference-s44.json
  working-s42.json ... working-s44.json
  failure-s42.json ... failure-s44.json
  both-s42.json ... both-s44.json
  summary.csv (3 reference context rows + 12 individual gate rows)
```

A high normalized entropy with nearly uniform probabilities suggests
uninformative membership predictions. Near-unity dominant-engine fraction
suggests hard-collapse to one cluster. High training teacher agreement but
poor validation occupancy or no RUL improvement may signal a mismatch
between pseudo-labels and the RUL objective. A small projection-to-base
ratio suggests the trained conditioning signal is weak. Treat these as
diagnostic evidence, not automatic causal conclusions.


## Post-training inference interventions and age-stratified gate analysis

After running the gate audit, run the **read-only inference-time intervention
study**. It replays the frozen validation-selected checkpoints for all 12 runs,
without any new fitting or Kubernetes Job submission. No `.experiments`
symlink is required; all artifacts are read from the explicit Prisma root.

```bash
git pull --ff-only
export PYTHONPATH="$PWD/src:$PWD/library/src"

ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_cluster_gate_interventions.py -v

python -m nexuml_library.evaluation.maintenance.rul_cluster_gate_interventions \
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \
  --device auto \
  --batch-size 64 \
  --age-windows-per-engine 64
```

The CLI accepts `--seeds 42` or `--arms both` for targeted diagnostics, but
the default audits all 12 runs. It uses a single GPU if available, otherwise
CPU. A partial dry run can verify one selected checkpoint:

```bash
python -m nexuml_library.evaluation.maintenance.rul_cluster_gate_interventions \
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \
  --device auto --arms working --seeds 42
```

**Validation replay guard:** each checkpoint is reconstructed by strictly
loading all saved weights and buffers into the identical `all_no_trend`
PatchFormer, sensor graph, multiscale, segmentation and CfC architecture.
The original prediction is evaluated on **every held-out validation window**
and its window-weighted MSE must match the native selected-checkpoint curve
(within a small numerical tolerance). The intervention audit fails closed on
any mismatch, instead of reporting potentially misleading counterfactuals.

**Interventions:** `original` leaves the trained model unchanged.
`zero_working` and `zero_failure` suppress just the corresponding additive
projection without altering its weights; `zero_both` suppresses both on the
combined arm. The deterministic `shuffle_memberships` intervention replaces
probability trajectories with those of **different validation engines**,
matched by relative position in each engine's history. It uses a fixed
derangement of 20 validation engines (seed 2026); in the combined arm the same
donor is used for both gates, preserving their joint membership distribution.
RUL labels, raw sensor histories, checkpoints and other representations are
left unchanged. No official test-engine observations are evaluated.

The metric is \(\Delta\mathrm{MSE} =
\mathrm{MSE}_{\mathrm{intervention}} -
\mathrm{MSE}_{\mathrm{original}}\), separately as window-weighted and
engine-macro MSE. Positive deltas indicate that suppressing/swapping the
gate **harms this trained head**; it does *not* prove that conditioning is
better than the separately trained reference. Per-engine deltas reveal whether
a small mean masks severe or heterogeneous changes.

**Age study:** each training engine contributes up to 64 evenly spaced
observations, annotated post hoc by its relative lifespan position
(first/middle/last third). For each gate, report engine-macro agreement with
its training-only *engine-level terminal pseudo-label*, negative log
likelihood, mean confidence and normalized entropy, as well as its majority
pseudo-label baseline. High early-life terminal label agreement could reflect
persistent engine identity rather than a real early-life failure precursor;
it is not physical fault-mode accuracy. Validation engines have no
pseudo-label targets and never enter this supervised diagnostic.

Output is written to:

```text
/mnt/juicefs_prisma/models/nexuml/.experiments/rul/enhanced-cfc/cluster-conditioned/fd003/gate-interventions-v1/
  reference-s42.json ... reference-s44.json
  working-s42.json ... working-s44.json
  failure-s42.json ... failure-s44.json
  both-s42.json ... both-s44.json
  validation_interventions.csv
  validation_engine_deltas.csv
  train_age_strata.csv
```

The three CSV files provide the model/seed-level interventions, exact
engine-level RUL differences, and training-only age-stage gate diagnostics.
These are descriptive, exploratory analyses; neither validation-engine
membership shuffling nor training pseudo-label agreement establishes
causality or generalization to unseen failure modes.
