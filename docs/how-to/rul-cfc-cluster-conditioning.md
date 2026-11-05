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
