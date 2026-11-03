# NEX-233: independent FD003 CfC confirmation

This branch-only follow-up tests the two strongest candidates in the
validation-only **14-branch, three-seed composite screening**. The composite
shared one optimizer and one checkpoint-selection process per seed; the
follow-up gives each candidate its **own optimizer, native NexuML fit
lifecycle, early stopping and validation-selected checkpoint**.

This is an **exploratory** FD003 replication, not an unseen confirmatory
test. FD003 test performance has already been inspected during development.

## Prespecified experiments

| Configuration | RUL encoder and head | Registered native scenario |
| --- | --- | --- |
| `baseline` | Frozen pretrained PatchFormer + CfC, no optional factors | `maintenance-rul-enhanced-cfc-confirm-fd003-baseline-s42` |
| `finetune` | Supervised PatchFormer fine-tuning, no other optional factors | `maintenance-rul-enhanced-cfc-confirm-fd003-finetune-s42` |
| `all_no_trend` | Fine-tuning, graph, multiscale, segmentation and trajectory penalty; excludes trend projection | `maintenance-rul-enhanced-cfc-confirm-fd003-all_no_trend-s42` |

Replace `s42` with `s43` or `s44` for the other independently trained
readout seeds. All nine registrations are in
`rul_enhanced_cfc_confirmation.py`.

These scenarios are wrappers around the existing standalone
`enhanced_cfc_rul_study`, and retain its 30-cycle × 15-sensor, engine-disjoint
FD003 data protocol (upstream split seed 42), pretrained PatchFormer
(seed 42), train-only endpoint normalization, Adam 0.001, batch size 64,
150-epoch maximum, validation `val/loss` selection and patience 20.
The trajectory penalty is applied **only during training**.

For each run, native callbacks write:

- Independent best and last Lightning checkpoints:
  `.experiments/rul/checkpoints/rul-FD003-enhanced-cfc-confirm-<configuration>-s<seed>/`
- Native best-checkpoint selection audit:
  `.experiments/rul/audits/rul-FD003-enhanced-cfc-confirm-<configuration>-s<seed>.checkpoint.json`
- Full-split evaluation-mode training/validation curves and validation-engine
  diagnostics:
  `.experiments/rul/enhanced-cfc/fd003/confirmation/rul-FD003-enhanced-cfc-confirm-<configuration>-s<seed>.curves.csv`
- Native test per-engine audit for transparency only, never input to selection:
  `.experiments/rul/audits/rul-FD003-enhanced-cfc-confirm-<configuration>-s<seed>.csv`

All confirmation paths are distinct from previous screening, so the composite
results remain unchanged.

## GPU launch on Kubernetes

The CUDA container image is
`ghcr.io/nexufed/nexuml:latest`, replacing the broken
`harbor.ika.rub.de/library/nexuml:latest` transfer. The Kueue `jobs`
queue, `lbecker` namespace, immutable source-digest staging and ephemeral
Python environment are retained. Each Job requests one NVIDIA GPU and
requires AMD64. Scheduling uses the global GPU pool with required hostname
`NotIn` affinity excluding `student-worker-1` and GH200 nodes.
The launcher discovers GH200 nodes from Kubernetes Node names and NVIDIA
GPU labels, rather than hardcoding guessed GH200 hostnames; if none
are identifiable, it stops without submitting Jobs. The previous CUDA
runtime-major node-label restriction was removed; nodes must nevertheless
have drivers compatible with this CUDA image.

Source staging stays on `/mnt/juicefs/job-staging/nexuml-nex233/`,
while checkpoints and prepared data are stored on
`/mnt/juicefs_prisma/models/nexuml/.experiments`, with dataset and
log volumes under `/mnt/juicefs_prisma/data` and
`/mnt/juicefs_prisma/logs`. The Pod accesses these at the same
internal mountpoints as the successful screening campaign.

The FD003 seed-42 upstream PatchFormer's native audit and selected
checkpoint and the frozen train-endpoint normalization artifact must already
be present in Prisma. The Job validates all three before training.

From the NEX-233 worktree on a machine with the source JuiceFS mount,
`rsync`, `kubectl` and cluster access:

```bash
cd /workspaces/NexuFed/NexuML-predictive
git pull --ff-only
pytest tests/library/test_rul_independent_confirmation.py \
       tests/library/test_rul_k8s_staging.py -q

# Inspect generated manifests without submitting.
python .k8s/scenarios/submit_rul_enhanced_cfc_confirm_fd003.py --dry-run

# Submit nine distinct Jobs. No old Jobs or output files are deleted.
python .k8s/scenarios/submit_rul_enhanced_cfc_confirm_fd003.py

# Observe scheduling; no Pod should be on student-worker-1.
kubectl -n lbecker get pods \
  -l nexuml-study=rul-fd003-cfc-confirm -o wide
kubectl -n lbecker get jobs \
  -l nexuml-study=rul-fd003-cfc-confirm
```

To run a **single pilot** first:

```bash
python .k8s/scenarios/submit_rul_enhanced_cfc_confirm_fd003.py \
  --configurations finetune --seeds 42

kubectl -n lbecker logs -f \
  job/nexuml-rul-fd003-cfc-confirm-finetune-s42
```

The launcher skips existing Jobs rather than replacing them. To rerun an
individual Job, delete **only that Job** explicitly and resubmit using
`--configurations` and `--seeds`. The native run will reset its own curves
and may replace its old checkpoints, so archive results first if needed.
The composite screening Jobs use a **different label** and are never touched.

The Job also writes one small `<configuration>-s<seed>.execution.json`
containing the worker name, source SHA-256 and wall time of the
`nexuml train` command. This **completion marker is written only after
training, final validation and testing return successfully**. Retries clear
their own old marker before starting, and the summary rejects an incomplete
or mixed-source campaign. Wall time includes fit, validate and test and is
**hardware-dependent**; it is not a comparable GPU benchmark when Kueue
schedules different nodes.

## Recover five incomplete Jobs using GHCR and the global GPU pool

Five independent confirmation Jobs originally failed to pull the Harbor image:
baseline seed 44, finetune seeds 43 and 44, and all-no-trend seeds 42 and 43.
The four completed Jobs must not be restarted. Kubernetes does not permit
in-place changes to a Job's Pod container image or affinity, so the five
incomplete Jobs must be recreated. The helper supports Jobs already manually
deleted (Kubernetes NotFound) as well as still-Pending Pods (unscheduled or
stuck pulling). It only deletes an existing Job after verifying Pending
status; it refuses to delete completed or actively training Jobs. RBAC
or API-server errors are never treated as a missing Job.

**Do not rerun the standard submit launcher for recovery.** It stages a
snapshot from the current local worktree, which may now differ from the
source used by completed seeds. Instead, use this guarded recovery helper:

```bash
cd /workspaces/NexuFed/NexuML-predictive
git pull --ff-only
pytest tests/library/test_rul_independent_confirmation.py -q

# Preflight: verify the four completed source digests, preserve the original
# manifests and validate the five pending Jobs against the live node pool.
python .k8s/scenarios/restart_rul_enhanced_cfc_confirm_fd003.py

# Recreate ONLY the five still image-pull-blocked Jobs.
python .k8s/scenarios/restart_rul_enhanced_cfc_confirm_fd003.py --apply

kubectl -n lbecker get pods \
  -l nexuml-study=rul-fd003-cfc-confirm -o wide
```

The helper reads the five original files from
`.k8s/scenarios/rendered/<job>.yaml` and generates
`.k8s/scenarios/rendered/global-ghcr-retries/<job>.yaml`. It changes
the image to the GHCR `latest` tag and sets a required hostname
`NotIn` affinity for the discovered GH200 nodes plus
`student-worker-1`. It removes the old CUDA runtime-major label
requirement. It **does not** restage or change the original immutable
experiment source digest, JuiceFS source PVC subPath, run name,
checkpoint paths, datasets, or training configuration.

Before deleting a Job, the helper checks its original source annotation,
its Pod state, the four completed runs' source digests, and the new manifest.
If a Job has already started training or an original manifest is missing,
inspect it rather than forcing a replacement.

The four completed trials used the original Harbor image; the recovered
five use the mutable GHCR `latest` tag. Although the source snapshot
and experiments remain fixed, **the container environments are not identical**.
Record this provenance difference when interpreting the final confirmation
comparison, and verify the new image is accessible to worker nodes.

## Validation-only final report

Once all nine Jobs succeeded, generate the summary from the working tree
with access to the mounted Prisma artifacts:

```bash
ROOT=/mnt/juicefs_prisma/models/nexuml/.experiments/rul/enhanced-cfc/fd003/confirmation

# Runs directly from this worktree even when the active virtualenv
# was created for a different NexuML checkout.
python library/src/nexuml_library/evaluation/maintenance/rul_independent_confirmation.py \
  --root "$ROOT"

python -m json.tool "$ROOT/independent-confirmation-summary.json"
```

The report loads the native `val/loss` best-checkpoint audit for each run,
then selects the **full-split evaluation-mode validation MSE at that exact
checkpoint epoch**, not the minimum observed MSE over unrelated epochs. It
reports each configuration's mean/std across three readout seeds, paired
differences to baseline, training/validation gap and worst validation-engine
group. It requires the checkpoint files and all nine curves; missing data
or mismatched checkpoint provenance are errors.

The aggregator **does not inspect test metrics or predictions**. The
upstream PatchFormer and the data split are shared across all seeds, so the
three replicates measure readout/training seed variation, not independent
upstream-encoder or test-population variation.

Do not claim a statistically definitive architecture winner from only three
seeds. If the simplified fine-tuning model performs similarly to the more
complex `all_no_trend` model, prefer further validation and efficiency
measurement before adding optional factors to a production model.
