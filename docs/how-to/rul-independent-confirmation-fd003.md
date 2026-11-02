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

The existing `harbor.ika.rub.de/library/nexuml:latest` image, Kueue `jobs`
queue, `lbecker` namespace, source digest staging and ephemeral Python
environment are retained. Each Job requests one NVIDIA GPU. After repeated
incomplete 3 GB Harbor image pulls on other GPU workers, **new** confirmation
Jobs require hostname affinity to the three workers that successfully completed
the previous runs: `cip-gpu-4`, `cip-gpu-9` and `cip-gpu-14`.
CUDA runtime major > 11 and AMD64 constraints remain. This allowlist
also excludes `student-worker-1`. Kubernetes may still queue Jobs if
these three workers have no free GPU; cache availability is not guaranteed.

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

## Recover the five image-pull-blocked Jobs without changing source

These five independent confirmation Jobs did not run due to `ErrImagePull`
or `ImagePullBackOff` (incomplete Harbor layer downloads): baseline seed 44,
finetune seeds 43 and 44, and all-no-trend seeds 42 and 43.
The four complete Jobs must not be restarted. Kubernetes Job pod affinity
cannot be changed in place, so the failed Jobs need to be recreated.

**Do not rerun the standard submit launcher for recovery.** It stages a
snapshot from the current local worktree, which may now differ from the
source used by completed seeds. Instead, use this guarded recovery helper:

```bash
cd /workspaces/NexuFed/NexuML-predictive
git pull --ff-only
pytest tests/library/test_rul_independent_confirmation.py -q

# Preflight: validate original rendered YAML, source digest of all four
# completed runs, live image-pull-failed Pod statuses and new manifests.
python .k8s/scenarios/restart_rul_enhanced_cfc_confirm_fd003.py

# Recreate ONLY the five still image-pull-blocked Jobs.
python .k8s/scenarios/restart_rul_enhanced_cfc_confirm_fd003.py --apply

kubectl -n lbecker get pods \
  -l nexuml-study=rul-fd003-cfc-confirm -o wide
```

The helper reads the five original files from
`.k8s/scenarios/rendered/<job>.yaml` and outputs the pinned versions to
`.k8s/scenarios/rendered/cached-pool-retries/<job>.yaml`. It changes only
the hostname affinity, leaving the original immutable source digest,
source PVC subPath, container image, run name and storage paths untouched.
Before deletion it checks each live Job's source annotation and Pod
image-pull failure state; it refuses to delete a completed or actively
training Job. If a Job has already recovered or original rendered manifests
are missing, stop and inspect instead of manually replacing artifacts.

`imagePullPolicy: IfNotPresent` permits reuse of already cached images.
For a strict image identity check, compare the actual completed Pods'
`imageID` values with the cached image digest on each worker. Because
the `:latest` tag is mutable, an image-cache match by tag alone is not
a cryptographic identity guarantee.

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
