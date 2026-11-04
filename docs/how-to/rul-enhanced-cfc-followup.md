# NEX-233: CfC reliability, component ablations and FD001 replication

The first independently optimized FD003 comparison selected
`all_no_trend` by mean validation MSE, but only two of three readout seeds
beat baseline. This follow-up evaluates stability, identifies interactions
and checks replication on a different C-MAPSS subset.

**This is a new experiment namespace.** It never modifies or ranks
the completed `.../fd003/confirmation` results. The upstream PatchFormer
weights and train-only endpoint normalization remain frozen at seed 42.
All model readout/training seeds are paired within an experiment.

## Prespecified matrix (41 native NexuML Jobs)

| Campaign | Subset | Configurations | Head seeds | Jobs |
| --- | --- | --- | --- | ---: |
| `reliability` | FD003 | `baseline`, `all_no_trend` | 42–51 | 20 |
| `ablation` | FD003 | `without_finetune`, `without_graph`, `without_multiscale`, `without_segmentation`, `without_trajectory` | 42–44 | 15 |
| `fd001` | FD001 | `baseline`, `all_no_trend` | 42–44 | 6 |

Each leave-one-out configuration starts from the five enhancements of
`all_no_trend` (fine-tuning, sensor graph, multiscale, segmentation,
trajectory), then disables exactly one. The baseline and complete
`all_no_trend` seeds 42–44 from the **new** reliability campaign serve as
paired controls for all five ablations. No redundant control Jobs are run.

The FD001 replication trains a **separate FD001 PatchFormer/CfC model**
with FD001's own 14 sensor inputs. It is a new dataset-subset experiment,
**not** a test of FD003-trained weights on unseen FD003 engines. Do not
interpret an FD001 validation score as a holdout of FD003's validation set.

## Prerequisites

From the NEX-233 branch worktree, with the normal JuiceFS source mount,
Prisma experiment mount, and a working Kubernetes/Kueue client:

```bash
cd /workspaces/NexuFed/NexuML-predictive
git pull --ff-only

ruff check .
ruff format --check .
ty check

# Prefer the library and NexuML code from THIS worktree even when the
# active venv was initially created in a sibling checkout.
PYTHONPATH="$PWD/src:$PWD/library/src" pytest \
  tests/library/test_rul_enhanced_cfc_followup.py \
  tests/library/test_rul_independent_confirmation.py \
  tests/library/test_rul_k8s_staging.py -v
```

The FD003 campaign expects the **existing** validated PatchFormer seed-42
checkpoint audit and matching train-only endpoint-normalization artifact.

Before launching FD001, verify its own upstream artifacts:

```bash
RUL=/mnt/juicefs_prisma/models/nexuml/.experiments/rul

test -f "$RUL/audits/rul-FD001-controlled-patchformer-seed42.checkpoint.json"
test -f "$RUL/prepared/rul-FD001-patchformer-sequence-cfc-flow-seed42-endpoints.pt"
```

The checkpoint referenced by the FD001 audit must also resolve within
the Pod. If missing, prepare the FD001 upstream stages with the original
native NexuML scenarios first. The follow-up launcher does **not** pretend
to train a new encoder or automatically generate unverified endpoints.

## Submit in stages

```bash
# 1. Inspect the twenty paired 10-seed FD003 Jobs first.
python .k8s/scenarios/submit_rul_enhanced_cfc_followup.py \
  --campaign reliability --dry-run

# 2. Submit the reliability experiment.
python .k8s/scenarios/submit_rul_enhanced_cfc_followup.py \
  --campaign reliability

# 3. Submit factor omissions (the reliability seeds 42-44 are controls).
python .k8s/scenarios/submit_rul_enhanced_cfc_followup.py \
  --campaign ablation

# 4. Once FD001 upstream audit and endpoint normalizer exist:
python .k8s/scenarios/submit_rul_enhanced_cfc_followup.py \
  --campaign fd001
```

Alternatively, `--campaign all --dry-run` previews all 41 Jobs. The
launcher discovers GH200 node names/labels from Kubernetes, excludes
all GH200 hosts and `student-worker-1`, requests **one NVIDIA GPU**
per Job, and requires AMD64. It uses
`ghcr.io/nexufed/nexuml:sha-f8d5490-cuda12.8.1`. CPU and RAM
requests are currently subject to the namespace's defaults; the Pod
manifest does not explicitly request either.

The source is copied to a content-addressed immutable snapshot at
`/mnt/juicefs/job-staging/nexuml-nex233/<source_sha256>`. Training
data, checkpoints, audits, and curves are persisted on
`/mnt/juicefs_prisma/models/nexuml/.experiments`. Each Job uses a
fresh ephemeral `uv` environment from that snapshot. The launcher
skips existing Jobs; it does not overwrite other campaigns.

```bash
kubectl -n lbecker get jobs -l nexuml-study=rul-cfc-followup
kubectl -n lbecker get pods -l nexuml-study=rul-cfc-followup -o wide
```

## Validation-only reporting

```bash
ROOT=/mnt/juicefs_prisma/models/nexuml/.experiments/rul/enhanced-cfc/followup

# Each report requires all of its own Jobs, and for ablation the
# three reliability all_no_trend controls.
python library/src/nexuml_library/evaluation/maintenance/rul_followup_summary.py \
  --root "$ROOT" --campaign reliability

python library/src/nexuml_library/evaluation/maintenance/rul_followup_summary.py \
  --root "$ROOT" --campaign ablation

python library/src/nexuml_library/evaluation/maintenance/rul_followup_summary.py \
  --root "$ROOT" --campaign fd001

# Once all 41 runs are complete:
python library/src/nexuml_library/evaluation/maintenance/rul_followup_summary.py \
  --root "$ROOT" --campaign all

python -m json.tool "$ROOT/all-summary.json"
```

The report verifies **every** success marker, exact container tag, source
digest, validation split sample sizes, selected validation checkpoint,
checkpoint epoch and evaluation-mode validation MSE. The ten-seed paired
difference includes mean, sample SD, wins, and an exploratory 95% Student
t interval **over readout seeds only**. No test predictions, raw test labels
or held-out test metrics are read by the aggregator.

FD003 testing was examined in previous exploratory work, so no FD003
follow-up result should be described as final held-out confirmation.
The FD001 subset provides an independent *benchmark configuration*
but has its own validation-based checkpoint selection. For a new
paper-quality held-out generalization claim, freeze the architecture and
experiment protocol first, then evaluate on a dataset or engine population
not used to choose either model design or hyperparameters.
