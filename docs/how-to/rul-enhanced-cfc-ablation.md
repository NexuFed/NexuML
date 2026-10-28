# NEX-233: Six-factor CfC remaining useful life ablation

This is an **exploratory** enhancement study for the PatchFormer + CfC RUL
estimator on C-MAPSS FD003. Each optional model component is a NexuML-native
adaptation of mechanisms from the published literature; the study is **not**
an exact reproduction of DMHA-ATCN, MSPA-GAT, DSNet, or TATFA.

**No Flow Matching displacement or guidance is used for supervised RUL.**

## Architecture and fixed protocol

All model variants start with the same FD003 seed-42 pretrained PatchFormer,
train-only standardized 30-cycle × 15-sensor windows, and train-endpoint DPAN
normalization. The CfC readout has width 64, dropout 0.1, a single temporal
layer, token attention pooling, and a 64 → 32 → 1 capped-RUL regressor.

The six **modeling factors** are:

| Factor | Enabled behavior | Disabled behavior |
|---|---|---|
| `finetune` | Update the pretrained PatchFormer DPAN with supervised RUL gradients | Keep DPAN frozen |
| `graph` | Add learned sensor-to-sensor attention on raw sensor channels | Directly encode normalized sensor values |
| `trend` | Add projected adjacent-cycle latent differences | No explicit trend |
| `multiscale` | Add depthwise 3-, 7-, 15-cycle temporal filters | No explicit multiscale block |
| `segmentation` | Add differentiable soft temporal segmentation | Individual time-aligned tokens |
| `trajectory` | Penalize deviations from real same-engine one-cycle RUL progression during fitting | Ordinary window MSE |

The model uses Adam with learning rate 0.001 and a maximum of 150 epochs,
with aggregate validation-loss early stopping (patience 20). Model weights
are initialized reproducibly per head seed with independent optional-module
subseeds. All models share the upstream PatchFormer seed 42.

The training data's paired predecessor windows never cross engine or split
boundaries. The additional trajectory penalty is applied during fitting
only; validation point MSE is directly comparable across branches.

### The six-factor on/off design

For each readout seed (42, 43, 44), the native comparison has **14 branches**:
`baseline`, six individual activations, `all`, and six leave-one-out
configurations (`all_no_<factor>`). This is 42 model branches across
three composite runs, **not** a complete `2^6 = 64` factorial.

The two effect estimates have different meanings:

- `single_minus_baseline`: enabling a technique in an otherwise baseline model.
- `all_minus_leave_one_out`: including that technique in the combined model.

Negative differences mean better validation MSE.

## Uncertainty is separate from the point-estimation ablation

The previous 16-branch experiment incorrectly treated `uncertainty` as a
modeling factor, although it only activates **post-fit** Monte Carlo dropout
and exploratory conformal calibration. Separate but nominally identical
branches consumed different dropout random streams and produced different
point-estimation losses. Such differences are not an architectural effect.

The corrected **six-factor** composite contains no `uncertainty` or
`all_no_uncertainty` branches. MC dropout and conformal diagnostics are
instead attached to the **same fitted `all` standalone model**, not trained
on a duplicate model:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-all-s42 \
  --output .experiments/rul/enhanced-cfc/fd003/all-s42.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/all-s42.yaml
```

The output is
`.experiments/rul/enhanced-cfc/fd003/rul-FD003-enhanced-cfc-all-s42.uncertainty.json`.
The stochastic prediction spread and normalized residual quantile are
descriptive diagnostics only. The calibration windows come from validation
engines that also participated in checkpoint selection, **so the intervals
carry no finite-sample marginal-coverage guarantee**. No official test labels
are read for calibration.

## Native NexuML comparison: resolve → train only

The reference FD003 PatchFormer and train-endpoint normalization artifact
must exist. For a fresh workspace, prepare them using the existing native
scenarios:

```bash
nexuml resolve maintenance-rul-controlled-fd003-patchformer-s42 \
  --output .experiments/rul/controlled-patchformer-fd003.yaml
nexuml train --config .experiments/rul/controlled-patchformer-fd003.yaml

nexuml resolve maintenance-rul-patchformer-sequence-cfc-fd003-s42 \
  --output .experiments/rul/sequence-cfc-fd003.yaml
nexuml train --config .experiments/rul/sequence-cfc-fd003.yaml
```

The second upstream scenario supplies the original **train-only endpoint
normalizer**, not a Flow feature used by the RUL heads. Existing compatible
artifacts can be reused.

Run the corrected comparison for seed 42:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-six \
  --output .experiments/rul/enhanced-cfc/fd003/compare-six-s42.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-six-s42.yaml
```

Then repeat for seeds 43 and 44:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-six-s43 \
  --output .experiments/rul/enhanced-cfc/fd003/compare-six-s43.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-six-s43.yaml

nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-six-s44 \
  --output .experiments/rul/enhanced-cfc/fd003/compare-six-s44.yaml
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-six-s44.yaml
```

**Seed 42 must be rerun.** The original seed-42 comparison trained 16
branches and used `1/16` loss weights. The six-factor comparison has
14 branches and `1/14` weights; pooling old seed-42 data with corrected
seed-43/44 results would confound the comparison.

The corrected study writes separate validation-only artifacts:

```text
.experiments/rul/enhanced-cfc/fd003/
  rul-FD003-enhanced-cfc-compare-six-s42.validation.json
  rul-FD003-enhanced-cfc-compare-six-s43.validation.json
  rul-FD003-enhanced-cfc-compare-six-s44.validation.json
  compare-six-seeds-42-43-44.summary.json
```

The cross-seed summary is produced automatically as soon as all three
matching **v2** results are present. It rejects old 16-factor reports,
reports per-configuration mean/SD across seeds and paired single-on and
leave-one-out effects. No custom runner, environment variable or
post-processing command is required.

## Native composite limitations

The comparison is analogous to the repository's `maintenance-rul-all`
scenario. Within each seed, 14 disjoint model heads share one optimizer,
minibatch stream, aggregate early stopping and an aggregate-best checkpoint.
Composite batch size is 16 to control memory use. Each branch's best
**observed** fitting-epoch validation MSE is reported; its own optimum
may not be the epoch restored by the common checkpoint.

Thus the three composite runs provide a **screening** ablation, not
14 independently checkpoint-selected experiments per seed. The registered
standalone scenarios are still available through `nexuml resolve` and
`nexuml train` for independent confirmation, at batch size 64.

Stochastic dropout remains a potential source of between-branch variance.
Use agreement over seeds and the independently fitted variants rather than
interpreting small differences as definitive causal effects. The official
FD003 test split has been inspected in prior work, so this ablation is
exploratory and must not be used to select the final model on test errors.

## Temporary Kubernetes campaign: immutable JuiceFS snapshot and pod-local venv

This experiment-only deployment **remains on NEX-233 and must not be merged
into `main`**. The regular NexuML Kubernetes templates, core training
infrastructure, and scenario APIs are unchanged.

It follows the established
[`privacy_measure_theory/generative-trajectory-toys`](https://github.com/LucaBeckerICA/privacy_measure_theory/tree/generative-trajectory-toys/.kube_ot)
pattern: compute a SHA-256 digest over actual source files, stage one
immutable and independently verified copy under JuiceFS, submit one
Kubernetes Job per seed, and construct the Python virtual environment on
**ephemeral pod storage**. Neither the staged source nor the pod requires
a Git checkout or `.git` metadata.

### What is staged

The launcher at
`.k8s/scenarios/submit_rul_enhanced_cfc_fd003.py`
stages only runtime source and dependency metadata:

- `src/`, `library/src/`, `scripts/`: Python and Bash files, plus
  required JSON/TOML/YAML/TXT package metadata; no model weights or data
- `pyproject.toml`, `uv.lock`, `README.md`, `LICENSE`,
  `library/pyproject.toml`, `library/README.md`, `library/LICENSE`

The launcher rejects source symlinks and hashes relative file names and
file contents. The same digest is checked after `rsync` and before the
snapshot is published. Subsequent submissions reuse the immutable staged
snapshot, and fail if the contents no longer match.

By default, a digest-addressed snapshot is stored at
`/mnt/juicefs/job-staging/nexuml-nex233/<source-sha256>`.
The rendered Kubernetes `batch/v1 Job` manifests live only under
`.k8s/scenarios/rendered/` (ignored by Git); do **not** apply the deleted
legacy static `job_rul_enhanced_cfc_fd003.yaml`.

### Pod runtime

Each Job uses the existing
`harbor.ika.rub.de/library/nexuml:latest` image—**no dedicated image**.
Its installed `/env` is deliberately not used for the NEX-233
workspace packages.

The source snapshot is mounted read-only under
`/mnt/nexuml-source`. We do **not** mount it at `/workspace` or nest
writable volumes beneath it: OCI cannot create absent mountpoint
directories in a read-only parent filesystem, which previously caused
`error mounting ... /workspace/.experiments ... read-only file system`.

The main container follows these stages:

1. Check the source SHA-256 sentinel, then copy the small staged runtime
   source into `/tmp/nexuml-run` (pod-local writable filesystem).
2. Symlink `/tmp/nexuml-run/.experiments` to the separate persistent
   JuiceFS mount `/mnt/nexuml-experiments`, and `data` to
   `/mnt/nexuml-data`. These paths are never copied into the snapshot.
3. Create a fresh virtual environment under
   `/tmp/nexuml-job-venv/.venv` (`emptyDir`), then run
   `uv sync --locked --no-dev --all-packages --all-extras`.
4. Verify both NexuML packages import and CUDA is available. Run native
   `nexuml resolve` and `nexuml train --config` for the selected seed.

The temporary source and virtual environment disappear with the Pod.
The immutable JuiceFS source snapshot remains reusable and unchanged.
The first start requires the target worker to download any missing locked
dependencies, potentially including large CUDA wheels.

Persistent volumes are **independent mountpoints**, not children of the
read-only source mount:

| JuiceFS PVC subpath | Container mount | Purpose |
| --- | --- | --- |
| content-hashed source snapshot | `/mnt/nexuml-source` (read-only) | Executable source files |
| `code/nexuml/.experiments` | `/mnt/nexuml-experiments` | Upstream models, configs, checkpoints and reports |
| `code/nexuml/data` | `/mnt/nexuml-data` | Cached NASA C-MAPSS data |
| `juicefs-prisma:data` | `/mnt/data` | Existing cluster dataset mount |
| `juicefs-prisma:logs` | `/mnt/logs` | Existing cluster logging mount |

**Prepare the upstream FD003 PatchFormer checkpoint and train-endpoint
normalizer in the shared `.experiments` tree before launching all three
seeds.** These Jobs do not recompute their upstream models.

The Job template retains the established Kueue `jobs` queue,
`lbecker` namespace, `normal` priority, NVIDIA runtime, `regcred`,
one GPU per Job, and AMD64/CUDA runtime constraints. GPU model/memory
filters and explicit CPU/RAM requests/limits remain omitted. Removing
CPU/RAM constraints does **not** reserve a node exclusively.

### Launch from the NEX-233 checkout

The machine running the launcher must have `rsync`, `kubectl`, access
to `/mnt/juicefs`, and an active kubeconfig for the `lbecker` namespace.

```bash
cd /workspaces/NexuFed/NexuML-predictive
git pull --ff-only

# First inspect the three rendered manifests (no upload or submission).
python .k8s/scenarios/submit_rul_enhanced_cfc_fd003.py --dry-run

# Clear ONLY the previous obsolete FD003 Jobs; persistent output is untouched.
kubectl -n lbecker delete jobs \
  -l nexuml-study=rul-fd003-cfc-six \
  --cascade=foreground --wait=true

# Stage one content-hashed source snapshot and submit all three Jobs.
python .k8s/scenarios/submit_rul_enhanced_cfc_fd003.py

kubectl -n lbecker get pods \
  -l nexuml-study=rul-fd003-cfc-six -o wide
kubectl -n lbecker logs -f job/nexuml-rul-fd003-cfc-six-s42
```

To submit a single pilot first:

```bash
python .k8s/scenarios/submit_rul_enhanced_cfc_fd003.py --seeds 42
```

The launcher never deletes a Job on its own. It skips already existing
Jobs and never removes staged sources or persistent outputs; if replacing
a failed Job, inspect its logs before deleting it explicitly. The source
snapshot is shared by all three seeds, with independent output and
checkpoint names. The existing validation-only cross-seed aggregation
runs when all three compatible results are present.

**Image caveat:** The inherited image's `latest` tag is mutable and
`IfNotPresent` may reuse a cached image. Its CUDA/Python tools still need
to be compatible with this branch's lockfile. A truncated image pull
(`unexpected EOF`) is a separate registry/node issue and cannot be
fixed through JuiceFS snapshots or Python dependency installation.

## Check locally

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_enhanced_cfc.py tests/library/test_rul_k8s_staging.py -v
```
