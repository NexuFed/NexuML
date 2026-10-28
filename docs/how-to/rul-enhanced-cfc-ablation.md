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

## Temporary Kubernetes execution — feature branch only

The three experimental Kubernetes Jobs in
`.k8s/scenarios/job_rul_enhanced_cfc_fd003.yaml` are adaptations of the
existing site-specific AudioSet `batch/v1 Job` template. They remain **only
on NEX-233** and are **not intended for a merge into `main`**. No new
NexuML execution backend, custom experiment environment variables or
shell-based training runners were introduced.

### Use the existing `latest` image without hiding its installation

The experimental Jobs use the existing
`harbor.ika.rub.de/library/nexuml:latest` image. No special
feature-branch image is built or pushed.

The image's `/env/bin/nexuml` entry point uses a workspace-installed
NexuML package. Mounting the entire `code/nexuml` JuiceFS tree over
`/workspace` hid the installed source and caused
`ModuleNotFoundError: No module named 'nexuml'`.

To avoid that failure, the Jobs now mount **only the NEX-233 source
packages** at the image's existing editable package directories,
while preserving the rest of `/workspace` and `/env`:

| Existing JuiceFS PVC subpath | Container mount | Purpose |
| --- | --- | --- |
| `code/nexuml/src/nexuml` | `/workspace/src/nexuml` | NexuML Python package |
| `code/nexuml/library/src/nexuml_library` | `/workspace/library/src/nexuml_library` | NEX-233 component/scenario definitions |
| `code/nexuml/.experiments` | `/workspace/.experiments` | Configs, checkpoints, and output |
| `code/nexuml/data` | `/workspace/data` | Dataset files |

**Before running:** The two source-package paths in JuiceFS must
contain the actual **NEX-233 feature-branch code**. The `latest`
image supplies Python packages and their dependencies, but it may
contain older `main` code. The mounted source paths override that
code. The container cannot use the developer's local worktree unless
its files have actually been copied/synced into these shared JuiceFS
paths. Do not overwrite a shared `code/nexuml` checkout being used
by another workload without coordinating access.

The `latest` image must also have compatible installed dependencies
and editable workspace installation paths; it cannot be guaranteed
from the repository Dockerfile alone that a previously published
`latest` image matches those expectations. The first init-container
`nexuml resolve` validates Python package discovery before training
starts.

The upstream FD003 seed-42 PatchFormer checkpoint, endpoint
normalization, and prepared RUL data must already exist under the
shared paths. Do not launch three simultaneous preparation attempts.

### Resolve and train using NexuML itself

Each Job has a native NexuML **init container** which runs
`nexuml resolve` against the same `/workspace` used by its training
container. This avoids the nonportable absolute paths that would result
from resolving a YAML on a developer's local `/workspaces/...` tree.

The main container then runs exactly:

```text
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-six-s42.yaml
```

The two other Jobs use seeds 43 and 44. All three share persistent output
storage and can run concurrently if GPU capacity permits. Each requests
one GPU, with no explicit CPU or RAM requests or limits, no GPU memory or
model allowlists, and the original CUDA runtime major-version constraint.
The jobs retain the `amd64` architecture restriction because the
standard image is built for `linux/amd64`.

### Redeploy failed or previously pending Jobs

```bash
kubectl -n lbecker delete jobs \
  -l nexuml-study=rul-fd003-cfc-six \
  --cascade=foreground --wait=true

kubectl -n lbecker apply \
  -f .k8s/scenarios/job_rul_enhanced_cfc_fd003.yaml

kubectl -n lbecker get pods \
  -l nexuml-study=rul-fd003-cfc-six -o wide

kubectl -n lbecker logs job/nexuml-rul-fd003-cfc-six-s42 \
  -c resolve-config

kubectl -n lbecker logs -f \
  job/nexuml-rul-fd003-cfc-six-s42 -c nexuml
```

The validation-only per-seed and combined three-seed summary JSON files
are written to the persistent `.experiments/rul/enhanced-cfc/fd003`
directory. Kubernetes cluster credentials, PVC contents, image-push
permissions, and real hardware availability must be verified on-site.

## Check locally

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_enhanced_cfc.py -v
```
