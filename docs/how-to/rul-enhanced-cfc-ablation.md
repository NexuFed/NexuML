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

## Temporary Kubernetes placement (NEX-233 branch only)

The existing site-specific Kubernetes `batch/v1 Job` template has an
experiment-only adaptation at
`.k8s/scenarios/job_rul_enhanced_cfc_fd003.yaml`. It consists of
**three ordinary Kubernetes Jobs**, one per native `compare-six` seed.
This is **not** a NexuML Kubernetes executor, not a core feature and
**not intended to be merged into `main`**. The AudioSet templates are
unchanged.

Each job directly executes the standard resolved NexuML command:

```text
nexuml train --config .experiments/rul/enhanced-cfc/fd003/compare-six-s42.yaml
```

The other jobs use the corresponding `s43` and `s44` YAML files.
There are no additional launch scripts or experiment environment
variables. Kubernetes controls GPU placement and the job lifecycle.

### Preconditions before applying the manifest

1. The cluster's existing `juicefs` and `juicefs-prisma` PVCs and
   `regcred` image pull secret must be available in the target namespace,
   matching the original AudioSet manifest.
2. The existing `juicefs` subpath `code/nexuml`, mounted at
   `/workspace`, must contain the **NEX-233 feature-branch code**.
   The `harbor.ika.rub.de/library/nexuml:latest` image is inherited from
   the existing template; verify its installed editable environment is
   compatible with this working tree. The normal image publishing workflow
   only builds `main` and release revisions, so **do not assume `latest`
   contains NEX-233**. If necessary, build a separate image from the feature
   branch and change only the image tag in the temporary manifest.
3. Required FD003 seed-42 upstream PatchFormer checkpoint, train-endpoint
   normalizer, and prepared RUL data must already be available under the
   paths resolved relative to `/workspace`. Prepare or copy them
   **before launching multiple seeds**, to avoid simultaneous writes to
   the same prepared-data file.
4. Resolve all three scenario YAML files **from `/workspace` on the same
   mounted filesystem**. The scenarios contain checkpoint callbacks with
   absolute paths produced by `Path.resolve()`. A YAML file resolved on a
   laptop under a different filesystem root is not portable unchanged.

Prepare the three configs in that mounted branch workspace:

```bash
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-six \
  --output .experiments/rul/enhanced-cfc/fd003/compare-six-s42.yaml
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-six-s43 \
  --output .experiments/rul/enhanced-cfc/fd003/compare-six-s43.yaml
nexuml resolve maintenance-rul-enhanced-cfc-fd003-compare-six-s44 \
  --output .experiments/rul/enhanced-cfc/fd003/compare-six-s44.yaml
```

Submit the three existing-style Kubernetes Jobs (from a machine with
cluster access and the intended Kubernetes namespace):

```bash
kubectl -n YOUR_NAMESPACE apply \
  -f .k8s/scenarios/job_rul_enhanced_cfc_fd003.yaml
kubectl -n YOUR_NAMESPACE get jobs -l nexuml-study=rul-fd003-cfc-six
kubectl -n YOUR_NAMESPACE logs -f job/nexuml-rul-fd003-cfc-six-s42
```

The jobs can run concurrently if three suitable GPUs are available; they
otherwise remain pending until scheduled. Their checkpoints, TensorBoard
logs, validation JSON files and three-seed summary persist on the shared
workspace. Jobs have no automatic time-to-live or restart retries, so
failed pods can be inspected without inadvertently restarting long fits.

**Hardware compatibility:** The inherited CUDA image publishing workflow
builds `linux/amd64`. The temporary manifest explicitly restricts
scheduling to amd64 GPU nodes. In particular it will **not** schedule
onto an ARM64 Grace Hopper GH200 node without an arm64-compatible image
and a corresponding manifest adjustment.

This workflow is limited to the NEX-233 experiment branch. It does not
alter the public NexuML training/CLI contracts.

## Check locally

```bash
git pull
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_enhanced_cfc.py -v
```
