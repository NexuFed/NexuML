# NEX-233: pretrained FISHER, normalization and OT coupling ablation

The existing MIMII log-mel healthy-to-faulty flow experiment remains
unchanged. The ablation tests three factors while reusing the existing
NexuML flow matching path, vector field, loss and Euler integrator:

| Factor | Conditions |
| --- | --- |
| Acoustic representation | Original log-mel spectral features; pretrained frozen FISHER |
| Source–target coupling | Reproducible independent pairs; training-only per-machine finite OT |
| Feature scaling | Unscaled features; per-coordinate pooled train-set standardization |

These produce **eight controlled feature-space experiments** at the
default [128, 128] flow MLP capacity. Two additional FISHER experiments
with [512, 512] MLPs provide a capacity check; do not conflate frozen
encoder size with trainable vector-field capacity.

This is supervised distributional transport, **not** an official MIMII
unsupervised anomaly-detection benchmark and **not** physical RUL inference.

## 1. Prepare the feature datasets

Install the existing optional pretrained requirements and SciPy:

```bash
uv sync --locked --all-packages --all-extras
uv pip install einops scipy
```

The frozen FISHER checkpoint can run on CUDA; the model will be downloaded
on first use. Only execute trusted Hugging Face remote code. For fully
reproducible runs, pass an audited immutable revision with --revision.

```bash
python -m nexuml_library.evaluation.maintenance.ablation_prepare \
  --frontend both \
  --device cuda \
  --machine-type fan \
  --noise-db 6 \
  --seed 42 \
  --output-dir .experiments/nex233-ablation/prepared/seed42
```

The input default is the same MIMII archive used by the existing
mimii-acoustic-flow scenario; pass --root PATH when appropriate.
Use --no-download if all WAV files already exist.

Output: one manifest.json and eight immutable artifacts:

```text
logmel-independent-raw-seed42.pt
logmel-independent-standardized-seed42.pt
logmel-ot-raw-seed42.pt
logmel-ot-standardized-seed42.pt
fisher-independent-raw-seed42.pt
fisher-independent-standardized-seed42.pt
fisher-ot-raw-seed42.pt
fisher-ot-standardized-seed42.pt
```

**Each frontend is instantiated only once during preparation.** Both
conditions use identical feature weights and preprocessing. All split
assignments and original source filenames are preserved.

For each raw feature matrix, standardization fits mean and standard
deviation on **training healthy and training faulty recordings only**:

\[
\mu=\frac{1}{2N_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}(x_{0i}+x_{1i}),\qquad
\sigma^2=\frac{1}{2N_{\mathrm{train}}}
\sum_{i\in\mathrm{train}}
\big((x_{0i}-\mu)^2+(x_{1i}-\mu)^2\big).
\]

OT is computed **after the selected scaling**, with squared Euclidean
cost and exact Hungarian assignment. It matches recordings only **within
each training machine identity**. Val/test pairings are never altered.
Feature-space OT reduces transport cost but is not a physical
wear-progression pairing. Each artifact records train-only normalization
parameters, coupling cost, and the target-velocity cancellation/local
alignment diagnostics from the previous investigation. It also retains
the exact paired recording correspondence. Those diagnostics are directly
available in the evaluation report's prepared_settings field.

Prepared artifacts are locally trusted PyTorch files (weights-only
loading) and are deliberately not overwritten. Separate seeds should
use separate preparation directories.

## 2. Train the eight matched configurations

The dedicated entry point uses NexuSession.fit() then validate(), but
**not** NexuSession.run(): the latter automatically tests every run.
This is necessary to keep the test machine sealed during selection.

The ablation factory requires a prepared file and is therefore deliberately
**not registered as a zero-argument global NexuML scenario**. This keeps
registry conformance independent of local datasets. Use the documented
`ablation_train` command or the module's `scenario()` scenario-file
entry point with `NEX233_PREPARED_FILE` set; do not invoke
`nexuml train mimii-acoustic-ablation` by registry name.

```bash
prepared=.experiments/nex233-ablation/prepared/seed42
checkpoint_root=.experiments/checkpoints/nex233-ablation

for frontend in logmel fisher; do
  for coupling in independent ot; do
    for scaling in raw standardized; do
      file="$prepared/$frontend-$coupling-$scaling-seed42.pt"
      python -m nexuml_library.evaluation.maintenance.ablation_train \
        --prepared-file "$file" \
        --checkpoint-root "$checkpoint_root" \
        --width 128 \
        --lr 0.0003 \
        --max-epochs 300 \
        --seed 42 \
        --accelerator gpu
    done
  done
done
```

Each configuration has a distinct run/checkpoint path and the same
optimization budget. An existing checkpoint directory is rejected to
prevent the Lightning last-v1/last-v2 ambiguity seen previously.

### Distributional validation and model selection

Flow Matching still **trains** on the sample-wise conditional-velocity
regression objective. That loss remains a training diagnostic, but it
does **not** select checkpoints or trigger early stopping.

Instead, after each validation epoch the DistributionalTransportValidation
callback transports **all healthy validation embeddings** through the
learned Euler flow and compares the resulting unordered sample set
against **all faulty validation embeddings** using seeded sliced W1.
Both distributions are drawn exclusively from the disjoint validation
machine. Original pair assignments do not affect this metric, so the
same protocol applies to independent and training-OT models.

- `val/transport_swd`: transported-to-faulty sliced W1; **minimize**
  for ModelCheckpoint and EarlyStopping (patience 25 by default).
- `val/healthy_to_faulty_swd`: no-transport reference (fixed per artifact).
- `val/swd_relative_reduction`: relative transport improvement;
  positive values indicate closer transported distributions.

All epochs and configurations within the same feature space use the
same 128 deterministic projection directions (seed 42 by default),
and the same flow integration as post-training evaluation. Checkpoint
selection uses the absolute SWD **within a run**. For comparisons across
frontends or feature scalings, use baseline-relative SWD improvement,
not absolute SWD. The held-out test partition is never accessed by
the validation callback.

**Existing best checkpoints were selected using val/loss, not SWD.**
Those runs cannot be retroactively treated as SWD-selected; retrain
using a fresh checkpoint directory for this experimental protocol.

### Reliable checkpoint semantics (Lightning 2.6.5)

Lightning 2.6.5's monitored ModelCheckpoint with save_top_k=1 and
save_last=True can leave last.ckpt stuck at the last *improving*
validation epoch. NEX-233 now uses **two independent callbacks**:

- ModelCheckpoint retains the validation-best model only.
- SaveLatestEpoch overwrites last.ckpt after **every completed training
  epoch**, even if no validation improvement occurs.

The latest-state callback is ordered after the validation-best checkpoint
callback so last.ckpt includes the current best-model reference. The
validation-only training command verifies that its saved global_step
matches the final trainer global_step, and aborts on a stale checkpoint.
The CLI prints the verified epoch and step count.

**Previously produced last.ckpt files are not retroactively fixed.**
For the seed-42 FISHER OT-standardized run that completed epoch 25 but
saved epoch 0, the actual final-state weights must be regenerated by
training in a new checkpoint directory. The epoch-0 validation-best
checkpoint remains available for comparison.

The [512, 512] vector field is a second-stage capacity experiment.
For an initial controlled comparison, run both coupling choices under
standardized FISHER features:

```bash
for coupling in independent ot; do
  python -m nexuml_library.evaluation.maintenance.ablation_train \
    --prepared-file "$prepared/fisher-$coupling-standardized-seed42.pt" \
    --checkpoint-root "$checkpoint_root" \
    --width 512 \
    --lr 0.0003 \
    --max-epochs 300 \
    --seed 42 \
    --accelerator gpu
done
```

Repeat on multiple seeds and retain all results, not only the strongest
run. For the first engineering check, seed 42 is sufficient; scientific
comparisons should use repeated seeds.

## 3. Validate and select; do not test all candidates

For example, evaluate the validation-best and last checkpoints
of the FISHER OT + standardized baseline:

```bash
run=fisher-ot-standardized-seed42-mlp-128x128

python -m nexuml_library.evaluation.maintenance.ablation_evaluate \
  --checkpoint-dir "$checkpoint_root/$run" \
  --device cuda \
  --seed 42 \
  --output ".experiments/nex233-ablation/reports/$run.json"
```

The default evaluator accesses only the training mean for the
constant-velocity baseline and the held-out **validation** machine.
It reports (best and last checkpoint):

- SWD to the faulty endpoint distribution at five Euler flow times.
- Healthy→faulty and transported→faulty SWD, including relative reduction.
- Fixed-time FM MSE, zero-velocity MSE, and train-mean-velocity MSE
  (diagnostics only, **not** model-selection criteria).
- Relative improvement over zero and a constant mean-shift SWD baseline.

**Absolute feature SWD and FM MSE are not comparable across FISHER,
log-mel, raw and standardized feature spaces.** Compare dimensionless
relative reductions, baseline-normalized velocity error, stability and
generalization instead. Use the same projection seed and count.

### First controlled follow-up: independent coupling versus OT

The independent-coupling prepared dataset already exists; do not
re-extract frozen FISHER features. Train it using the same MLP 128x128,
optimizer, training budget, and fixed validation-SWD selection as the
existing OT experiment. Use a distinct checkpoint root:

```bash
prepared=.experiments/nex233-ablation/prepared/seed42
python -m nexuml_library.evaluation.maintenance.ablation_train \
  --prepared-file "$prepared/fisher-independent-standardized-seed42.pt" \
  --checkpoint-root .experiments/checkpoints/nex233-independent-swd \
  --width 128 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --seed 42 --accelerator gpu
```

Evaluate both trained variants again with the extended evaluator.
The OT command below reuses the **existing SWD-selected run** without
retraining it:

```bash
report_dir=.experiments/nex233-ablation/reports
mkdir -p "$report_dir"

python -m nexuml_library.evaluation.maintenance.ablation_evaluate \
  --checkpoint-dir .experiments/checkpoints/nex233-independent-swd/fisher-independent-standardized-seed42-mlp-128x128 \
  --device cuda \
  --output "$report_dir/fisher-independent-standardized-swd-trainval.json"

python -m nexuml_library.evaluation.maintenance.ablation_evaluate \
  --checkpoint-dir .experiments/checkpoints/nex233-swd/fisher-ot-standardized-seed42-mlp-128x128 \
  --device cuda \
  --output "$report_dir/fisher-ot-standardized-swd-trainval.json"

python -m nexuml_library.evaluation.maintenance.ablation_compare \
  --independent-report "$report_dir/fisher-independent-standardized-swd-trainval.json" \
  --ot-report "$report_dir/fisher-ot-standardized-swd-trainval.json" \
  --output "$report_dir/fisher-coupling-comparison.json"
```

Each report now includes `training_best` and `training_last`: pairing-invariant
SWD over **all training recordings**, along with `by_machine` SWD for
`id_00` and `id_02`. These diagnostics use the same Euler integrator,
seeded projection count and initial/terminal SWD definition as validation.
Training metrics never affect checkpoint selection. The prepared
artifact includes all splits, but test metrics are computed only when
`--include-test` is explicitly supplied. The comparison command rejects
reports that contain test metrics or are marked unsealed.

The comparison validates matching preparation provenance, MLP widths,
and source/target distribution baselines. Compare relative SWD improvement
on training and validation at both best/last weights. Distinguish
within-machine transport (training SWD improves) from machine-disjoint
generalization (validation SWD improves). A negative validation improvement
does **not** meet the no-transport baseline.

Select configurations using validation results only. After a
configuration and all hyperparameters are fixed, run the final best
checkpoint on the **test** machine by adding --include-test to the
evaluator command (and choose a distinct output JSON file).

## 4. Diagnose machine compatibility before conditioning

Do **not** infer a shared fault mechanism from the pooled training SWD
alone. Machine-wise shifts may be non-IID. Before changing the flow
architecture, evaluate the frozen healthy/faulty marginals for
`id_00`, `id_02` (train) and `id_04` (validation).

Use the already prepared **FISHER independent standardized** artifact.
Within-machine OT only permutes faulty observations; all diagnostics
below are defined on unordered marginal distributions and should be
identical for the independent and OT artifacts.

```bash
prepared=.experiments/nex233-ablation/prepared/seed42
diagnostics=.experiments/nex233-ablation/diagnostics
python -m nexuml_library.evaluation.maintenance.machine_compatibility \
  --prepared-file "$prepared/fisher-independent-standardized-seed42.pt" \
  --output "$diagnostics/fisher-standardized-machine-compatibility.json" \
  --plots-dir "$diagnostics/fisher-standardized-plots" \
  --projections 128 --bootstrap 64 --seed 42
```

The report contains:

1. `healthy_swd` and `faulty_swd`: symmetric inter-machine sliced-W1
   distance matrices using the same fixed projection directions and
   descriptive bootstrap intervals. Unequal sample counts are matched
   through seeded subsampling; no endpoint correspondence is assumed.
2. `healthy_split_half_swd`: reference distances between randomly
   partitioned healthy recordings **from the same machine**, providing
   a scale for interpreting inter-machine distances.
3. `mean_fault_direction_cosine`: machine-specific mean fault shifts
   `E[F|m]-E[H|m]`, pairwise cosine similarities and bootstrap intervals.
4. `healthy_cluster_hypotheses`: descriptive average-linkage merging
   based **only on healthy SWD**, an exploratory two-group assignment,
   and the closest training-machine reference for the validation machine.
5. `mean_shift_transfer_to_validation`: evaluates the training-only
   mean displacements from each training machine and the pooled training
   distribution on healthy `id_04` against faulty `id_04`. A separate
   **validation-derived oracle mean shift** is included as a diagnostic
   only; it must not enter model training, tuning or deployment.

Four optional images are generated: healthy/faulty SWD heatmaps,
fault-direction cosine heatmap, and a 2D healthy embedding projection.
The PCA is fitted **only on healthy training embeddings** and then
applied to validation healthy recordings; it does not influence grouping.

**Scientific limitations:**

- There are only **two distinct training machines**. A pseudo-type
  partition derived using validation `id_04` is transductive/exploratory,
  not a deployable learned classifier. A one-machine pseudo-type cannot
  support out-of-machine conditional generalization.
- Healthy proximity does not prove matching fault mechanisms.
  Compare healthy SWD with the mean fault directions and faulty SWD.
- Per-machine mean shifts summarize only the first moment; similar
  directions do not guarantee a common full distributional transport.
- All descriptive uncertainty reflects within-recording resampling.
  It does **not** measure variation across unseen machine identities.
- These diagnostics intentionally use both healthy and faulty
  **validation** observations to study transfer. That is exploratory
  analysis, not training of a conditional model; report any subsequent
  model design based on these diagnostics as validation-informed.
- The `id_06` test machine is not indexed or evaluated. Neither
  pseudo-type assignment nor cluster selection uses test observations.
- Absolute SWD cannot be compared across raw/standardized feature
  spaces, and 2D PCA proximity is not reliable evidence of clustering
  in all 1536 dimensions.

If healthy clusters *and* corresponding fault-shift compatibility
appear consistent, the next research stage is to study
`q(C|healthy reference recordings)` and conditional vector fields.
That stage requires **additional training machine identities** to
evaluate transfer within pseudo-types. Otherwise, prioritize better
domain-invariant representations or a more controlled fault dataset.

## 5. Complementary within-machine FISHER Flow Matching

The preceding experiment trains on `id_00` and `id_02` and selects on
an **unseen machine** `id_04`. To distinguish fundamental within-machine
learnability from cross-machine generalization failure, train a **separate
unconditioned Flow Matching model for each individual machine**.

Use the existing immutable `fisher-independent-raw-seed42.pt` as the
source. The single-machine preparer selects one machine's healthy/faulty
recordings, deterministically creates a **75% train / 25% validation**
recording-disjoint split, and optionally standardizes the embeddings.
Standardization parameters are fitted only on that machine's **new
training partition**, never on the validation recordings.

The original independent pairing is retained. Training still minimizes
Flow Matching velocity MSE; validation, checkpoint selection and early
stopping use **the complete within-machine distributional sliced W1**.
Each machine receives its **own independently initialized** 128x128
MLP flow, optimizer, checkpoints and report. No pseudo-type conditioning
is introduced in this diagnostic.

### Prepare the individual machines

```bash
source=.experiments/nex233-ablation/prepared/seed42/fisher-independent-raw-seed42.pt
single=.experiments/nex233-ablation/single-machine/seed42
python -m nexuml_library.evaluation.maintenance.single_machine_prepare \
  --source-file "$source" \
  --output-dir "$single/prepared" \
  --scaling standardized --val-fraction 0.25 --seed 42
```

By default, this creates three datasets:

| Machine | Original role | New experiment |
| --- | --- | --- |
| `fan:id_00` | train | machine-local 96 train / 32 val |
| `fan:id_02` | train | machine-local 96 train / 32 val |
| `fan:id_04` | val | machine-local 96 train / 32 val |
| `fan:id_06` | sealed test | **excluded by default** |

Numbers assume the default 128 pairs per machine. No recordings
overlap between the new train and validation partitions.

### Train and evaluate each machine independently

```bash
python -m nexuml_library.evaluation.maintenance.single_machine_run \
  --manifest "$single/prepared/manifest.json" \
  --checkpoint-root "$single/checkpoints" \
  --reports-dir "$single/reports" \
  --width 128 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --accelerator gpu
```

The command trains machines sequentially, saving a validation-selected
checkpoint and a true last-epoch checkpoint for each. It evaluates each
machine's **own** training and validation distributions, and writes
`reports/fan-id_00.json`, `reports/fan-id_02.json`,
`reports/fan-id_04.json` and `reports/summary.json`.

The main comparison is **relative validation SWD reduction** on each
machine versus no transport. Inspect best and last checkpoints, and
compare against the previous cross-machine validation failures.
Validation sample size is only 32 in the default split; repeat with
multiple seeds/splits before inferring robustness.

### Fourth machine: intentionally retire the original test split

After the three other within-machine runs complete, `id_06` can be
repurposed as an **exploratory within-machine experiment**. Select only
that machine to avoid retraining the already completed `id_00`,
`id_02` and `id_04` runs. Both preparation and training require
explicit original-test-machine opt-in:

```bash
python -m nexuml_library.evaluation.maintenance.single_machine_prepare \
  --source-file "$source" \
  --output-dir "$single/including-id06/prepared" \
  --scaling standardized --val-fraction 0.25 --seed 42 \
  --machine-group fan:id_06 \
  --include-original-test-machine

python -m nexuml_library.evaluation.maintenance.single_machine_run \
  --manifest "$single/including-id06/prepared/manifest.json" \
  --checkpoint-root "$single/including-id06/checkpoints" \
  --reports-dir "$single/including-id06/reports" \
  --width 128 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --accelerator gpu \
  --allow-original-test-machine
```

**This deliberately breaks the original `id_06` test seal.** Reports
for the repurposed machine are marked unsealed, and `--include-test`
is prohibited for within-machine holdouts (which have only train and
validation partitions). The original multi-machine evaluation must not
subsequently present `id_06` as an untouched confirmatory test set.
The `--machine-group` selector retains the same deterministic
machine-specific split as the four-machine preparation would, while
creating just one new checkpoint run.

Single-machine success would demonstrate within-domain learnability,
**not** transferable fault mechanisms or calibrated degradation time.

## 6. Identify machine identities and exploratory pseudo-types

**Before implementing a machine classifier + conditional flow**, measure
whether FISHER embeddings contain enough information to recognize a machine
from either **healthy** or **faulty** recordings.

This is **not** classification of an established pseudo-machine-type.
MIMII supplies four *known machine identities* (`id_00`, `id_02`,
`id_04`, `id_06`), but no ground-truth labels for broader machine
types that share transferable fault mechanisms. We measure both (a)
known-machine identity recognition, and (b) learnability of provisional
groups defined **only from healthy training distributions**.

Because the previous exploratory work explicitly retired `id_06` as a
sealed machine-disjoint test set, **all four identities participate**.
We independently stratify recording pairs **within each machine** into
60% training, 20% validation and 20% separate assessment observations.
Healthy and faulty endpoints of the same pair remain in the same split,
and recording file identities must be unique to prevent leakage.
The assessment subset is exploratory: do not present it as a fresh,
unseen-machine confirmatory test.

### Reproducible evaluation

Install the already-defined optional evaluation dependencies:

```bash
uv sync --locked --all-packages --all-extras
```

The source artifact must be **original FISHER-independent-raw**, not the
machine-specific standardized artifacts, to avoid reusing previous
training-only normalization fitted on the old machine split.

```bash
python -m nexuml_library.evaluation.maintenance.pseudo_machine_identification \
  --source-file \
    .experiments/nex233-ablation/prepared/seed42/fisher-independent-raw-seed42.pt \
  --output \
    .experiments/nex233-ablation/identification/fisher-machine-identification.json \
  --plots-dir .experiments/nex233-ablation/identification/plots \
  --seeds 42 43 44 45 46 \
  --train-fraction 0.6 --validation-fraction 0.2 \
  --pca-components 32 --projections 128 --bootstrap 32
```

For each seed, a `StandardScaler` and a 32-dimensional PCA representation
are fitted **exclusively on healthy training embeddings** and then
applied unchanged to all healthy/faulty validation and assessment
embeddings. This controls the risk of learning faulty-specific preprocessing
or leaking validation data into the representation.

Three fixed, untuned logistic-regression classifiers are trained to
predict each target:

| Training observations | Assessment on | Interpretation |
| --- | --- | --- |
| Healthy only | Healthy and faulty | Primary inference/deployment diagnostic |
| Faulty only | Healthy and faulty | Is identity recoverable from faulty signals? |
| Healthy + faulty | Healthy and faulty | Does exposure to faults improve robustness? |

The experiment measures two prediction targets:

1. **Machine identity**: four real, closed-set known IDs. Report 4×4
   confusion matrices and per-machine recall. A healthy-centroid
   nearest-prototype classifier gives a simple baseline.
2. **Pseudo-type hypotheses**: fix `K=2` and `K=3` *a priori*.
   Machine-wise healthy **training** distributions are compared with
   seeded sliced-W1 and grouped by deterministic average linkage.
   A classifier then predicts the resulting coarse pseudo-labels
   for held-out healthy and faulty recordings.

For each condition the report includes accuracy, balanced accuracy,
macro-F1, mean maximum predictive probability, multiclass Brier
score and ten-bin expected calibration error. The CLI aggregates
metrics over five independent within-machine recording splits, and
reports per-seed confusion matrices.

The pseudo-cluster study also reports bootstrap mean adjusted Rand index
and the fraction of bootstrap draws yielding the exact same partition.
The bootstrap resamples **healthy training recordings only**.

For each cluster hypothesis, the report also measures pairwise mean
healthy-to-faulty displacement cosine similarities from **assessment**
recordings and summarizes the within-cluster versus across-cluster
similarities. Faulty assessment observations are used *after* the
healthy-only clustering is fixed; they cannot change pseudo-type
memberships. This is a diagnostic of whether acoustically similar
machines actually have compatible fault transformations, not a
criterion for choosing K.

### Interpret with care

A high machine-ID score means FISHER retains machine identity, not
necessarily that a machine-general conditional flow will succeed.
High pseudo-cluster classification can be vacuous if a cluster
contains only one machine: learning the machine ID then mapping to
a cluster is not learning a transferable machine *type*.

Faulty inference is performed only on observed anomalous endpoints,
**not** on intermediate physical degradation stages. No RUL labels or
synthetic flow-time intermediates are available here.

If the healthy-only classifier is accurate on healthy but not faulty
signals, a deployed router trained only on normal reference data may
require explicit fault robustness. If both conditions are accurate,
we can then evaluate whether the inferred groups actually share
compatible healthy-to-faulty transport before conditioning the flow.

Generated figures compare identity accuracy under healthy/faulty
conditions, display the first-seed training-healthy machine SWD,
and show held-out healthy/faulty confusion matrices for identity and
pseudo-type targets.

## 7. Quality checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_ablation_distributional_validation.py -v
pytest tests/library/test_ablation_coupling_comparison.py -v
pytest tests/library/test_machine_compatibility.py -v
pytest tests/library/test_single_machine_fisher.py -v
pytest tests/library/test_pseudo_machine_identification.py -v
pytest tests/library/test_ablation_checkpoint.py -v
pytest tests/library/test_mimii_acoustic_ablation.py -v
pytest tests/library/test_fisher_encoder.py -v
pytest tests/library/test_mimii_flow_transport.py -v
pytest tests/_registry -v
```

Preparation and scenario unit tests do not download MIMII or FISHER,
and the existing public NexuML core is unchanged.

**Limitations:** The current split uses two training machine IDs and
one validation ID. With only 256 training pairs, either representation
can overfit or generalize poorly; train-standardization may also amplify
low-variance features on unseen machines. The experiment measures
transport between healthy/faulty recording distributions, not true
monotonic degradation trajectories.
