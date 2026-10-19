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

Select configurations using validation results only. After a
configuration and all hyperparameters are fixed, run the final best
checkpoint on the **test** machine by adding --include-test to the
evaluator command (and choose a distinct output JSON file).

## 4. Quality checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_ablation_distributional_validation.py -v
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
