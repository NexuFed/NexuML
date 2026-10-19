# NEX-233 Section 3: verify learned acoustic healthy-to-faulty transport

This evaluation follows [real MIMII flow training](mimii-acoustic-flow-training.md).
It deliberately **does not estimate degradation time or RUL**. MIMII has
normal/abnormal endpoints but no run-to-failure labels. Its original task is
unsupervised anomaly detection; the current experiment is **supervised
transport** and must not be reported as an official MIMII anomaly benchmark.

## 1. Validation-best checkpoint versus final checkpoint

The original `nexuml train mimii-acoustic-flow` run finishes by validating
and testing the **in-memory final state**. With early stopping, this need not
be the model that achieved the best validation loss. Lightning's
`ModelCheckpoint(monitor="val/loss", mode="min", save_top_k=1, save_last=True)`
has already stored the two states.

The post-training evaluator resolves the **validation-best checkpoint from
the ModelCheckpoint callback state in `last.ckpt`**. It does not select any
checkpoint using test results. On **validation only**, it measures both the
best and last checkpoints with the same deterministic FM interpolation grid
`t={0.25, 0.5, 0.75}`; this differs from the random `t` used in training.
Only the **best-validation checkpoint** is subsequently evaluated on the
held-out test machine ID. This prevents using the test loss to select the
model.

If the checkpoint folder contains several interrupted runs or a custom
naming scheme, specify both paths with `--best-checkpoint` and
`--last-checkpoint`. Ambiguous model identification raises an error.

## 2. Evaluate real-healthy-initialized transport

From the predictive-maintenance worktree, with the same local MIMII data
used for training:

```bash
python -m nexuml_library.evaluation.maintenance.flow_transport \
    --checkpoint-dir .experiments/checkpoints/mimii-acoustic-flow \
    --output-dir .experiments/nex233-mimii-evaluation
```

Use `--device cuda` to run flow integration on a CUDA-capable machine.
Use `--no-plots` if `matplotlib` is unavailable.

The evaluator reproduces the training configuration, loads its original
validation/test units, and computes:

- **Fixed-time FM MSE** at the same three times for every candidate checkpoint
  on the validation split, then once on held-out test using the selected
  best-validation model.
- **Sliced 1-Wasserstein distance** from real healthy log-mel features to
  real faulty features, and from **transported healthy** features to faulty
  features. Random unit projection directions are fixed by the evaluation
  seed and reused at every transport time. Samples are sorted along each
  projected dimension; no arbitrary file-level pairing is treated as
  physical trajectory ground truth.
- **Transport path diagnostic:** distance to real faults at
  `t=0, 0.25, 0.5, 0.75, 1`, obtained with the **trained NexuML Euler
  integrator** and genuine healthy observations as the initial state.
- Mean RMS feature displacement and the relative SW1 reduction
  `1 - SW1(transported, faulty) / SW1(healthy, faulty)`.

A **positive distance reduction** means the transported distribution is
closer to the observed faulty distribution under this estimator.
A **negative reduction** means it moved farther away. This is a diagnostic
of distributional movement, not proof of valid degradation time or RUL.
The transport need not be monotone in sliced Wasserstein distance.

The default protocol is:

| Split | MIMII machine IDs | Model use |
| --- | --- | --- |
| Training | `id_00`, `id_02` | Trained before this evaluation |
| Validation | `id_04` | Compare best/last (fixed-time MSE and SW1) |
| Held-out test | `id_06` | Evaluate selected validation-best once |

No retraining occurs in the evaluator. The automatic sampler used in the
Flow layer's test-mode `forward` starts from random noise; **it is not
used** here. The evaluator explicitly calls `flow.sample(healthy_features)`
so its trajectory represents the intended healthy-to-faulty transport.

## 3. Convergence curves

TensorBoard already stores loss histories for your trained run. The
evaluator accepts **one explicit TensorBoard version directory** to avoid
mixing metrics from separate experimental runs:

```bash
find .experiments/tensorboard -name 'events.out.tfevents.*'
```

For example:

```bash
python -m nexuml_library.evaluation.maintenance.flow_transport \
    --checkpoint-dir .experiments/checkpoints/mimii-acoustic-flow \
    --tensorboard-run \
      ".experiments/tensorboard/NEX-233 MIMII acoustic health-to-fault transport/mimii-fan-seed-42_v4"
```

The script extracts available `train/loss`, `train/loss_epoch`, and
`val/loss` scalars and plots them against global training step. Omitting
`--tensorboard-run` skips convergence plots rather than accidentally
selecting a different experiment's event directory.

## 4. Artifacts

Under `.experiments/nex233-mimii-evaluation/`:

| File | Contents |
| --- | --- |
| `metrics.json` | Best/last epoch and global-step provenance, validation and held-out test metrics, SW1 trajectories and optional training history |
| `transport_distance.png` | Sliced W1 distance to faulty distribution along the flow |
| `feature_pca.png` | Real healthy, real faulty and transported endpoints in a **descriptive** 2D PCA projection |
| `loss_curves.png` | Train/val TensorBoard losses, only when `--tensorboard-run` is specified |

PCA is used for illustration only; the quantitative distances are computed
in the full feature space. No inference is made about true machine age.

## 5. Tests and quality checks

All synthetic tests run without downloading MIMII or reading your trained
checkpoint:

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_mimii_flow_transport.py -v
pytest tests/library/test_mimii_acoustic_flow.py -v
pytest tests/library/test_flow_matching.py -v
pytest tests/_registry -v
```

The tests cover exact transport of a known translation field, negative
transfer, determinism and invalid inputs for sliced Wasserstein, correct
checkpoint selection and rejection of ambiguous checkpoint directories,
and real acoustic-format validation/test batches from temporary fixture WAVs.

## 6. Interpreting the real-data experiment

A checkpoint with a lower fixed-time FM MSE may still have a worse SW1
distribution match; report **both** rather than treating a low training
loss as proof of physically meaningful transport. The next NEX-233 stage
(single-sample flow-position estimation) should not proceed on the basis
of training loss alone.
