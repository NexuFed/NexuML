# NEX-233: investigate acoustic Flow Matching velocity disagreement

This diagnostic tests whether **independently coupled healthy and faulty
recordings produce conflicting target velocities**, and whether a trained
vector field has learned those velocities. It is a post-training analysis
of the existing supervised MIMII experiment, not a change to its training
objective, and **does not infer degradation time or RUL**.

The paired path defines

```text
target_velocity = faulty_log_mel - healthy_log_mel
state(t) = healthy_log_mel + t * target_velocity
```

Squared-error Flow Matching optimizes a conditional mean velocity at
each interpolation state and time. Opposing target velocities can reduce
the optimal predicted magnitude, but **small learned displacements alone
do not prove cancellation**: undertraining, feature scaling and model
capacity are other possible explanations.

## Run on an existing training run (no retraining)

Select the correct checkpoint run explicitly. For example, to diagnose
the validation-best checkpoint associated with the previously reported
`last-v4.ckpt`:

```bash
git pull --ff-only

python -m nexuml_library.evaluation.maintenance.velocity_diagnostics \
  --checkpoint-dir .experiments/checkpoints/mimii-acoustic-flow \
  --last-checkpoint .experiments/checkpoints/mimii-acoustic-flow/last-v4.ckpt \
  --device cuda \
  --output .experiments/nex233-velocity-diagnostics-run5.json
```

The evaluator resolves the **validation-best checkpoint referenced inside
the selected last checkpoint**. It reads training and validation machine
IDs only; the held-out test set is not opened. All training-only
comparators are fitted on the same training recording subset.

Requires SciPy for the diagnostic Hungarian assignment. No new MIMII
download should occur if the original dataset directory is still present.

## Reported diagnostics

| JSON field | What it measures |
| --- | --- |
| `training.velocity_statistics` | Target-velocity RMS, mean-velocity RMS, feature scale and unconditional cancellation |
| `training.neighbor_alignment` | Descriptive local alignment of target velocities among nearest interpolated states, at three fixed flow times |
| `training.pairing_alternative` | Actual independent pair MSE versus hypothetical minimum-cost training-only matching, **separately per machine ID** |
| `training.predictor.fixed_times` | Trained predictor MSE and cosine agreement versus zero and constant train-mean velocities |
| `validation.predictor.fixed_times` | The same predictor comparison on unseen validation machine ID |
| `validation.transport_baselines` | Healthy, learned flow and constant train-mean-shift SWD to validation faulty observations |

### A. Unconditional cancellation fraction

The training-only diagnostic uses

\[
v_i=x_{1,i}-x_{0,i},\qquad \bar v=\frac1N\sum_i v_i
\]

and reports

\[
C=1-\frac{\|\bar v\|^2}{\frac1N\sum_i\|v_i\|^2}.
\]

A value near 1 means that endpoint velocity vectors have a small
**unconditional** mean relative to their RMS magnitude. A value near 0
means they are strongly aligned. This does **not directly estimate**
\(\mathbb E[v\mid x_t,t]\), the optimal flow predictor.

### B. Local target-velocity conflict

For each `t = 0.25, 0.5, 0.75`, nearest interpolation states are located
using the pooled **training-only feature standard deviations**. The
mean cosine between each target velocity and its neighbours' mean target
velocity provides a qualitative test of local inconsistency. In high
dimensions, nearest neighbours may still be far apart: interpret this
as an exploratory indicator, not a proof of cancellation.

### C. Hypothetical OT matching

Within each **training machine** separately, the diagnostic compares
the observed squared distances between independent pairs with the
minimum-cost finite assignment from SciPy's Hungarian algorithm.

A substantial assignment-cost reduction indicates that many independent
pairs create unnecessarily long feature-space paths. It does **not**
establish that those matched pairs describe physical degradation. Nor
does target reassignment change the global mean (\bar v) inside a
machine, because the faulty marginal remains unchanged. **No pairing is
changed in the actual training pipeline.**

### D. Learned versus baseline velocities

At fixed times the diagnostic compares the trained velocity prediction
against:

- `zero_velocity_mse`: always predicting no movement.
- `training_mean_velocity_mse`: predicting the constant mean velocity
  computed from **training** endpoint pairs.
- `learned_velocity_mse`: the trained `Flow` layer's actual prediction.

The ratios of predicted/target RMS and their cosine similarity help
distinguish target conflict from an optimizer that simply learned
near-zero velocities. Validation SWD also compares a fixed train-mean
translation to the learned transport; **lower is better**.

## What would support the cancellation hypothesis?

Evidence would include a low global velocity-energy fraction, low or
negative local velocity alignment, and a substantial gap between
independent versus reassigned training pair costs.

If target velocities are coherent but the learned model's output RMS
remains far below target RMS and the model cannot beat the constant
train-mean predictor, investigate feature normalization, learning rate,
hidden dimensions and optimization before changing coupling.

If hypothetical OT pairing substantially reduces target-path variance,
the next controlled experiment could compare **independent versus OT
coupling under identical train/validation splits, model and optimizer**.
Do not use held-out test performance to select the pairing method.

## Verify

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_mimii_velocity_diagnostics.py -v
pytest tests/library/test_mimii_flow_transport.py -v
pytest tests/_registry -v
```

The tests use mathematical examples and small synthesized WAV fixtures
without downloading any MIMII archives. Numerical findings on the
actual machine dataset require running the CLI above.
