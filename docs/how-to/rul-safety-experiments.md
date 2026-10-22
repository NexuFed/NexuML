# NEX-233 — Safety-aware RUL experiments

This is a **new method**, not a reproduction of Goel et al. (2026).
The existing `maintenance-rul-goel2026-lstm-protocol` and
`maintenance-rul-goel2026-cnn-protocol` scenarios are unchanged.

## Motivation and data separation

The previous restored-checkpoint FD001 audit found large, positive
RUL errors on held-out engines 27, 67 and 79. Engine 84 showed a
different model-specific error. These observations motivate a safety
objective, **not** fitting to those four engines: none of the 100
official FD001 test engines is ever used to construct losses, stage
statistics, normalize sensor inputs, select checkpoints or adjust
hyperparameters.

The new scenarios share the Goel architecture, optimizer, source
C-MAPSS FD001 split, 30-cycle windows, 14-sensor selection, RMSprop,
and train-only z-score normalization. Each has its own run identity,
checkpoint directory, and test reports.

## Training objective

Let `e = RUL_predicted - RUL_true` in cycles. Instead of MSE, train
with the stable expectile-style objective

```text
L = mean( w(e) * e**2 )
w(e) = 2 if e > 0 else 1
```

An overestimated RUL carries twice the squared-error penalty of an
equal-magnitude underestimate. This does **not** optimize the test
NASA score, and the weight is **not fitted** on held-out FD001 engines.
The standard `val/loss` now uses the same asymmetric objective, so
Lightning's validation checkpoint and early-stopping selection are
cost-sensitive. The Goel best-weights restore callback and publication
architecture are retained; only the safety scenarios' loss is changed.

This method can trade MAE/RMSE for reduced dangerous overestimation.
It is not guaranteed to outperform the published-protocol baselines
on held-out engines. Compare all results against both controls.

## Native experiments

```bash
nexuml resolve maintenance-rul-safety-lstm \
  --output .experiments/rul-safety-lstm.yaml
nexuml train --config .experiments/rul-safety-lstm.yaml

nexuml resolve maintenance-rul-safety-cnn \
  --output .experiments/rul-safety-cnn.yaml
nexuml train --config .experiments/rul-safety-cnn.yaml
```

Training uses the existing official FD001 data download/cache.
The two experiments can share the underlying immutable prepared
Goel data artifact; they never share model checkpoints. Always
re-resolve after pulling changed library components.

## Diagnostics

For each new run, NexuML writes under `.experiments/rul/audits/`:

- `<run>.sensors.json`: **training and validation engines only**.
  For each RUL interval (0–30, 30–60, 60–90, 90–120, and 120+
  cycles) it records last-cycle normalized sensor means and standard
  deviations, temporal slopes within the causal 30-cycle window,
  and sample counts. It ranks features by the standardized
  60–90 versus 0–30 contrast *using training windows only*.
- `<run>.checkpoint.json`: best `val/loss` checkpoint path,
  validation score, epoch and step.
- `<run>.csv` and `<run>.json`: after testing, one row per
  official held-out FD001 engine, with prediction, original and
  capped targets, signed errors and NASA penalties, and the
  original split/normalizer provenance.

The stage audit uses window-level summaries: many windows from one
engine are correlated, so a large sample count is **not** a count of
independent engines. This is a descriptive signal-discrimination
diagnostic rather than a statistical proof of a sensor's causal
contribution.

## Evaluation plan

1. Compare **validation** loss, RMSE, mean RUL overestimation and
   whether mid-life sensor patterns distinguish from late-stage
   patterns; select the safety formulation without consulting held-out
   FD001 labels.
2. Compare the unchanged Goel-control runs with both new safety
   scenarios on the original 100-engine test set. Keep raw and
   capped test metrics distinct.
3. Inspect whether the extreme positive errors on engines 27, 67
   and 79 are reduced **without** introducing more harmful errors
   elsewhere. Inspect engine 84 as a model-disagreement case.
4. Prefer repeated independent training/validation splits and
   seeds, and a separate holdout, before concluding that the method
   generalizes. The current 2:1 penalty is one prespecified method,
   not a tuned optimal value.

### Validation

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_safety.py \
       tests/library/test_rul_native_cli.py \
       tests/library/test_rul_benchmarks.py \
       tests/_registry -v
```

No performance improvement is claimed until new runs complete.
