# NEX-233: paired FD001/FD003 generalization study

## Scientific status

These experiments test the hypothesis that the fixed 2:1 asymmetric
squared-error objective improves Goel-style LSTM/CNN prognostics.

This hypothesis was motivated by inspecting **FD001 test residuals**.
Even though the FD001 holdout never enters training, further FD001
comparisons are exploratory. The FD003 holdout provides a useful
different-fault-condition generalization check; do not tune this
objective based on FD003 results and then report the same data as an
independent confirmatory test.

Goel et al. (2026), arXiv:2604.27234v1, Table III report:

| Model | FD001 RMSE | FD001 NASA | FD003 RMSE | FD003 NASA |
|---|---:|---:|---:|---:|
| LSTM | 14.931 | 410.97 | 14.204 | 409.1 |
| CNN | 16.971 | 365.0 | 15.678 | 647.6 |

**Publication comparability limitation:** The paper specifies a
piecewise linear 130-cycle cap and gives the NASA score formula, but
does not unambiguously disclose whether the *official test* RULs
were capped in its numerical results. For every NexuML experiment
report both capped and original official test-label metrics.
Do not present a capped NexuML value as a proven direct comparison
against the publication without clarifying that convention.

FD003 is **not** the same input as FD001: retain sensor s6 on FD003
(15 sensors), but drop it on FD001 (14 sensors), following the paper.
Both use training-only z-score normalization, 30-cycle windows,
zero-padding of short test histories and 130-cycle target capping.

## Paired experiment design

For each of `FD001` and `FD003`, compare the original Goel
MSE control against the safety-aware asymmetric squared loss for
LSTM and CNN, at seeds `42`, `43`, `44`:

- Each seed controls the engine-disjoint training/validation split,
  initialization and training seed.
- A control/safety pair shares exactly the same preprocessed data
  artifact and 80/20 engine split for that seed.
- FD001 and FD003 use separate preparation artifacts and sensor sets.
- Every run has its own checkpoint and prediction audit artifacts.
- Early stopping and model selection use validation loss, never
  the 100 official test-engine labels or NASA score.
- The asymmetric overestimation weight remains fixed at 2; it is
  not tuned separately for FD003 or for any observed test result.
- No changes are made to the original public
  `maintenance-rul-goel2026-lstm-protocol`,
  `maintenance-rul-goel2026-cnn-protocol`,
  `maintenance-rul-safety-lstm` and
  `maintenance-rul-safety-cnn` default experiments.

These are **paired random-seed repeats**, not independent test
replicates: every seed reuses the same official 100-engine test set.
With only three seeds the summary is descriptive; do not infer
statistical significance from a seed standard deviation.

## Native NexuML execution

Run from the repository root after pulling the NEX-233 branch.
The regular C-MAPSS source auto-download also supports FD003.

The following Bash loop calls only the public `nexuml resolve` and
`nexuml train` commands. It writes one unified text log, fresh
study-specific checkpoints and one resolved YAML per experiment.

```bash
set -euo pipefail
mkdir -p .experiments/rul/generalization
export NEXUML_RUL_CHECKPOINT_ROOT="$(pwd)/.experiments/rul/checkpoints/repro-study-1"
log=".experiments/rul/generalization/study.log"
: > "$log"

for subset in fd001 fd003; do
  for seed in 42 43 44; do
    for arch in lstm cnn; do
      for method in goel safety; do
        if [[ "$subset" == "fd001" && "$seed" == "42" ]]; then
          if [[ "$method" == "goel" ]]; then
            scenario="maintenance-rul-goel2026-$arch-protocol"
          else
            scenario="maintenance-rul-safety-$arch"
          fi
        else
          scenario="maintenance-rul-repro-$subset-$arch-$method-s$seed"
        fi
        config=".experiments/rul/generalization/$scenario.yaml"
        {
          printf '\n===== %s =====\n' "$scenario"
          nexuml resolve "$scenario" --output "$config"
          nexuml train --config "$config"
        } 2>&1 | tee -a "$log"
      done
    done
  done
done
```

**Freshness:** use a distinct, initially empty
`NEXUML_RUL_CHECKPOINT_ROOT` for each complete study.
Reusing older checkpoint directories can make provenance unclear.

The output CSVs and split/normalizer manifests appear under
`.experiments/rul/audits/`; the `*.checkpoint.json` files record
selected validation checkpoints. Safety models also write
`*.sensors.json` for training/validation-only sensor-stage evidence
and per-engine train-stage matching after testing.

## Post-run comparison without fitting

After all 24 experiments have completed:

```bash
python -m nexuml_library.evaluation.maintenance.rul_repro_summary \
  --audit-root .experiments/rul/audits \
  --seeds 42 43 44 \
  --output .experiments/rul/generalization/paired-summary.json
```

The summary utility:

1. Requires exactly 100 unique test engines per run.
2. Verifies paired capped and raw test targets, and consistency of
   targets across seeds. Missing results are errors, not ignored.
3. Recomputes RMSE, MAE, mean positive RUL error and NASA score
   from original cycle predictions for capped and raw targets.
4. Reports the **safety-minus-control** metric difference for
   each matched seed, then its descriptive mean and sample standard
   deviation, plus the number of improved seeds.
5. Includes Goel's published metrics only as references with the
   unresolved test-target-convention warning.

Smaller values are better. To call this a robust improvement,
look for a consistent reduction in harmful late RUL predictions
**across paired seeds and on FD003**, with no unacceptable RMSE
tradeoff, before making claims beyond this dataset family.

## Testing

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_generalization.py \
       tests/library/test_rul_safety.py \
       tests/library/test_rul_benchmarks.py \
       tests/library/test_rul_native_cli.py \
       tests/_registry -v
```

The work adds only library components, native registered scenarios,
unit/integration tests, and documentation. No custom training loop,
data-leakage shortcut or core NexuML framework change is required.
