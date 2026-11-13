# NEX-233: within-machine flow versus non-flow RUL

## Purpose and research question

Investigate **within-machine degradation dynamics first**, before a global
or mixture-of-experts model. The experiment uses the frozen original FD003
PatchFormer (seed 42) and compares three estimators at exactly the same
observation cutoff:

1. **Direct RUL regression, without dynamics:** a fixed affine ridge readout
   fitted on **completed training engines only**.
2. **Direct future-state forecasting, without flow:** a separate neural
   forecaster adapted to each validation engine's **unlabeled prefix**.
3. **Conditional Flow Matching:** a neural conditional transport vector field
   adapted to the *same* engine prefix with identical transition pairs,
   optimizer, network parameter count and training budget.

The dynamic methods are converted into RUL estimates with the **same frozen
train-engine-only RUL readout** and the same conversion rule. Results are in
NASA FD003 capped physical RUL cycles, not normalized labels.

This is a **research/offline evaluation component in NexuLibrary**. It is run
via a Python module, not a `nexuml train` scenario, because each validation
engine gets its own online-adapted, unsupervised neural dynamics model while
all supervised calibration stays train-engine-only. No changes to NexuML's
core training/CLI/registry are needed.

## Mathematical setup

For a particular engine \(i\), a frozen encoder transforms the observed
30-cycle sensor history into \(z_{i,t}\in\mathbb R^{128}\).
The representation is standardized using statistics fitted only on
completed **training** engines.

The direct-future non-flow model learns a displacement:

\[
\widehat z_{t+h}
=z_t+f_\theta(z_t,z_t,0,h).
\]

The flow model learns an **initial-state and physical-horizon conditional**
vector field from true, chronological **same-engine** pairs
\((z_t,z_{t+h})\):

\[
z_s=(1-s)z_t+sz_{t+h}, \quad
u^*=z_{t+h}-z_t,
\]

\[
\min_\theta\mathbb E_{s,t,h}
\|v_\theta(z_s,z_t,s,h)-u^*\|_2^2.
\]

For prediction, integrate the conditional ODE over interpolation coordinate
\(s\in[0,1]\), holding the requested **physical cycle horizon** \(h\) fixed:

\[
\frac{dz_s}{ds}=v_\theta(z_s,z_t,s,h),\qquad z_0=z_t.
\]

**Warning:** \(s\) is the Flow Matching interpolation variable. It is *not*
a machine's chronological age or a learned physical degradation clock.
The horizon \(h\) is measured in actual C-MAPSS cycles. Thus the method is a
conditional continuous latent *forecast* (not yet an autonomous long-term
degradation ODE).

Both models use the exact same MLP parameter count (two 128-wide hidden
layers, 258 input dimensions and 128 output dimensions) and the same target
endpoint displacement. The direct model uses \(s=0\); the flow model samples
\(s\) and integrates with eight Euler steps by default.

### From forecast dynamics to RUL

A train-only ridge readout \(r(z)=w^\top z+b\) supplies an identical
degradation coordinate for both dynamic models. For physical horizons
\(h\in\{1,5,10,20\}\), regress \(r(\widehat z_{t+h})-r(z_t)\) on \(h\)
through the origin to estimate the local slope \(\dot r\).

The dynamic RUL estimate is then

\[
\widehat{\mathrm{RUL}}
=\operatorname{clip}\left(
  \frac{r(z_t)}{-\dot r}, 0,130
\right)
\]

for a sufficiently negative slope. Non-degrading or ambiguous slopes are
**right-censored at 130 cycles** and explicitly flagged in the per-engine
output. This is a *first*, interpretable local-rate conversion, not yet an
optimized/validated flow-to-RUL MLP. The direct baseline simply returns
\(\operatorname{clip}(r(z_t),0,130)\).

This distinction matters: the flow model is trained with *unlabeled future
sensor embeddings*, not with the target machine's failure time.

## Leakage-safe causal evaluation protocol

- Frozen pretrained seed-42 FD003 PatchFormer, and existing
  `cmapss-FD003-seed42-controlled-w30-s42.pt` prepared windows.
- **80 train engines:** fit the frozen, engine-balanced, ridge RUL decoder and
  feature standardization. They are completed engines, so the original RUL
  labels are legitimate *prior knowledge*.
- **20 validation engines:** each gets two independently trained dynamics
  models on only the first 70% of its chronological sensor observations.
  They receive **no RUL labels**, terminal cluster IDs or later observations.
- Evaluate direct RUL, non-flow RUL and flow RUL at the **last observed prefix
  cycle** using the true RUL solely for scoring.
- Evaluate latent forecasting at physical horizons 1/5/10/20 cycles against
  future validation windows, and include a *persistence* latent forecast.
  These future windows never influence training or selection.
- Original official test engines are **never encoded or evaluated**.
- Hyperparameters are prespecified; no fitting or selection on a validation
  suffix is performed. Validation-prefix adaptation is *transductive and
  unsupervised*; results are exploratory and **not comparable directly**
  with the earlier all-engine `all_no_trend` validation MSE.

This solves the fundamental single-machine identifiability issue: a target
machine that has not yet failed cannot supply its own absolute RUL labels
without leaking future lifetime. A shared *train-engine-only* supervised
RUL reference is used for calibration, while the **dynamics themselves are
machine-specific**. The later global/ensemble approach is deliberately
deferred.

## Rebuilding inputs after experiment-storage loss

Only three derived inputs are required by this study:

1. `.experiments/rul/prepared/cmapss-FD003-seed42-controlled-w30-s42.pt`
   (official FD003 data, 30-cycle, 15 selected sensors, 80/20 train/validation engines);
2. `.experiments/rul/audits/rul-FD003-controlled-patchformer-seed42.checkpoint.json`
   and its selected `.ckpt` under `.experiments/rul/checkpoints/`;
3. `.experiments/rul/prepared/rul-FD003-patchformer-sequence-cfc-flow-seed42-endpoints.pt`
   (frozen sequence embedding normalization, **not** a trained flow model).

The original 12 cluster-conditioned CfC checkpoints and their ablation CSVs
are **not required**. The official NASA C-MAPSS FD003 tables can be downloaded
again from their public source. The prepared RUL tensor and endpoint statistics
can be regenerated deterministically. **Lost PatchFormer weights cannot be
recovered from a CSV or model recipe**: restore the original checkpoint/audit
from a backup or retrain the native controlled model. A retrained encoder has
a new SHA-256 and is a new experimental lineage, so earlier metrics are not
guaranteed to reproduce exactly.

First verify that JuiceFS is mounted, and investigate snapshots/backups **before
writing into a potentially recoverable storage volume**:

```bash
findmnt /mnt/juicefs_prisma
```

With the existing main-worktree NexuML virtual environment activated,
prepend **main** core and **feature-worktree** library paths:

```bash
export PYTHONPATH="/workspaces/NexuFed/NexuML/src:/workspaces/NexuFed/NexuML-predictive/library/src${PYTHONPATH:+:$PYTHONPATH}"
```

A read-only inventory (does not download, train or create directories):

```bash
python -m nexuml_library.evaluation.maintenance.rul_machine_flow_recover \\
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments
```

After confirming that backup restoration is unavailable or complete, rebuild
missing artifacts and, only if necessary, **train a new seed-42 PatchFormer**:

```bash
python -m nexuml_library.evaluation.maintenance.rul_machine_flow_recover \\
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \\
  --rebuild --retrain-missing
```

Omit `--retrain-missing` to prepare public data without launching expensive
training; the command then stops if the checkpoint is absent. Recovery never
overwrites surviving prepared files, endpoint tensors, audits or partial
checkpoints. Native `nexuml resolve` and `nexuml train` run on the current
workstation, with checkpoints persisted under the specified experiment root;
this command does **not** submit Kubernetes jobs. A separate cluster run
requires choosing a permitted GPU node profile.

When recovery reports `"status": "ready"`, run the study below with the
same `--experiment-root`. Its manifest records the new upstream SHA-256s.

## Running the study

From the NEX-233 feature worktree:

```bash
git pull --ff-only
export PYTHONPATH="$PWD/src:$PWD/library/src"

ruff check .
ruff format --check .
ty check
pytest tests/library/test_rul_machine_flow_study.py -v
```

Small pilot on three **validation** engines (no official test):

```bash
python -m nexuml_library.evaluation.maintenance.rul_machine_flow_study \
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \
  --mode causal \
  --device auto \
  --max-engines 3 \
  --prefix-fraction 0.7 \
  --train-steps 300 \
  --batch-size 64
```

Then the complete exploratory validation study:

```bash
python -m nexuml_library.evaluation.maintenance.rul_machine_flow_study \
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \
  --mode causal \
  --device auto \
  --prefix-fraction 0.7 \
  --train-steps 300 \
  --batch-size 64
```

To inspect **retrospective** individual-machine latent reconstruction on
five *completed training engines*:

```bash
python -m nexuml_library.evaluation.maintenance.rul_machine_flow_study \
  --experiment-root /mnt/juicefs_prisma/models/nexuml/.experiments \
  --mode retrospective --device auto \
  --max-engines 5 --train-steps 300
```

Retrospective evaluation fits and tests on the **same completed training
engines**. Its full-life supervised direct RUL model is an **oracle,
in-sample reconstruction bound**, not valid prospective RUL prediction.
Its in-sample trajectory errors should not be ranked against the causal
validation suffix forecasts.

### Output

Files are written to distinct directories under the explicit Prisma root:

```text
rul/within-machine-flow/fd003/<mode>-seed42-prefix70-steps300-max{3|all}-batch64-cal64-int8-ridge10-lr0.001/
    manifest.json
    causal_per_machine.csv
    causal_forecast_horizons.csv
    causal_summary.json
    retrospective_per_machine.csv      # retrospective mode only
    retrospective_summary.json          # retrospective mode only
```

`causal_per_machine.csv` contains capped RUL estimates, signed RUL errors,
estimated degradation slopes, rate-censor flags and dynamic training losses.
`causal_forecast_horizons.csv` contains latent MSE at each forecast horizon
and the persistence error for comparison. `causal_summary.json` provides
engine-macro RUL MAE/RMSE and horizon-specific future-state MSE. All outputs
record provenance and that official test data is untouched.

## Interpretation and subsequent global study

This comparison separates three hypotheses:

- Is the frozen PatchFormer trajectory predictive of actual future
  within-engine sensor embeddings?
- Does conditional Flow Matching improve this trajectory prediction over
  a matched, ordinary direct-future neural estimator?
- Does the resulting local degradation-rate estimate improve capped
  prospective RUL beyond the supervised train-only direct readout?

Negative or right-censored flow results must be reported without hiding
failed-to-degrade cases. Only after those findings should we consider
trajectory-conditioned global flows, ensembles, or mixtures of specialized
machine experts. A final publication-style claim will require an untouched
official test protocol and a new selection/confirmation strategy.
