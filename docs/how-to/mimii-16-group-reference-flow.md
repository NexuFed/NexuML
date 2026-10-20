# NEX-233: sixteen-group MIMII healthy-reference Flow Matching

## Scientific aim

Determine whether a **single**, healthy-reference-conditioned velocity field
learns fault transport that transfers to unseen industrial machine identities
or whole machine families. Unlike the four-fan pilot, this study uses all
**16 MIMII family-qualified machine IDs**, i.e. four IDs (`00,02,04,06`)
for each of `fan`, `pump`, `slider`, and `valve`.

This is a supervised, **independently paired normal-to-abnormal distribution
transport experiment**, not the official MIMII anomaly-detection protocol.
MIMII provides normal and abnormal machine recordings, not longitudinal RUL
trajectories. The Flow Matching integration coordinate is **not physical age**.

## 1. Extract all sixteen machine groups once

The extractor reuses the existing `MIMIIAcousticEndpointPairs` data source,
automatic checksum-verified Zenodo downloader and PCM decoder, and instantiates
the **same frozen FISHER model once for all four machine families**.
Every group carries `machine_type`, `machine_id`, `group_id`, original
recording identities, original split and FISHER model provenance.

Run from the NexuML repository root:

```bash
python -m nexuml_library.evaluation.maintenance.multifamily_fisher_prepare \
  --data-root data/mimii \
  --output-file \
    .experiments/nex233-multifamily/features/mimii16-fisher-independent-raw.pt \
  --noise-db 6 \
  --max-pairs-per-machine 128 \
  --clip-num-samples 16384 \
  --batch-size 16 \
  --device cuda \
  --model-name jiangab/FISHER-small-0723 \
  --seed 42
```

Auto-download is enabled by default. The four original `6_dB_*.zip`
archives can be large. Use `--no-download` when verified extracted data
already exist and network access is undesired. Existing fan recordings in
`data/mimii/6_dB_fan` will be reused; the three additional types are
downloaded only if missing. Existing output is never overwritten.

For strict model reproducibility, optionally pin `--revision` to an actual
FISHER Hugging Face commit. The resulting feature file is a standard
`nex233-features-v1` tensor with `[N,2,1536]` for FISHER-small, plus
a small companion JSON manifest.

**Important:** The original `id_06` test IDs were previously unsealed.
Including them in training folds is therefore an **exploratory** expansion,
not a valid restoration of the original sealed test set.

## 2. Prepare both holdout protocols for multiple seeds

```bash
python -m nexuml_library.evaluation.maintenance.multifamily_reference_prepare \
  --source-file \
    .experiments/nex233-multifamily/features/mimii16-fisher-independent-raw.pt \
  --output-dir .experiments/nex233-multifamily/folds \
  --protocols both \
  --seeds 42 43 44 45 46 \
  --reference-size 32 \
  --validation-fraction 0.2
```

This creates the following distinct protocols:

| Protocol | Training groups | Held-out groups | Folds per seed |
|---|---:|---:|---:|
| `machine` | 15 | 1 machine identity | 16 |
| `family` | 12 (three complete families) | 4 IDs of one unseen family | 4 |

All 20 folds are constructed per seed. The split index is saved at
`.experiments/nex233-multifamily/folds/index.json`. Each individual fold
contains:

- A **training-only** artifact: training machine healthy references,
  independent healthy/faulty query pairs, and recording-disjoint validation
  queries. Every training identity has its own fixed `K=32` support set.
- One or four **assessment-only** artifacts: the unseen machines' healthy
  support sets and disjoint held-out healthy/faulty assessment queries.
- Train-only feature standardization: fit on healthy training-machine
  queries and healthy training-machine supports. No validation or unseen
  machine observation contributes to the fitted statistics.

The support recordings do **not** count as flow training query pairs.
Even the held-out machine's healthy support is read by the runner **only
after checkpoint selection**. It provides conditioning but no gradient
or hyperparameter signal. The held-out faulty recordings are accessed
only during final assessment.

Note that split seeds resample recordings of the **same 16 machine IDs**.
They are not independent new machines or independent confidence-interval
replicates at the machine-population level.

## 3. Pilot a single unseen machine

Before running all folds, run one matched pair:

```bash
python -m nexuml_library.evaluation.maintenance.multifamily_reference_run \
  --prepared-index .experiments/nex233-multifamily/folds/index.json \
  --output-root .experiments/nex233-multifamily/pilot-machine \
  --protocol machine --seeds 42 \
  --heldout-machine fan:id_04 \
  --modes reference constant \
  --width 128 --context-dim 32 --encoder-width 128 \
  --num-steps 50 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --patience 25 --accelerator gpu
```

This isolates one fold and compares healthy-reference DeepSets conditioning
against an identical-width constant-context control. Use a fresh output root
for every new pilot or different configuration; existing checkpoints and
reports are deliberately not overwritten.

## 4. Known-family machine transfer: four fan identities

To test whether healthy-reference Flow Matching transfers to **new machines
within a known type**, run four separate leave-one-machine-out folds.

Every fold trains on **15 distinct machine IDs**: three other fan IDs and
all twelve pump/slider/valve IDs. Only one fan machine is excluded from
training, validation, and fitted normalization statistics at a time.

The existing multi-family fold index already contains these partitions;
**do not rerun FISHER extraction or fold preparation**.

```bash
python -m nexuml_library.evaluation.maintenance.multifamily_reference_run \
  --prepared-index .experiments/nex233-multifamily/folds/index.json \
  --output-root .experiments/nex233-multifamily/pilot-known-fan \
  --protocol machine --seeds 42 \
  --heldout-family fan \
  --modes reference constant \
  --width 128 --context-dim 32 --encoder-width 128 \
  --num-steps 50 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --patience 25 --accelerator gpu
```

The command executes **eight separate trainings**: four held-out fan IDs ×
two context variants. Output:
`.experiments/nex233-multifamily/pilot-known-fan/aggregate-machine.json`.
A zero-transport baseline is always 0% relative SWD reduction.

The earlier unseen-family pilot remains separately preserved as
`.experiments/nex233-multifamily/pilot-family/aggregate-family.json`.

Once both are complete, compare the **same four fan machine identities and
seed 42** under both generalization protocols:

```bash
python -m nexuml_library.evaluation.maintenance.machine_family_comparison \
  --machine-aggregate \
    .experiments/nex233-multifamily/pilot-known-fan/aggregate-machine.json \
  --family-aggregate \
    .experiments/nex233-multifamily/pilot-family/aggregate-family.json \
  --family fan \
  --output-file \
    .experiments/nex233-multifamily/comparisons/fan-known-vs-unseen-seed42.json
```

The comparison reports per-machine SWD reduction for both contexts under
both protocols; the number of machines with positive transport; the
conditioning advantage within each protocol; and known-family vs
unseen-family transfer advantage. It refuses mismatched seeds, folds,
duplicate records, or incomplete four-ID families.

**Interpret carefully:** The raw embedding data and held-out machine IDs are
shared, but the two protocols use different training-only standardization
statistics. Compare *relative* SWD reductions, not absolute SWD distances.
Different split seeds resample the same physical machine groups, not new
independent machines.

## 5. Pilot an unseen family

```bash
python -m nexuml_library.evaluation.maintenance.multifamily_reference_run \
  --prepared-index .experiments/nex233-multifamily/folds/index.json \
  --output-root .experiments/nex233-multifamily/pilot-family \
  --protocol family --seeds 42 \
  --heldout-machine family:fan \
  --modes reference constant \
  --width 128 --context-dim 32 --encoder-width 128 \
  --num-steps 50 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --patience 25 --accelerator gpu
```

All four fan IDs are omitted from training and checkpoint selection.
The runner evaluates `fan:id_00`, `fan:id_02`, `fan:id_04` and
`fan:id_06` separately after selection, using each one's own healthy
reference set. The family-level performance is an **unweighted average**
of the four machine-level relative SWD reductions, not a pooled distance.

## 6. Full benchmark

For each protocol, choose a fresh output root and **omit
`--heldout-machine`**. Set `--seeds 42 43 44 45 46` and
`--modes reference constant` to run all folds. Example:

```bash
python -m nexuml_library.evaluation.maintenance.multifamily_reference_run \
  --prepared-index .experiments/nex233-multifamily/folds/index.json \
  --output-root .experiments/nex233-multifamily/full-family \
  --protocol family --seeds 42 43 44 45 46 \
  --modes reference constant \
  --width 128 --context-dim 32 --encoder-width 128 \
  --num-steps 50 --lr 0.0003 --batch-size 16 \
  --max-epochs 300 --patience 25 --accelerator gpu
```

For the full 16-fold machine protocol, use `--protocol machine` and
a different fresh output root. The runner generates separate
`seed<seed>/<protocol>/<mode>/` reports/checkpoints and a
root-level `aggregate-<protocol>.json`.

The full five-seed machine protocol runs **160 trainings**:
16 machine folds × 2 context variants × 5 seeds. The family protocol
runs **40 trainings**: 4 family folds × 2 variants × 5 seeds.

### Evaluation

Best checkpoints are chosen only using unweighted mean validation SWD
across **training machine groups**. After training, each unseen machine
is assessed from **real healthy initial observations**, not Gaussian
noise. We report:

- `initial_swd`, `transported_swd`,
  `swd_relative_reduction = 1 - transported_swd / initial_swd`
- Machine-wise and family-wise macro-reductions, count of machines
  better than the **no-transport (0%) baseline**, and context swapping
  sensitivity diagnostics.
- Paired improvement of `reference` relative to `constant` for each
  identical held-out machine/seed/fold.

The dataset's original machine-family effects and the physical meaning
of faults may vary. A positive metric is not RUL, not proof of a universal
fault mechanism, and not an automatic indication of meaningful
degradation trajectories.

### Quality checks

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_multifamily_reference_flow.py -v
pytest tests/library/test_machine_family_comparison.py -v
pytest tests/library/test_healthy_reference_flow.py -v
pytest tests/_registry -v
```
