# NEX-233 Section 3: first real acoustic Flow Matching training

This is a **custom supervised, machine-disjoint acoustic transport**
experiment using the original [MIMII dataset](https://zenodo.org/records/3384388)
(Purohit et al., DCASE 2019). MIMII contains **real microphone
recordings** of normal/abnormal fans, pumps, valves and slide rails. Its
original recordings are 16-kHz, 16-bit, eight-channel WAV files.

**This is not the official unsupervised MIMII anomaly-detection protocol.**
It explicitly uses anomalous recordings as training endpoints; results must
not be compared as if they were official unsupervised MIMII benchmark scores.
The learned flow coordinate is also **not a physical machine lifetime or RUL
estimate**.

## 1. Automatic source download (default)

The initial experiment uses the +6 dB fan archive from [Zenodo
3384388](https://zenodo.org/records/3384388) under CC BY-SA 4.0.
The official ZIP is approximately **10.2 GB compressed**. Leave space for both
the cached ZIP and the extracted recordings. The dataset is not stored in Git.

**No manual download is required.** `MIMIIAcousticEndpointPairs` defaults to
`download=True`. On first dataset build it fetches **only** the selected
machine type and noise level, streams the archive into a disk cache, checks
the exact MD5 checksum published by Zenodo, rejects unsafe ZIP paths and
symlinks, then extracts into a staging directory before final publication.
Existing extracted data take precedence; subsequent runs do not download.

To choose an alternate cache/data root, set `NEXUML_MIMII_ROOT` to an empty
or existing dataset directory. For offline usage, configure
`mimii_acoustic_flow(download=False, root=...)`. You can additionally select
`noise_db=-6`, `0`, or `6`, and `machine_type` from `fan`, `pump`, `slider`,
`valve`; the corresponding official archive and checksum are selected
automatically. A damaged cached ZIP is reported explicitly and must be
deleted before retrying.

**Note:** This is an approximately 10 GB first-run download, so a reliable
connection and substantial free disk space are required. Failed transfers
are not reported as successful. The standard ZIP cache is stored alongside
the extraction directory under `<noise>_dB_<machine>.zip`.
The decoder reads the original 8-channel PCM WAV directly (without
TorchCodec/FFmpeg), downmixes channels to mono and only reads the first
configured window. Nothing is copied or repackaged into a pseudo-dataset.
The data source accepts either the directory containing `id_00` etc.
or its immediate machine-type parent, for example:

```text
data/mimii/6_dB_fan/fan/
  id_00/
    normal/*.wav
    abnormal/*.wav
  id_02/
    normal/*.wav
    abnormal/*.wav
  id_04/
    normal/*.wav
    abnormal/*.wav
  id_06/
    normal/*.wav
    abnormal/*.wav
```

If archive extraction produces a different nesting level, point
`NEXUML_MIMII_ROOT` to the folder with `id_00` or its immediate parent.

## 2. Protocol: truly disjoint machine IDs

| Split | Machine IDs | Permitted usage |
| --- | --- | --- |
| Training | `id_00`, `id_02` | Fit healthy-to-faulty vector field |
| Validation | `id_04` | Loss monitoring, checkpoint selection and early stopping |
| Test | `id_06` | Strictly held out; never used during fitting or checkpoint selection |

Each machine has genuine normal and abnormal recordings. Pair normal and
abnormal examples **within** the same machine ID. A fixed RNG independently
shuffles each set. Pair no more than `min(n_normal, n_abnormal, 128)`
recordings per machine without reusing a file. Metadata records the
individual files, machine IDs, condition and split. Do not change the test
partition after observing its performance.

A source is rejected if machine IDs overlap across the three partitions,
or if a required machine/condition has no original recordings.

## 3. Resolve and run the existing NexuML experiment

From the NEX-233 worktree, with `download=True` as the default:

```bash
export NEXUML_MIMII_ROOT="$PWD/data/mimii/6_dB_fan"
```

Use the **trusted scenario-file entrypoint** for deterministic
`lightning.seed_everything(42, workers=True)`, source-file provenance,
training, and an optional artifact directory:

```bash
nexuml train \
    --scenario-file library/src/nexuml_library/scenarios/flow/mimii_acoustic_flow.py \
    --artifact-dir .experiments/nex233-mimii-provenance
```

The factory is additionally registered as `mimii-acoustic-flow`; a
normal registered-scenario run is also supported:

```bash
nexuml resolve mimii-acoustic-flow -o configs/nex233-mimii.yaml
nexuml train -c configs/nex233-mimii.yaml
```

The first, trusted scenario-file command additionally seeds the global
Torch/Lightning RNG; the second follows the configured source pairing seed
but does not itself guarantee full global RNG determinism.

Initial experiment defaults:

| Parameter | Value |
| --- | --- |
| Input | Unconditional 16 kHz mono acoustic endpoints |
| Audio window | 16,384 samples (~1.024 seconds) |
| Feature map | LMBE: 32 mel bands; FFT 512; hop 256 |
| Coupling | Independent pairing within a machine ID; linear path |
| Flow model | NEX-196 MLP: hidden widths 128, 128 |
| Flow loss | Velocity-matching MSE |
| Euler sampling | 50 integration steps, starting from **healthy** features |
| Optimizer | Adam, LR 0.0003 |
| Mini-batch | 16 |
| Training | Up to 30 epochs; early stopping patience 8 |
| Logging | TensorBoard (MLflow optional, disabled by default) |
| Checkpoints | Best and last monitored at `val/loss` |

Checkpoints are stored under
`.experiments/checkpoints/mimii-acoustic-flow/`. TensorBoard logging
defaults to `.experiments/tensorboard/`:

```bash
tensorboard --logdir .experiments/tensorboard
```

Before claiming a successful first experiment, inspect training and
validation loss curves, report machine IDs and sample counts, and
verify that the best checkpoint exists. A low training loss alone does
**not** demonstrate useful degradation modeling.

For sampling after training, pass a healthy endpoint **log-mel tensor**
to the trained flow model's `sample()`; its result is a transported
feature tensor, not a raw WAV.

## 4. Validation and next stage

The library includes small **fixture-only** integration tests for the
original MIMII layout. They check ID isolation, no repeated recording
pairing, provenance, YAML round-trip, model compilation, gradients, and
sample decoding without downloading any dataset:

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_mimii_acoustic_flow.py -v
pytest tests/library/test_audio_health_to_fault_flow.py -v
pytest tests/library/test_flow_matching.py -v
pytest tests/_registry -v
```

The fixture tests cannot stand in for training on actual MIMII data.
This repository change provides the **real-data training recipe**, not
a pretrained model or empirical validation scores. Synthetic download tests
mock the network and verify checksum checking, offline mode, archive caching
and rejection of unsafe ZIP entries. They do not download the real 10.2 GB ZIP. A production
experiment should log the run checkpoint, training curves and held-out
unit-level outcomes before Section 4's observation-level transport
position estimator is attempted.

A separate experiment can eventually compare the source/target
selection policy and frontend using equal machine-disjoint data and
resources. No RUL labels exist in MIMII and this experiment should not
report RUL scores.
