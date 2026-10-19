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

## 1. Acquire the source recordings

The initial experiment uses the +6 dB fan archive from [Zenodo
3384388](https://zenodo.org/records/3384388) under CC BY-SA 4.0. This
archive is approximately **10.2 GB compressed**. Leave sufficient disk
space for extraction and checkpoints. Original data are *not* committed
to GitHub.

One way to obtain it (from a terminal with external network access):

\`\`\`bash
mkdir -p data/mimii/6_dB_fan
wget -c "https://zenodo.org/records/3384388/files/6_dB_fan.zip?download=1" \
    -O data/mimii/6_dB_fan.zip
unzip -q data/mimii/6_dB_fan.zip -d data/mimii/6_dB_fan
find data/mimii/6_dB_fan -type d -name id_00
\`\`\`

The decoder reads the original 8-channel PCM WAV directly (without
TorchCodec/FFmpeg), downmixes channels to mono and only reads the first
configured window. Nothing is copied or repackaged into a pseudo-dataset.
The data source accepts either the directory containing \`id_00\` etc.
or its immediate machine-type parent, for example:

\`\`\`text
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
\`\`\`

If archive extraction produces a different nesting level, point
\`NEXUML_MIMII_ROOT\` to the folder with \`id_00\` or its immediate parent.

## 2. Protocol: truly disjoint machine IDs

| Split | Machine IDs | Permitted usage |
| --- | --- | --- |
| Training | \`id_00\`, \`id_02\` | Fit healthy-to-faulty vector field |
| Validation | \`id_04\` | Loss monitoring, checkpoint selection and early stopping |
| Test | \`id_06\` | Strictly held out; never used during fitting or checkpoint selection |

Each machine has genuine normal and abnormal recordings. Pair normal and
abnormal examples **within** the same machine ID. A fixed RNG independently
shuffles each set. Pair no more than \`min(n_normal, n_abnormal, 128)\`
recordings per machine without reusing a file. Metadata records the
individual files, machine IDs, condition and split. Do not change the test
partition after observing its performance.

A source is rejected if machine IDs overlap across the three partitions,
or if a required machine/condition has no original recordings.

## 3. Resolve and run the existing NexuML experiment

From the NEX-233 worktree:

\`\`\`bash
export NEXUML_MIMII_ROOT="$PWD/data/mimii/6_dB_fan"
nexuml registry list data
nexuml registry list scenarios
\`\`\`

Use the **trusted scenario-file entrypoint** for deterministic
\`lightning.seed_everything(42, workers=True)\`, source-file provenance,
training, and an optional artifact directory:

\`\`\`bash
nexuml train \
    --scenario-file library/src/nexuml_library/scenarios/flow/mimii_acoustic_flow.py \
    --artifact-dir .experiments/nex233-mimii-provenance
\`\`\`

The factory is additionally registered as \`mimii-acoustic-flow\`; a
normal registered-scenario run is also supported:

\`\`\`bash
nexuml resolve mimii-acoustic-flow -o configs/nex233-mimii.yaml
nexuml train -c configs/nex233-mimii.yaml
\`\`\`

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
| Checkpoints | Best and last monitored at \`val/loss\` |

Checkpoints are stored under
\`.experiments/checkpoints/mimii-acoustic-flow/\`. TensorBoard logging
defaults to \`.experiments/tensorboard/\`:

\`\`\`bash
tensorboard --logdir .experiments/tensorboard
\`\`\`

Before claiming a successful first experiment, inspect training and
validation loss curves, report machine IDs and sample counts, and
verify that the best checkpoint exists. A low training loss alone does
**not** demonstrate useful degradation modeling.

For sampling after training, pass a healthy endpoint **log-mel tensor**
to the trained flow model's \`sample()\`; its result is a transported
feature tensor, not a raw WAV.

## 4. Validation and next stage

The library includes small **fixture-only** integration tests for the
original MIMII layout. They check ID isolation, no repeated recording
pairing, provenance, YAML round-trip, model compilation, gradients, and
sample decoding without downloading any dataset:

\`\`\`bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_mimii_acoustic_flow.py -v
pytest tests/library/test_audio_health_to_fault_flow.py -v
pytest tests/library/test_flow_matching.py -v
pytest tests/_registry -v
\`\`\`

The fixture tests cannot stand in for training on actual MIMII data.
This repository change provides the **real-data training recipe**, not
a pretrained model or empirical validation scores. A production
experiment should log the run checkpoint, training curves and held-out
unit-level outcomes before Section 4's observation-level transport
position estimator is attempted.

A separate experiment can eventually compare the source/target
selection policy and frontend using equal machine-disjoint data and
resources. No RUL labels exist in MIMII and this experiment should not
report RUL scores.
