# Acoustic health-to-fault Flow Matching (NEX-233, step 2)

This scenario learns an **unconditional** transport from real *healthy* acoustic
recordings to real *faulty* acoustic recordings. It does **not** interpret
transport integration time as physical degradation time or predict RUL yet.

The scenario uses only the **audio modality**. The bearing datasets introduced
in step 1 contain accelerometer *vibration* signals and are not silently
relabeled as microphone audio. Those datasets remain available for a later,
explicitly scoped experiment.

## Required data partition

Prepare one or more acoustic-source roots with WAV recordings:

\`\`\`text
audio_endpoints/
  train/
    healthy/
      machine-a/
        normal-001.wav
    faulty/
      machine-b/
        abnormal-001.wav
  val/
    healthy/
      machine-c/
        normal-002.wav
    faulty/
      machine-d/
        abnormal-002.wav
  test/                         # optional; not read by default
    healthy/...
    faulty/...
\`\`\`

The folder directly below each condition class acts as the unit/session/group
identifier; when there are no subfolders, each filename stem is its own group.
**The same group identifier must not occur in train, validation, or test
simultaneously.** Both acoustic conditions are required in each enabled split.
Missing faults are an error; they are never synthesized or sourced from a held-out
anomaly detection test set.

Do not feed DCASE Task 2 official test anomalies into this scenario's training
tree. Standard DCASE Task 2 training sets predominantly contain **healthy**
examples, so they are not, by themselves, a valid pair of training endpoint
distributions. Use a dataset with appropriately licensed, genuinely labeled
healthy and faulty *training* recordings, with an independently held-out
validation split.

Within each split, healthy and faulty records are independently permuted using
a fixed seed and paired by cycling over the two record lists. This forms an
independent coupling, **not** a physically matched degradation trajectory.
Pairing does not cross train/validation boundaries.

## NexuML scenario

\`\`\`python
from nexuml_library.scenarios.flow.audio_health_to_fault import (
    audio_health_to_fault_flow,
)

scenario = audio_health_to_fault_flow(
    data_roots=["/datasets/machine-sounds/source-a", "/datasets/machine-sounds/source-b"],
    sample_rate=16000,
    clip_num_samples=2048,
    n_fft=256,
    hop_length=128,
    n_mels=16,
    hidden_dims=[128, 128],
    num_steps=30,
    batch_size=16,
    max_epochs=20,
)
\`\`\`

Each file is decoded as mono float32 audio. The runtime verifies the sample
rate (no implicit resampling), truncates the clip to the configured sample count,
or right-pads short recordings. No label-based conditioning or guidance occurs.

The composed NexuML pipeline is:

1. \`LMBE\` (existing library component) on each endpoint waveform,
   yielding \`healthy_spectrum\` and \`faulty_spectrum\`.
2. \`PairedLinearFlowPath\`, using NEX-196's \`_linear_path\` on **both**
   observed endpoints to produce \`flow_state\`, \`flow_time\`, and
   \`flow_target_velocity\`.
3. \`Flow\` (NEX-196's existing time-conditioned MLP vector field).
4. \`FlowMatchingLoss\` (NEX-196's existing velocity regression loss).

The \`Flow\` model's Euler sampler should be given **healthy endpoint
spectra**, not Gaussian noise, for this experiment. The output is a log-mel
representation of a transported observation, **not** a generated waveform.
Audio reconstruction is deliberately out of scope.

All registered components are immutable public definitions with private
runtimes and TensorDict routing; the NexuML core is unchanged.
The \`audio_endpoint_data\` fragment uses
\`DatasetSpec(split_type="keep", modality="audio")\` and the native Torch
loader. All sources must share sample rate and feature geometry.

## Scope and limitations

This is step 2: training and sampling only. No localization of an unseen
sample on the learned flow (step 3) and no flow-time-to-RUL calibration (step 4).
No test-cohort tuning or RUL score is performed. For step 5, benchmark
evaluation needs complete, unit-disjoint held-out cohorts.

Fixed first-window selection and short clips make this a tractable initial
experiment, not a production acoustic condition-monitoring benchmark.
Compare alternative windowing, frontend normalization and condition-matched
sampling in later ablations.

## Verification

\`\`\`bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_audio_health_to_fault_flow.py -v
pytest tests/library/test_flow_matching.py -v
pytest tests/_registry -v
\`\`\`
