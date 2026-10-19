# Pretrained FISHER industrial-signal embeddings

The optional public FISHER encoder lives at
`nexuml_library.layers.model.fisher.FISHEREncoder`.
It wraps the official MIT-licensed FISHER industrial-signal model rather
than copying or vendoring any of its weights or implementation.

Sources: [FISHER upstream](https://github.com/jianganbai/FISHER),
[FISHER-small checkpoint](https://huggingface.co/jiangab/FISHER-small-0723).

## Install and trust the checkpoint

Install the library's optional pretrained-model requirements with the
normal NexuML workspace workflow:

```bash
uv sync --locked --all-packages --all-extras
uv pip install einops
```

The upstream checkpoint's custom model code uses einops. It is not yet
included in the workspace lockfile, so install it explicitly after sync.
The public library remains importable without Transformers or einops.

FISHER currently loads with Hugging Face's
`AutoModel.from_pretrained` and requires
`trust_remote_code=True`. **Only load trusted model repositories.**
Use an immutable model revision for reproducibility:

```python
from nexuml import nn_module
from nexuml_library.layers.model.fisher import FISHEREncoder

fisher = nn_module(
    FISHEREncoder,
    model_name="jiangab/FISHER-small-0723",
    sample_rate=16000,
    pooling="native",
    frozen=True,
    trust_remote_code=True,
    revision=None,  # Pin the audited Hugging Face commit for production runs.
)
```

No checkpoint files are bundled with NexuML. The first construction
downloads the model into the Hugging Face cache. Subsequent runs can use
`local_files_only=True` when all model files
are already present.

This implementation does not patch Transformers internals. If the pinned
remote-code model is incompatible with the installed Transformers major
version, use a compatible Transformers release and pin it in the
reproducible execution environment rather than globally monkey-patching
other models.

## NexuML TensorDict routing

Use FISHER exactly like any other generic PyTorch module:

```python
from nexuml import nn_module
from nexuml.core.types import LayerSpec
from nexuml_library.layers.model.fisher import FISHEREncoder

fisher_layer = LayerSpec(
    component=nn_module(
        FISHEREncoder,
        model_name="jiangab/FISHER-small-0723",
        sample_rate=16000,
        frozen=True,
        pooling="native",
    ),
    keys_in=["healthy_waveform"],
    keys_out=["healthy_embedding"],
)
```

The configuration is serializable and restorable through NexuML's existing
generic `NnModule` identity; no registry,
compiler or core modifications are needed.

For paired Flow Matching, healthy and faulty recordings must pass through
the **same frozen feature map**. When composing an experiment with two
independent FISHER layer instances, ensure both load the identical pinned
checkpoint and use identical preprocessing and pooling settings. For
efficient production training, cache extracted embeddings and use a shared
model instance or shared offline feature-extraction process to avoid
duplicating GPU memory. The current generic adapter does not automatically
share two instantiated backbone instances.

## Preprocessing and embeddings

FISHER receives **waveforms**, not precomputed LMBE/log-mel features.
The wrapper reproduces the official published recipe:

1. Per-recording waveform mean removal.
2. STFT with a Hann window of 25 ms, hop 10 ms and center=False.
3. Log magnitude with additive epsilon = 1e-10.
4. Published normalization: (log_magnitude + 3.017344307886898) /
   (2 * 2.1531635155379805).
5. Time truncation to 1024 frames and frequency padding to at least the
   checkpoint's band width.
6. Forward through FISHER and return one rank-2 embedding tensor.

Input: float tensor shaped `[B, T]` or
`[B, C, T]`. Multichannel inputs are
averaged to mono. The caller is responsible for supplying audio at the
configured sample rate and sufficiently long clips (at least one FFT
window). The wrapper does **not** resample audio.

Pooling options:

| Method | Meaning |
| --- | --- |
| `native` (default) | Official
  `model.extract_features()` output;
  concatenates frequency-band representations |
| `mean` | Experimental mean across the
  feature-only token sequence |
| `mean_std` | Experimental concatenation of
  token mean and standard deviation |

The native output dimension depends on sample rate and checkpoint band
width. Do not hard-code the embedding dimension until the model is
materialized and its actual output shape is known.

## Real-checkpoint smoke test (separate from offline CI)

Once the official model weights and approved remote Python code are
available in the execution environment, explicitly test compatibility
with the project's installed Transformers version:

```bash
python - <<'PY'
import torch
from nexuml_library.layers.model.fisher import FISHEREncoder

encoder = FISHEREncoder(
    model_name="jiangab/FISHER-small-0723",
    local_files_only=False,
).eval()
with torch.no_grad():
    embedding = encoder(torch.randn(2, 16384))
print("Embedding shape:", tuple(embedding.shape))
assert embedding.ndim == 2 and embedding.shape[0] == 2
assert torch.isfinite(embedding).all()
PY
```

This explicitly loads the external checkpoint. It may download weights
and execute trusted remote model code; it is intentionally **not** part of
the regular offline unit-test suite. In particular, verify compatibility
with the locked Transformers 5.x release before starting GPU ablations.

## Freezing and adaptation

The backbone is frozen by default. Calling
`encoder.train()` does not enable dropout
or gradients inside the frozen FISHER model. Gradients can still flow
through the downstream trainable Flow Matching vector field.

Set `frozen=False` only for a separately
controlled finetuning experiment, with a suitable training budget. Using a
larger pretrained embedding does not by itself increase the trainable
flow network's capacity.

## Verification

```bash
ruff check .
ruff format --check .
ty check
pytest tests/library/test_fisher_encoder.py -v
pytest tests/_registry -v
```

The tests mock pretrained model loading: **no Hugging Face weights or
network access are required**. Real checkpoint compatibility and a
quantitative acoustic validation are separate empirical checks to run
before including FISHER in the NEX-233 OT/normalization ablation.
