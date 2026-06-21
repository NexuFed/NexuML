# Model export and reload

NexuML can export a trained pipeline to a portable artifact for inference, deployment, or fine-tuning. Three formats are supported: `package` (default), `safetensors`, and `onnx`.

## Prerequisites

- NexuML installed (`uv sync`)
- A trained scenario with saved checkpoints

## Export via CLI

```bash
# Export the trained pipeline for a scenario
nexuml export <scenario-name>

# With a specific checkpoint
nexuml export my-scenario --checkpoint .experiments/checkpoints/my-scenario/best.ckpt

# Custom output directory
nexuml export my-scenario -o packages/my-scenario/
```

| Option | Description |
|---|---|
| `SCENARIO_NAME` | Scenario name to export (required) |
| `--output` / `-o PATH` | Export directory (default: `exported_model`) |
| `--checkpoint PATH` | Checkpoint to export (latest if not specified) |

## Export layout (package format)

```
packages/my-scenario/
├── state_dict.pt       # model weights as PyTorch state dict
├── config.yaml         # full ResolvedConfig (pipeline + spec)
└── metadata.json       # provenance: scenario name, version, eval results
```

## Load and infer in Python

### Full package reload

```python
from nexuml.core.export import load_package

pipeline = load_package("packages/my-scenario/")
```

Returns the compiled `CompiledPipeline`. Use `pipeline(tensordict)` to run inference.

### Inference-only package

```python
from nexuml.core.export import load_inference_package
import tensordict as td
import torch

pkg = load_inference_package("packages/my-scenario/")
x = td.TensorDict({"features": torch.randn(1, 64)}, batch_size=[1])
output = pkg.infer(x)
print(output)   # TensorDict with all output keys
```

`load_inference_package` returns a lightweight wrapper suitable for deployment — it does not require the full NexuML training stack.

## Reload for fine-tuning (selective weight loading)

To load weights from an exported package into a new scenario for fine-tuning:

```python
from nexuml.core.types import ScenarioSpec, CheckpointLoadSpec

ScenarioSpec(
    name="fine_tuned",
    checkpoint=CheckpointLoadSpec(
        source="packages/my-scenario/state_dict.pt",
        include=["encode"],          # load only the encoder stages
        exclude=["head"],            # skip the classification head
        allow_missing=True,          # new layers without weights are OK
        allow_shape_mismatch=True,   # mismatched shapes are skipped with a warning
        freeze_loaded=False,         # set True to freeze loaded weights
    ),
    ...
)
```

See [Checkpoints](checkpoints.md) for the full `CheckpointLoadSpec` reference.

## Alternative export formats

### SafeTensors

```python
from nexuml.core.export import export_safetensors

export_safetensors(pipeline, output_dir="packages/my-scenario-safetensors/")
```

Or declare in `ScenarioSpec.exports`:

```python
from nexuml.core.types import ExportSpec

ScenarioSpec(
    name="my-scenario",
    exports=[ExportSpec(kind="safetensors", output="packages/my-scenario-st/")],
    ...
)
```

### ONNX

```python
from nexuml.core.export import export_onnx
import torch

# dummy_input must match the pipeline's first-layer input shape
dummy_input = {"features": torch.randn(1, 64)}
export_onnx(pipeline, dummy_input, output_path="packages/my-scenario.onnx")
```

Or in exports spec:

```python
ExportSpec(kind="onnx", output="packages/my-scenario.onnx")
```

## Smoke test

The `smoke` command includes the full export → reload → infer cycle:

```bash
nexuml smoke my-scenario --max-epochs 2
```

## Implementation map

- `src/nexuml/core/export.py` — `export_package`, `load_package`, `load_inference_package`, `infer`, `export_safetensors`, `export_onnx`
- `src/nexuml/core/types.py` — `ExportSpec`, `CheckpointLoadSpec`
- `src/nexuml/cli/main.py` — `export` command

## See also

- [Checkpoints](checkpoints.md)
- [Dataset export](export-dataset.md)
- [Backends](../reference/backends.md)
