# Export and reload a model

NexuML's primary portable artifact is a package directory containing the compiled pipeline, resolved configuration, weights, provenance, and runtime dependency information.

There are two common export paths.

## Export the live trained model from a scenario

For local training, declare a train-package export in the scenario:

```python
from nexuml.core.types import ExportSpec

ScenarioSpec(
    ...,
    exports=[ExportSpec(kind="train_package", output="packages/my-model")],
)
```

After `NexuSession.run()` completes, `nexuml train` packages the live trained pipeline/trainer. This avoids guessing a checkpoint path.

## Package an explicit checkpoint

Use the standalone command when you already know which Lightning checkpoint should supply the weights:

```bash
nexuml export my-scenario \
  --checkpoint /path/to/model.ckpt \
  -o packages/my-model
```

`--checkpoint` is optional syntactically, but omission does **not** mean "find the latest checkpoint". Without a supplied checkpoint the command packages the scenario pipeline in its currently constructed state.

## Package layout

A normal package directory can contain:

```text
packages/my-model/
├── pipeline.package
├── state_dict.pt
├── resolved_config.yaml
├── metadata.json
├── requirements.txt
├── training_state.pt      # when training state is available
└── lightning.ckpt         # when a checkpoint/live trainer can provide it
```

`pipeline.package` is the self-contained torch.package payload for NexuML-owned/custom source code. Heavy runtime dependencies such as PyTorch remain external and are recorded in the dependency metadata/`requirements.txt`.

## Reload against the current codebase

```python
from pathlib import Path
from nexuml.core.export import load_package

pipeline, resolved_config, metadata = load_package(Path("packages/my-model"))
```

`load_package` reconstructs the pipeline from the resolved configuration and loads its state dict. Use this when you want the current installed codebase to materialize the runtime.

## Load the packaged pipeline directly

```python
from pathlib import Path
from nexuml.core.export import load_inference_package

pipeline, resolved_config, metadata = load_inference_package(Path("packages/my-model"))
```

This loads the pipeline object stored inside `pipeline.package` directly.

Run it like any compiled pipeline:

```python
from tensordict import TensorDict
import torch

x = TensorDict({"features": torch.randn(1, 64)}, batch_size=[1])
x_out, y_out = pipeline(x, None)
```

Use input/output keys that match the exported scenario.

## Reuse weights for training

An exported directory, package, state dict, SafeTensors file, or compatible Lightning checkpoint can be used as a selective weight source. See [Checkpoints](checkpoints.md).

## SafeTensors and ONNX

The Python export API also provides `export_safetensors(...)` and `export_onnx(...)` for those formats. They are separate from the primary train-package contract.

`ExportSpec` can represent `onnx` and `safetensors`, but the current automatic post-training CLI path handles `train_package`; use the direct Python functions for alternative formats until the CLI automation supports them.

## See also

- [Checkpoints](checkpoints.md)
- [API: export](../reference/api/nexuml/core/export.md)
