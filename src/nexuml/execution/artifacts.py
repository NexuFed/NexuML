"""Requested exports reuse existing NexuML export functions and native results."""

from pathlib import Path
from typing import Any


def export_requests(
    source: Any, requests: list[Any], checkpoint: str | None = None
) -> list[dict[str, str]]:
    """Export configured formats from the actual pipeline/trainer.

    Returns:
        Existing-format local artifact references, not invented remote downloads.
    """
    from nexuml.core.export import export_onnx, export_package, export_safetensors

    artifacts = []
    for request in requests:
        output = Path(request.output or "exported_model")
        if request.kind == "train_package":
            path = export_package(
                source.pipeline,
                output,
                lightning_module=source.lightning_module,
                trainer=source.trainer if checkpoint is None else None,
                checkpoint_path=checkpoint,
            )
            paths = [file for file in path.rglob("*") if file.is_file()]
        elif request.kind == "safetensors":
            path = export_safetensors(
                source.pipeline, output, include=request.include, exclude=request.exclude
            )
            paths = [path, path.with_suffix(".json")]
        else:
            paths = [export_onnx(source.pipeline, output)]
        artifacts.extend({"path": str(path.resolve()), "kind": request.kind} for path in paths)
    return artifacts
