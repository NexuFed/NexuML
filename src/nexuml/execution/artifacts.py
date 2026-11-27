"""Requested exports reuse existing NexuML export functions and native results."""

from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


def export_local_training(result: Any, requests: list[Any]) -> list[dict[str, str]]:
    """Write one RL export after all ranks participate in checkpointing.

    Returns:
        Artifact references on the exporting rank, or none on non-owner ranks.
    """
    from nexuml.training.reinforcement import NexuRLLightningModule

    module = result.lightning_module
    if not isinstance(module, NexuRLLightningModule) or result.trainer.world_size == 1:
        return export_requests(result, requests)
    with TemporaryDirectory(prefix="nexuml-rl-export-") as directory:
        checkpoint = None
        if any(request.kind == "train_package" for request in requests):
            checkpoint = str(Path(directory) / "checkpoint.ckpt")
            result.trainer.save_checkpoint(checkpoint)
        with module._rank_guard("export"):
            if result.trainer.is_global_zero:
                return export_requests(result, requests, checkpoint)
    return []


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
                source.policy if getattr(source, "policy", None) is not None else source.pipeline,
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
