"""Restore frozen Transformer, Conformer or CfC full-sequence Flow models.

Only validation-selected local Lightning checkpoints from the matching
NEX-233 sequential flow run are accepted. No official test RUL labels
participate in checkpoint discovery.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Literal

import torch
from torch import Tensor

from nexuml.core.components import LayerBuildContext
from nexuml_library.layers.model.flow import SequenceFlow, _FlowRuntime

SequenceArchitecture = Literal["transformer", "conformer", "cfc"]
_SEQUENCE_SHAPE = (30, 128)


def sequence_flow_run(
    subset: Literal["FD001", "FD003"],
    seed: int,
    architecture: SequenceArchitecture,
) -> str:
    """Return the exact registered sequential-Flow upstream run name.

    Returns:
        Run identifier used by the native checkpoint callback.
    """
    return f"rul-{subset}-patchformer-sequence-{architecture}-flow-seed{seed}"


def resolve_sequence_flow_checkpoint(
    subset: Literal["FD001", "FD003"],
    seed: int,
    architecture: SequenceArchitecture,
) -> Path:
    """Choose native SWD-best Flow checkpoint, excluding last.ckpt.

    Returns:
        Path to the checkpoint with best validated transport SWD.

    Raises:
        FileNotFoundError: If this exact sequential flow has not been trained.
        ValueError: If multiple checkpoints have no trustworthy SWD metadata.
    """
    run = sequence_flow_run(subset, seed, architecture)
    directory = Path(".experiments/rul/checkpoints") / run
    candidates = sorted(
        path.resolve() for path in directory.glob("*.ckpt")
        if path.name != "last.ckpt"
    )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        scored: list[tuple[float, Path]] = []
        for candidate in candidates:
            # Trusted local Lightning files contain ModelCheckpoint callback state.
            saved = torch.load(candidate, map_location="cpu", weights_only=False)
            if not isinstance(saved, dict):
                raise ValueError(f"Invalid Flow checkpoint: {candidate}")
            callbacks = saved.get("callbacks")
            if not isinstance(callbacks, dict):
                raise ValueError(
                    "Ambiguous Flow checkpoints without validation selection: "
                    f"{directory}"
                )
            selected: list[float] = []
            for key, value in callbacks.items():
                if (
                    not isinstance(key, str)
                    or "ModelCheckpoint" not in key
                    or "val/transport_swd" not in key
                    or not isinstance(value, dict)
                ):
                    continue
                score = value.get("best_model_score")
                if isinstance(score, (float, int, Tensor)):
                    scalar = float(score)
                    if not math.isfinite(scalar):
                        raise ValueError("Nonfinite checkpoint validation SWD")
                    selected.append(scalar)
            if len(selected) != 1:
                raise ValueError(
                    f"Cannot identify one validation-selected Flow score: {candidate}"
                )
            scored.append((selected[0], candidate))
        return min(scored, key=lambda item: (item[0], str(item[1])))[1]
    raise FileNotFoundError(
        "Missing validation-selected sequential Flow checkpoint; first train "
        f"maintenance-rul-patchformer-sequence-{architecture}-"
        f"{subset.lower()}-s{seed}"
    )


def _file_sha256(path: Path) -> str:
    """Hash the complete trusted local training checkpoint.

    Returns:
        SHA-256 hexadecimal checkpoint digest.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_sequence_flow(
    checkpoint: Path, architecture: SequenceArchitecture, seed: int
) -> tuple[_FlowRuntime, str]:
    """Load and freeze the exact sequence vector field with strict key matching.

    Args:
        checkpoint: Trusted local native Lightning checkpoint.
        architecture: Transformer, Conformer or CfC trained upstream.
        seed: Matching model initialization seed.

    Returns:
        Frozen flow runtime with 50-step Euler sampler and checkpoint SHA-256.

    Raises:
        ValueError: If the checkpoint has missing, ambiguous or mismatched keys.
    """
    definition = SequenceFlow(
        architecture=architecture,
        model_dim=128,
        depth=2,
        attention_heads=4,
        conv_kernel=7,
        dropout=0.0,
        num_steps=50,
        initialization_seed=seed,
    )
    runtime = definition.build(
        LayerBuildContext(
            input_sizes={"flow_state": _SEQUENCE_SHAPE, "flow_time": (1,)},
            keys_in=["flow_state", "flow_time"],
            keys_out=["predicted_velocity"],
        )
    )
    if not isinstance(runtime, _FlowRuntime):
        raise ValueError("Expected the native NexuML sequence Flow runtime")
    # Only load checkpoints produced by the user's own trusted training runs.
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError("Sequence flow checkpoint is not a Lightning dictionary")
    state = payload.get("state_dict", payload)
    if not isinstance(state, dict):
        raise ValueError("Sequence flow checkpoint has no parameter state_dict")
    matched: dict[str, Tensor] = {}
    for key, expected in runtime.state_dict().items():
        possible = [
            value for name, value in state.items()
            if isinstance(name, str)
            and (name == key or name.endswith(f".{key}"))
            and isinstance(value, Tensor)
        ]
        if len(possible) != 1 or possible[0].shape != expected.shape:
            raise ValueError(
                f"Missing, ambiguous or incompatible sequence flow parameter: {key}"
            )
        matched[key] = possible[0]
    runtime.load_state_dict(matched, strict=True)
    runtime.eval()
    runtime.requires_grad_(False)
    return runtime, _file_sha256(checkpoint)
