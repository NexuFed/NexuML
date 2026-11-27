"""Shared renderer validation and bounded frame sizing for rollout capture."""

import json
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory

import numpy as np

from nexuml.core.types import RLVideoSpec
from nexuml.interaction.runtime import RGBEnvironmentRuntime


def capture_rgb(
    renderer: RGBEnvironmentRuntime, max_size: int, *, expected_size: list[int] | None
) -> tuple[np.ndarray, list[int]]:
    """Read one owned RGB frame with stable native dimensions.

    Returns:
        A copied, bounded frame and its source width/height.

    Raises:
        ValueError: If the renderer output format or dimensions are invalid.
    """
    frame = renderer.render_frame()
    if (
        not isinstance(frame, np.ndarray)
        or frame.dtype != np.uint8
        or frame.ndim != 3
        or frame.shape[2] != 3
        or not frame.size
    ):
        raise ValueError("Recording frames must be nonempty CPU uint8 HWC RGB arrays")
    height, width = frame.shape[:2]
    source_size = [width, height]
    if expected_size is not None and expected_size != source_size:
        raise ValueError("Recording frame dimensions changed within the episode")
    ratio = min(1.0, max_size / max(width, height))
    size = (max(1, round(width * ratio)), max(1, round(height * ratio)))
    if size != (width, height):
        from PIL import Image

        frame = np.asarray(Image.fromarray(frame).resize(size, Image.Resampling.BILINEAR))
    return frame.copy(), source_size


def write_playback_media(
    output: Path, frames: list[np.ndarray], metadata: dict, video: RLVideoSpec
) -> None:
    """Encode before publishing media and a final manifest in a new directory."""
    from nexuml.tracking.logger import _encode_evaluation_gif, _encode_evaluation_mp4

    encoders = {
        "gif": _encode_evaluation_gif,
        "mp4": _encode_evaluation_mp4,
    }
    formats = ("gif", "mp4") if video.format == "both" else (video.format,)
    with TemporaryDirectory(dir=output.parent, prefix=".nexuml-play-") as staged:
        root = Path(staged)
        encoded = {}
        for fmt in formats:
            encoded[fmt] = encoders[fmt](frames, root / f"playback.{fmt}", metadata["fps"])
        manifest = {**metadata, "encoders": encoded}
        (root / "playback.json").write_text(
            json.dumps(manifest, indent=2, allow_nan=False) + "\n", "utf-8"
        )
        output.mkdir(exist_ok=False)
        for name in (*(f"playback.{fmt}" for fmt in formats), "playback.json"):
            copyfile(root / name, output / name)
