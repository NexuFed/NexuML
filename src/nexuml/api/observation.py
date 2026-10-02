"""Record actual Lightning callbacks without replacing the session lifecycle."""

import json
import math
from pathlib import Path

from lightning.pytorch.callbacks import Callback


def scalar_values(values: dict) -> dict:
    """Keep only observable finite numeric scalars.

    Returns:
        Plain scalar metrics, not tensor payloads or invented measurements.
    """
    result = {}
    for key, value in values.items():
        if hasattr(value, "numel"):
            if value.numel() != 1:
                continue
            value = value.detach().cpu().item()
        if isinstance(value, (int, float)) and math.isfinite(value):
            result[str(key)] = value
    return result


class Observation(Callback):
    """Small callback appended to the existing Trainer's callback list."""

    def __init__(self, path: Path):
        self.path = path

    def record(self, kind: str, trainer, **payload):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "kind": kind,
                        "payload": {
                            "epoch": trainer.current_epoch,
                            "step": trainer.global_step,
                            **payload,
                        },
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def on_train_epoch_end(self, trainer, pl_module):
        self.record("metrics", trainer, values=scalar_values(trainer.callback_metrics))

    def on_validation_epoch_end(self, trainer, pl_module):
        self.record("metrics", trainer, values=scalar_values(trainer.callback_metrics))

    def on_test_epoch_end(self, trainer, pl_module):
        self.record("metrics", trainer, values=scalar_values(trainer.callback_metrics))

    def on_fit_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="fit")

    def on_fit_end(self, trainer, pl_module):
        self.record("progress", trainer, phase="fit complete")

    def on_validation_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="validate")

    def on_predict_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="predict/post-training fit")

    def on_test_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="test")
