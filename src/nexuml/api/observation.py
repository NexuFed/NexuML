"""Record actual Lightning callbacks without replacing the session lifecycle."""

import json
import math
import time
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
        self.last_progress = 0.0
        self.last_phase = ""
        self.completed = 0

    def record(self, kind: str, trainer=None, **payload):
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "kind": kind,
                        "payload": {
                            **(
                                {"epoch": trainer.current_epoch, "step": trainer.global_step}
                                if trainer is not None
                                else {}
                            ),
                            **payload,
                        },
                    },
                    allow_nan=False,
                )
                + "\n"
            )

    def metrics(self, trainer, prefixes, **payload):
        values = scalar_values(
            {
                key: value
                for key, value in trainer.callback_metrics.items()
                if key.startswith(prefixes)
                or not key.startswith(("train/", "val/", "test/", "eval/"))
            }
        )
        if values:
            self.record("metrics", trainer, values=values, **payload)

    def on_train_epoch_end(self, trainer, pl_module):
        self.metrics(trainer, "train/", phase="train")

    def on_validation_end(self, trainer, pl_module):
        # Module epoch-end hooks finalize accuracy/F1 before this callback runs.
        if not trainer.sanity_checking:
            self.metrics(trainer, "val/", phase="validate")

    def on_test_end(self, trainer, pl_module):
        self.metrics(trainer, ("test/", "eval/"), phase="test")

    def on_fit_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="fit")

    def progress(self, trainer, phase, batch, total, *, force=False):
        """Publish real counters at most four times per second, plus boundaries.

        Returns:
            Whether a progress sample was published on this tick.
        """
        if isinstance(total, (list, tuple)):
            total = sum(total)
        total = int(total) if isinstance(total, (int, float)) and math.isfinite(total) else None
        now = time.monotonic()
        if force or phase != self.last_phase or batch == total or now - self.last_progress >= 0.25:
            self.last_progress = now
            self.last_phase = phase
            self.record(
                "progress",
                trainer,
                phase=phase,
                batch=batch,
                total=total,
                max_epochs=trainer.max_epochs,
            )
            return True
        return False

    def on_train_epoch_start(self, trainer, pl_module):
        self.progress(trainer, "train", 0, trainer.num_training_batches, force=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        published = self.progress(trainer, "train", batch_idx + 1, trainer.num_training_batches)
        if published or batch_idx == 0:
            self.metrics(trainer, "train/", phase="train", batch=batch_idx + 1)

    def on_validation_epoch_start(self, trainer, pl_module):
        self.completed = 0
        self.progress(
            trainer,
            "sanity check" if trainer.sanity_checking else "validate",
            0,
            trainer.num_sanity_val_batches if trainer.sanity_checking else trainer.num_val_batches,
            force=True,
        )

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        self.completed += 1
        self.progress(
            trainer,
            "sanity check" if trainer.sanity_checking else "validate",
            self.completed,
            trainer.num_sanity_val_batches if trainer.sanity_checking else trainer.num_val_batches,
        )

    def on_test_epoch_start(self, trainer, pl_module):
        self.completed = 0
        self.progress(trainer, "test", 0, trainer.num_test_batches, force=True)

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.completed += 1
        self.progress(trainer, "test", self.completed, trainer.num_test_batches)

    def on_predict_epoch_start(self, trainer, pl_module):
        self.completed = 0
        self.progress(
            trainer, "predict/post-training fit", 0, trainer.num_predict_batches, force=True
        )

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self.completed += 1
        self.progress(
            trainer, "predict/post-training fit", self.completed, trainer.num_predict_batches
        )

    def on_fit_end(self, trainer, pl_module):
        self.record("progress", trainer, phase="fit complete")

    def on_validation_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="validate")

    def on_predict_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="predict/post-training fit")

    def on_test_start(self, trainer, pl_module):
        self.record("progress", trainer, phase="test")
