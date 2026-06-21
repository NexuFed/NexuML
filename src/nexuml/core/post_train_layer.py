"""PostTrainFitLayer — pipeline layer with a post-training accumulate/fit lifecycle."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any

from tensordict import TensorDict

from nexuml.core.base_layer import LightningMode, PipelineLayer


class PostTrainLayerNotFittedError(RuntimeError):
    """Raised when a PostTrainFitLayer is asked to transform before fitting."""


class PostTrainFitLayer(PipelineLayer):
    """Abstract pipeline layer that requires a post-training fit pass.

    Lifecycle (managed by NexuSession._fit_post_train_layers):
      1. Orchestrator arms exactly one unfitted layer per predict pass: ``layer._armed = True``
      2. During that predict pass, ``forward()`` calls ``collect_batch(x, y)`` and returns
         ``(x, y)`` unchanged (no output key written yet).
      3. ``on_predict_end()`` calls ``finalize_fit()`` and sets ``_fitted = True``.
      4. On subsequent calls, ``forward()`` calls ``_transform_forward(x, y)`` (fitted mode).
      5. Other unfitted layers in the same pass are not armed — they pass through silently.

    Fitted state is persisted via Lightning checkpoint hooks so loading a checkpoint
    restores a fully fitted pipeline without re-running fit passes.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._fitted: bool = False
        self._armed: bool = False
        self._fitted_state: dict[str, Any] = {}

    @abstractmethod
    def collect_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        """Accumulate statistics from one train batch during the fit predict pass."""

    @abstractmethod
    def finalize_fit(self) -> None:
        """Finalize fitting after all train batches have been collected."""

    @abstractmethod
    def _transform_forward(
        self, x: TensorDict, y: TensorDict | None
    ) -> tuple[TensorDict, TensorDict | None]:
        """Apply fitted transformation and return (x, y) with output key(s) written."""

    def forward(  # ty: ignore[invalid-method-override]
        self, x: TensorDict, y: TensorDict | None = None
    ) -> tuple[TensorDict, TensorDict | None]:
        if self._fitted:
            return self._transform_forward(x, y)
        if self._armed:
            # This layer is the current fit target — collect, pass through unchanged
            self.collect_batch(x, y)
            return x, y
        if self.lightning_mode in (LightningMode.PREDICTING, LightningMode.NONE):
            # Either a different layer is being fitted in this pass, or we are in
            # compiler/shape-propagation mode — pass through silently
            return x, y
        # In test or training without being fitted: raise
        raise PostTrainLayerNotFittedError(
            f"{self.__class__.__name__} has not been fitted. "
            "Run NexuSession._fit_post_train_layers() before test()."
        )

    def on_predict_end(self) -> None:
        if self._armed:
            self.finalize_fit()
            self._fitted = True
            self._armed = False
        super().on_predict_end()

    # --- Checkpoint serialization ---

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        checkpoint[f"post_train_fitted_{self.__class__.__name__}"] = {
            "fitted": self._fitted,
            "state": self._get_fit_state(),
        }

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        key = f"post_train_fitted_{self.__class__.__name__}"
        saved = checkpoint.get(key) or {}
        if saved.get("fitted"):
            self._set_fit_state(saved.get("state") or {})
            self._fitted = True

    def _get_fit_state(self) -> dict[str, Any]:
        """Return fitted state for serialization. Override in subclasses."""
        return {}

    def _set_fit_state(self, state: dict[str, Any]) -> None:
        """Restore fitted state from checkpoint. Override in subclasses."""
