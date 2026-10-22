"""Training-free regression tests for post-training fitted layers."""

import pytest
import torch
from tensordict import TensorDict

from nexuml.core.base_layer import LightningMode
from nexuml.core.post_train_layer import PostTrainFitLayer, PostTrainLayerNotFittedError


class _MeanScore(PostTrainFitLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total = 0.0
        self.count = 0
        self.center = 0.0

    def collect_batch(self, x, y):
        values = x[next(iter(self.keys_in))]
        self.total += values.sum().item()
        self.count += values.numel()

    def finalize_fit(self):
        self.center = self.total / self.count

    def _transform_forward(self, x, y):
        x[self.keys_out[0]] = (x[next(iter(self.keys_in))] - self.center).abs().mean(dim=-1)
        return x, y

    def _get_fit_state(self):
        return {"center": self.center}

    def _set_fit_state(self, state):
        self.center = state["center"]


def test_post_train_fit_lifecycle_and_checkpoint():
    kwargs = dict(input_sizes={"features": (1,)}, keys_in=["features"], keys_out=["score"])
    fitted = _MeanScore(**kwargs)
    fitted.on_test_start()
    with pytest.raises(PostTrainLayerNotFittedError):
        fitted(TensorDict({"features": torch.tensor([[4.0]])}, batch_size=[1]))

    fitted.on_predict_start()
    fitted._armed = True  # Emulate the native orchestrator's fit pass.
    collection = TensorDict({"features": torch.tensor([[1.0], [3.0]])}, batch_size=[2])
    assert "score" not in fitted(collection)[0]
    fitted.on_predict_end()
    assert fitted._fitted and not fitted._armed

    fitted.on_test_start()
    transformed, _ = fitted(TensorDict({"features": torch.tensor([[4.0]])}, batch_size=[1]))
    torch.testing.assert_close(transformed["score"], torch.tensor([2.0]))
    assert transformed["features"].item() == 4.0
    assert fitted.lightning_mode == LightningMode.TESTING

    checkpoint = {}
    fitted.on_save_checkpoint(checkpoint)
    restored = _MeanScore(**kwargs)
    restored.on_load_checkpoint(checkpoint)
    assert restored._fitted and restored.center == 2.0
    restored.on_test_start()
    transformed, _ = restored(TensorDict({"features": torch.tensor([[4.0]])}, batch_size=[1]))
    torch.testing.assert_close(transformed["score"], torch.tensor([2.0]))
