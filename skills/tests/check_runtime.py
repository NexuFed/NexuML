"""Training-free checks against installed NexuML: python skills/tests/check_runtime.py."""

import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch
from tensordict import TensorDict

from nexuml.core.base_layer import LightningMode
from nexuml.core.components import LayerBuildContext
from nexuml.core.post_train_layer import PostTrainFitLayer, PostTrainLayerNotFittedError
from nexuml.core.types import ScenarioSpec, TrainingSpec
from nexuml.tuning.optuna_tuner import _resolve_search_space, build_objective


class _MeanScore(PostTrainFitLayer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.total = 0.0
        self.count = 0
        self.center = 0.0

    def collect_batch(self, x, y):
        values = x[self.keys_in[0]]
        self.total += values.sum().item()
        self.count += values.numel()

    def finalize_fit(self):
        self.center = self.total / self.count

    def _transform_forward(self, x, y):
        x[self.keys_out[0]] = (x[self.keys_in[0]] - self.center).abs().mean(dim=-1)
        return x, y

    def _get_fit_state(self):
        return {"center": self.center}

    def _set_fit_state(self, state):
        self.center = state["center"]


class _Trial:
    number = 0

    def __init__(self):
        self.params = {}

    def suggest_categorical(self, name, choices):
        self.params[name] = choices[0]
        return choices[0]


def main():
    """Check documented routing, post-fit lifecycle and current tuning edge cases.

    Raises:
        AssertionError: If the installed behavior differs from the guidance.
    """
    text = (Path(__file__).resolve().parents[1] / "nexuml-library/SKILL.md").read_text()
    snippet = re.search(r"```python\n(.*?)\n```", text, re.S).group(1)
    namespace = {}
    exec(snippet, namespace)
    definition = namespace["dropout"]
    assert not set(definition.keys_in) & set(definition.keys_out)
    context = LayerBuildContext(
        input_sizes={"features": (4,)}, keys_in=definition.keys_in, keys_out=definition.keys_out
    )
    runtime = definition.component.build(context)
    x = TensorDict({"features": torch.ones(2, 4)}, batch_size=[2])
    original = x["features"].clone()
    output, _ = runtime(x, None)
    torch.testing.assert_close(output["features"], original)
    assert output[definition.keys_out[0]].shape == original.shape

    kwargs = dict(input_sizes={"features": (1,)}, keys_in=["features"], keys_out=["score"])
    fitted = _MeanScore(**kwargs)
    fitted.on_test_start()
    try:
        fitted(TensorDict({"features": torch.tensor([[4.0]])}, batch_size=[1]))
    except PostTrainLayerNotFittedError:
        pass
    else:
        raise AssertionError("Unfitted test transformation was accepted.")
    fitted.on_predict_start()
    fitted._armed = True  # emulate the native orchestrator, not user-run fitting code
    collection = TensorDict({"features": torch.tensor([[1.0], [3.0]])}, batch_size=[2])
    assert "score" not in fitted(collection)[0]
    fitted.on_predict_end()
    fitted.on_test_start()
    transformed, _ = fitted(TensorDict({"features": torch.tensor([[4.0]])}, batch_size=[1]))
    torch.testing.assert_close(transformed["score"], torch.tensor([2.0]))
    assert transformed["features"].item() == 4.0
    checkpoint = {}
    fitted.on_save_checkpoint(checkpoint)
    restored = _MeanScore(**kwargs)
    restored.on_load_checkpoint(checkpoint)
    assert restored._fitted and restored.center == 2.0
    assert fitted.lightning_mode == LightningMode.TESTING

    space = {"training.batch_size": {"type": "categorical", "choices": [16, 32]}}
    assert _resolve_search_space(_Trial(), space) == ({}, {"training": {"batch_size": 16}})
    base = ScenarioSpec(name="contract", training=TrainingSpec(max_epochs=3, batch_size=64))
    observed = []

    def train_stub(scenario, **kwargs):
        observed.append(scenario)
        return SimpleNamespace(
            trainer=SimpleNamespace(logged_metrics={"val/loss": 0.5}), eval_algorithm_results={}
        )

    def factory(training):
        return ScenarioSpec(
            name="rebuilt", training=TrainingSpec(batch_size=training["batch_size"])
        )

    # Only test doubles: no Optuna study, actual trial or training job is launched.
    with patch.dict(sys.modules, {"optuna": SimpleNamespace(Trial=object)}):
        with patch("nexuml.training.lightning.train", train_stub):
            build_objective(base, space)(_Trial())
            build_objective(base, space, build_factory=factory)(_Trial())
    assert observed[0].training.batch_size == 64  # sampled categorical value ignored without build
    assert observed[1].training.batch_size == 16
    assert observed[1].training.max_epochs == 10  # initial epoch override lost on factory rebuild
    help_text = subprocess.check_output(
        [sys.executable, "-m", "nexuml.cli.main", "tune", "--help"], text=True
    )
    assert "--max-epochs" not in help_text and "--override" in help_text
    print("PASS: distinct keys, fitted scoring/state, nested tuning params and CLI/factory limits")


if __name__ == "__main__":
    main()
