"""Documented example smoke: python skills/tests/check_runtime.py."""

import re
from pathlib import Path

import torch
from tensordict import TensorDict

from nexuml.core.components import LayerBuildContext


def main():
    """Check the documented library example's routing and runtime behavior."""
    text = (Path(__file__).resolve().parents[1] / "nexuml-library/SKILL.md").read_text()
    match = re.search(r"```python\n(.*?)\n```", text, re.S)
    assert match is not None, "Library skill is missing its Python example."
    snippet = match.group(1)
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

    print("PASS: documented library example, distinct keys and preserved input")


if __name__ == "__main__":
    main()
