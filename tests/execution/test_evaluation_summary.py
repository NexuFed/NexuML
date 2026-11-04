"""Hermetic presentation checks; no training, policy or native runtime."""

import ast
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest


@pytest.mark.parametrize("frames", [None, 0, 64])
def test_policy_summary_keeps_metric_and_capture_scopes_explicit(frames):
    source = Path(__file__).resolve().parents[2] / "src/nexuml/cli/main.py"
    tree = ast.parse(source.read_text())
    function = next(
        n
        for n in tree.body
        if isinstance(n, ast.FunctionDef) and n.name == "_print_policy_evaluation_summary"
    )
    lines = []
    namespace: dict[str, Any] = {"Any": Any, "console": SimpleNamespace(print=lines.append)}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    namespace[function.name](
        {
            "eval/rl/episodes": 4.0,
            "eval/rl/return_mean": 3.0,
            "eval/rl/task_successes": 1.0,
            "eval/rl/task_success_rate": 0.25,
            "eval/rl/native_truncated": 2.0,
            "eval/rl/evaluation_horizon_limits": 1.0,
            "eval/rl/action_violations": 7.0,
            "eval/rl/robots/left/return_mean": 2.0,
        },
        SimpleNamespace(episodes=4, video=SimpleNamespace(enabled=True, max_frames=8)),
        frames,
    )
    text = "\n".join(lines)
    assert "Episodes: 4/4" in text
    assert f"source physical frames: {frames if frames is not None else 'unavailable'}" in text
    assert "successes: 1.0/4; rate: 0.25" in text
    assert "not training reward" in text
    assert "not necessarily timeouts" in text and "horizon limits: 1.0" in text
    assert "Action violations: 7.0; action rejections: unavailable" in text
    assert "Group/agent robots/left: 2.0" in text
    assert sum("Group/agent" in line for line in lines) == 1
    assert "first episode only" in text and "cap 8 frames" in text
    assert "does not limit the metric episode budget" in text
