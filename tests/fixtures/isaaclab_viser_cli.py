"""Prepare inert, bounded public CLI jobs; never start training or native apps here."""

import json
import os
from pathlib import Path
import sys

from isaaclab_viser_cli_hooks import MODES, configure_probe, observer_callback


def prepare(mode: str, output: Path, port: int) -> dict:
    if mode not in MODES:
        raise ValueError(f"Unsupported probe mode: {mode}")
    if not 1 <= port <= 64535:
        raise ValueError("Probe port must be in 1..64535")

    from nexuml.core.config import ResolvedConfig
    from nexuml.core.types import (
        CallbackSpec,
        ExportSpec,
        LoggingSpec,
        PolicyEvaluationSpec,
        TensorBoardSpec,
    )
    from nexuml_library.environments.isaaclab_config import ViserSpec, isaac_task_config
    from nexuml_library.scenarios.reinforcement.isaaclab_sensors import (
        isaac_mixed_cartpole_ppo,
    )

    output = output.resolve()
    replay = mode.startswith("replay")
    scenario = isaac_mixed_cartpole_ppo(total_frames=256, frames_per_batch=64, num_envs=2)
    scenario.training.algorithm = scenario.training.algorithm.model_copy(
        update={"update_epochs": 1, "minibatch_size": 64}
    )
    definition = scenario.interaction.environment
    base_config = definition.task_config.model_dump()
    definition = definition.model_copy(
        update={"task_config": isaac_task_config(configure_probe, base_config=base_config)}
    )
    if mode.endswith("-viewer"):
        definition = definition.model_copy(
            update={
                "viewer": ViserSpec(
                    purposes=("eval",) if replay else ("train",),
                    port=port,
                    live_plots_update_interval=1,
                    eye=(4.0, -4.0, 4.0),
                    lookat=(0.0, 0.0, 2.0),
                )
            }
        )
    scenario.interaction.environment = definition
    scenario.logging = LoggingSpec(
        tensorboard=TensorBoardSpec(log_dir=str(output / "tensorboard")),
        diagram=None,
        temp_artifact_dir=str(output / "artifacts"),
    )
    if replay:
        # Browser/nonlearning comparison, not the old manual 128-step replay protocol.
        scenario.policy_evaluation = PolicyEvaluationSpec(
            deterministic=True, seed=11, episodes=32, max_steps_per_episode=4
        )
    else:
        scenario.callbacks = [CallbackSpec.from_factory(observer_callback)]
        scenario.exports = [ExportSpec(kind="train_package", output=str(output / "policy"))]

    output.mkdir(parents=True, exist_ok=False)
    config = output / "config.yaml"
    ResolvedConfig.from_scenario(scenario).save(config)
    argv = [sys.executable, "-m", "nexuml.cli.main"]
    if replay:
        argv += [
            "evaluate",
            str(output.parent / "train-viewer" / "policy"),
            "--config",
            str(config),
            "--device",
            "cuda:0",
        ]
    else:
        argv += ["train", "--config", str(config)]
    fixture_dir = str(Path(__file__).resolve().parent)
    return {
        "argv": argv,
        "env": {
            "NEXUML_SPECTATOR_PROBE_OUTPUT": str(output),
            "NEXUML_SPECTATOR_PROBE_MODE": mode,
            "NEXUML_LOGS_ROOT": str(output / "native-logs"),
            "PYTHONPATH": os.pathsep.join(
                [fixture_dir, *filter(None, [os.environ.get("PYTHONPATH")])]
            ),
            "PYTHONDONTWRITEBYTECODE": "1",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
        },
        "config": str(config),
        "port": port,
        "mode": mode,
        "instrumentation": "test-owned instance observer; public CLI owns execution",
        "comparison": "bounded short-horizon browser/nonlearning" if replay else "bounded PPO",
    }


if __name__ == "__main__":
    if len(sys.argv) != 5 or sys.argv[1] != "prepare":
        raise SystemExit("usage: isaaclab_viser_cli.py prepare MODE OUTPUT PORT")
    print(json.dumps(prepare(sys.argv[2], Path(sys.argv[3]), int(sys.argv[4]))))
