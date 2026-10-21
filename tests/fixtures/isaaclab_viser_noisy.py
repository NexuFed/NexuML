"""Importable native noise configuration used only by the spectator RNG oracle."""

from typing import Any


def configure_noisy(cfg: Any) -> None:
    from isaaclab.utils.noise import UniformNoiseCfg

    cfg.observations.policy.enable_corruption = True
    cfg.observations.policy.joint_pos_rel.noise = UniformNoiseCfg(n_min=-0.1, n_max=0.1)


def fail_observation(env: Any, *, receipt_path: str) -> Any:
    import json
    from pathlib import Path

    (viewer,) = env.sim.visualizers
    server = viewer._viewer._server
    assert server.get_host() == "127.0.0.1"
    Path(receipt_path).write_text(
        json.dumps({"host": server.get_host(), "port": server.get_port()})
    )
    raise RuntimeError("injected native observation construction failure")


def configure_failure(cfg: Any, *, receipt_path: str) -> None:
    cfg.observations.policy.joint_pos_rel.func = fail_observation
    cfg.observations.policy.joint_pos_rel.params = {"receipt_path": receipt_path}
