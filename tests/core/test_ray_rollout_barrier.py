"""Reject stale or incomplete synchronous Ray rollouts before PPO sees them."""

import pytest
import torch
from tensordict import TensorDict


@pytest.mark.parametrize("corruption", [None, "stale", "missing", "short", "no_time_axis"])
def test_synchronous_ray_batch_validates_worker_versions_and_geometry(corruption):
    pytest.importorskip("ray")
    pytest.importorskip("torchrl")
    from nexuml.reinforcement.ray_collector import _validate_synchronous_rollout

    shape = (2, 2, 2)
    current = torch.full(shape, 3, dtype=torch.long)
    previous_observation = current.clone()
    previous_observation[..., 0] -= 1
    batch = TensorDict(
        {"policy_version": previous_observation, "next": {"policy_version": current}},
        batch_size=shape,
        names=[None, None, "time"],
    )
    if corruption == "stale":
        batch["next", "policy_version"][1, 0, 0] = 2
    elif corruption == "missing":
        del batch["next", "policy_version"]
    elif corruption == "short":
        batch = batch[0:1]
    elif corruption == "no_time_axis":
        batch.names = [None, None, None]
    args = {"version": 3, "frames": 8, "workers": 2}
    if corruption is None:
        _validate_synchronous_rollout(batch, **args)
    else:
        with pytest.raises(RuntimeError, match="stale policy version|incomplete rollout"):
            _validate_synchronous_rollout(batch, **args)
