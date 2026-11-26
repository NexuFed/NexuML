"""Reference client for the same actor on native PhysX and CPU MuJoCo."""

import hashlib
import json
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.discovery import environment
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import (
    InteractionContract,
    PhysicalSystemContract,
    PhysicalTensorContract,
    PolicyEvaluationSpec,
    TensorFieldContract,
)
from nexuml.evaluation.policy import evaluate_policy
from nexuml.training import local_fit, local_fit_manifest
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.environments.isaaclab import IsaacLabEnvironment, isaac_task_config
from nexuml_library.scenarios.reinforcement.isaaclab import isaac_cartpole_ppo


PERIOD = 1 / 60
HORIZON = 120
JOINTS = ("slider_to_cart", "cart_to_pole")
ISAAC_URL = (
    "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/6.2/"
    "Isaac/IsaacLab/Robots/Classic/Cartpole/cartpole.usd"
)
ISAAC_SHA = "c98ce5dbb174876998052d486036fb79e07f52320851526a4ff61c60ed2db043"
MUJOCO_REFERENCE_SHA = "80910a9af85cd47072be82d6f92c5e6a115d0eecc3eb464e58542b285c89fb7f"


def _digest(path):
    with Path(path).open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def _contract(source, digest):
    return InteractionContract(
        action_space="continuous",
        control_period_s=PERIOD,
        physical=PhysicalSystemContract(
            embodiment_id="prismatic-cart-unactuated-hinge-pole",
            calibration_id="upright-zero-positive-tilt-opposes-positive-cart-axis-v1",
            asset_source=source,
            asset_id=f"sha256:{digest}",
        ),
        observations={
            "observation": TensorFieldContract(
                shape=(4,),
                dtype="float32",
                modality="vector",
                physical=PhysicalTensorContract(
                    units=("m", "rad", "m/s", "rad/s"),
                    frame="joint",
                    joints=JOINTS * 2,
                    sensor_id="relative-joint-position-and-velocity",
                ),
            )
        },
        actions={
            "action": TensorFieldContract(
                shape=(1,),
                dtype="float32",
                low=-1,
                high=1,
                rate_hz=60.0,
                physical=PhysicalTensorContract(
                    units=("N",),
                    frame="joint",
                    joints=JOINTS[:1],
                    control_mode="effort",
                    scale=(100.0,),
                    offset=(0.0,),
                ),
            )
        },
    )


def configure_isaac_reference(cfg: Any) -> None:
    """Use the installed native task with a shared horizon and pole angle limit."""
    from isaaclab.envs import mdp
    from isaaclab.managers import SceneEntityCfg, TerminationTermCfg

    assert cfg.scene.robot.spawn.usd_path == ISAAC_URL
    assert cfg.sim.dt * cfg.decimation == PERIOD
    cfg.episode_length_s = HORIZON * PERIOD
    cfg.observations.policy.enable_corruption = False
    cfg.observations.policy.history_length = 0
    for name in ("reset_cart_position", "reset_pole_position"):
        params = getattr(cfg.events, name).params
        params.update(position_range=(-0.01, 0.01), velocity_range=(-0.01, 0.01))
    cfg.terminations.cart_out_of_bounds = TerminationTermCfg(
        func=mdp.joint_pos_out_of_manual_limit,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[JOINTS[1]]),
            "bounds": (-0.2, 0.2),
        },
    )


@environment("test.paired_cartpole.mujoco")
class _MujocoReference(GymnasiumEnvironment):
    def validate_local_fit(self) -> None:
        assert self.env_name == "InvertedPendulum-v5"

    def _make(self, render_mode=None):
        from gymnasium.wrappers import DtypeObservation
        import numpy as np

        assert self.contract is not None and self.contract.physical is not None
        assert self.contract.physical.asset_id == f"sha256:{_digest(self.kwargs['xml_file'])}"
        env = super()._make(render_mode)
        try:
            native = env.unwrapped
            model = native.model
            names = tuple(model.joint(i).name for i in range(model.njnt))
            assert names == JOINTS and native.dt == PERIOD
            np.testing.assert_array_equal(model.jnt_axis, [[1, 0, 0], [0, -1, 0]])
            np.testing.assert_array_equal(model.jnt_qposadr, [0, 1])
            np.testing.assert_array_equal(model.jnt_dofadr, [0, 1])
            np.testing.assert_array_equal(model.actuator_gear[0], [100, 0, 0, 0, 0, 0])
            np.testing.assert_array_equal(model.actuator_ctrlrange, [[-1, 1]])
            return DtypeObservation(env, np.float32)
        except BaseException:
            env.close()
            raise


def _scenario(domain, root):
    scenario = isaac_cartpole_ppo(1024, 128, 8)
    training: Any = scenario.training
    training.algorithm = training.algorithm.model_copy(
        update={"update_epochs": 1, "minibatch_size": 64}
    )
    if domain == "isaac":
        definition = IsaacLabEnvironment(
            task="Isaac-Cartpole",
            physics="isaacsim_physx",
            task_config=isaac_task_config(configure_isaac_reference),
            contract=_contract(ISAAC_URL, ISAAC_SHA),
        )
    elif domain == "mujoco":
        model = root / "cartpole.xml"
        definition = _MujocoReference(
            env_name="InvertedPendulum-v5",
            kwargs={
                "xml_file": str(model),
                "reset_noise_scale": 0.01,
                "max_episode_steps": HORIZON,
            },
            contract=_contract("gymnasium:InvertedPendulum-v5", _digest(model)),
        )
        training.accelerator = "cpu"
        training.collector = training.collector.model_copy(
            update={"env_device": "cpu", "storing_device": "cpu"}
        )
    else:
        raise ValueError(domain)
    assert scenario.interaction is not None
    scenario.interaction.environment = definition
    return scenario


def prepare(root):
    """Pin the upstream model and configure physical units, range and clock."""
    import gymnasium

    source = Path(gymnasium.__file__).parent / "envs/mujoco/assets/inverted_pendulum.xml"
    assert _digest(source) == MUJOCO_REFERENCE_SHA
    tree = ET.parse(source)
    for path, values in (
        ("option", {"timestep": str(PERIOD / 2)}),
        (".//joint[@name='slider']", {"name": JOINTS[0]}),
        (".//joint[@name='hinge']", {"name": JOINTS[1], "axis": "0 -1 0"}),
        ("actuator/motor", {"joint": JOINTS[0], "ctrlrange": "-1 1"}),
    ):
        element = tree.find(path)
        assert element is not None
        element.attrib.update(values)
    tree.write(root / "cartpole.xml")
    torch.manual_seed(17)
    base = CompiledPolicy.from_pipeline(compile(_scenario("isaac", root))).state_dict()
    torch.save(base, root / "base.pt")


def _controls(domain, env, device):
    rows = []
    runtime = env.build(EnvironmentBuildContext(purpose="eval", device=device))
    try:
        if domain == "isaac":
            from isaaclab.utils.assets import read_file

            native = runtime.native
            robot = native.scene["robot"]
            assert hashlib.sha256(read_file(ISAAC_URL).getvalue()).hexdigest() == ISAAC_SHA
            assert tuple(robot.joint_names) == JOINTS and native.step_dt == PERIOD
            assert not robot.data.default_joint_pos.torch.any()
            term = native.action_manager.get_term("joint_effort")
        else:
            native = runtime.envs[0].unwrapped
        for action in (-0.5, 0.5):
            before = runtime.reset(seed=23)["observation"]
            if domain == "isaac":
                state = torch.cat((robot.data.joint_pos.torch[0], robot.data.joint_vel.torch[0]))
            else:
                state = torch.tensor(native._get_obs(), dtype=torch.float32)
            torch.testing.assert_close(before, state)
            after = runtime.step(TensorDict({"action": torch.tensor([action], device=device)}, []))
            force = (
                term.processed_actions.item() if domain == "isaac" else native.data.qfrc_actuator[0]
            )
            assert force == 100 * action
            dv = after.observation["observation"][2:] - before[2:]
            assert (dv * action > 0).all()
            rows.append({"action": action, "force": float(force), "delta_velocity": dv.tolist()})
        return rows
    finally:
        runtime.close()


def run(domain, phase, root):
    """Run one native job and save only the bounded fixture outputs."""
    if phase == "prepare":
        prepare(root)
        return
    scenario = _scenario(domain, root)
    assert scenario.interaction is not None
    env = scenario.interaction.environment
    device = "cuda:0" if domain == "isaac" else "cpu"
    with env.job_scope():
        if phase == "controls":
            receipt = _controls(domain, env, device)
        elif phase in {"fit-1", "fit-2"}:
            index = int(phase[-1])
            base_name = "base" if index == 1 else "combined"
            base = torch.load(root / f"{base_name}.pt", weights_only=True)
            version = f"paired-{index}"
            manifest = local_fit_manifest(scenario, base, global_version=version)
            result = local_fit(
                trusted_recipe=scenario,
                initial_tensors=base,
                model_manifest=manifest,
                global_version=version,
                budget={"frames": 1024},
                seed=(31, 47)[index - 1],
                device=device,
            )
            torch.save(result.updated_tensors, root / f"{domain}-{index}.pt")
            receipt = result.provenance
        else:
            name = phase.removeprefix("eval-")
            base = torch.load(root / f"{name}.pt", weights_only=True)
            policy = CompiledPolicy.from_pipeline(compile(scenario))
            policy.load_state_dict(base, strict=True)
            scenario.policy_evaluation = PolicyEvaluationSpec(
                episodes=4, max_steps_per_episode=HORIZON, seed=211
            )
            reports = []
            evaluate_policy(policy.to(device), scenario, episode_report_sink=reports.append)
            (receipt,) = reports
        (root / f"{domain}-{phase}.json").write_text(json.dumps(receipt, allow_nan=False) + "\n")
