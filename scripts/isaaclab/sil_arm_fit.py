"""Simulation-only mixed PPO/behavior-cloning acceptance, not a federation client."""

import argparse
from hashlib import sha256
from importlib.metadata import version
from itertools import combinations
import json
from pathlib import Path
from statistics import mean

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from nexuml.core.base_layer import PipelineLayer
from nexuml.core.compiler import compile
from nexuml.core.components import DataSourceDefinition, EnvironmentBuildContext, LayerDefinition
from nexuml.core.discovery import data_source, environment, layer
from nexuml.core.policy import CompiledPolicy
from nexuml.core.types import (
    CollectorSpec,
    DataSpec,
    DatasetSpec,
    FactorySpec,
    InteractionContract,
    InteractionSpec,
    LayerSpec,
    PhysicalSystemContract,
    PhysicalTensorContract,
    PolicyEvaluationSpec,
    PolicySpec,
    ReinforcementLearningSpec,
    RLEvaluationSpec,
    ScenarioSpec,
    TargetSpec,
    TensorFieldContract,
    TrainingSpec,
)
from nexuml.data.dataset import NexuDataset
from nexuml.evaluation.policy import evaluate_policy
from nexuml.training import local_fit, local_fit_manifest
from nexuml.training.local_fit import _state_hash
from nexuml_library.action_adapters.adapters import BoundedNormal, GaussianActionAdapter
from nexuml_library.environments.gymnasium import GymnasiumEnvironment
from nexuml_library.environments.isaaclab import IsaacBindings, IsaacLabEnvironment
from nexuml_library.environments.ros2.joint_reach import Ros2JointReachEnvironment
from nexuml_library.reinforcement.algorithms.ppo import PPO
from nexuml_library.reinforcement.policies import continuous_policy_mlp

from ros2_sil import _HORIZON, _JOINT_FIELDS, _JOINTS, _PERIOD, _TARGET, _TOLERANCE


HERE = Path(__file__).resolve().parent
FRAMES, STEPS = 512, 8
VERSION = "sil-arm-v1"
CLIENTS = ("isaac-0", "isaac-1", "mujoco-0", "ros2-0")
WEIGHTS = {
    "equal_client": (0.25, 0.25, 0.25, 0.25),
    "equal_lane": (1 / 6, 1 / 6, 1 / 3, 1 / 3),
    "native_only": (1 / 4, 1 / 4, 1 / 2, 0.0),
}
ACTION = PhysicalTensorContract(
    units=("rad",) * 2,
    frame="joint",
    joints=_JOINTS,
    control_mode="absolute_position",
    scale=(1.0, 1.0),
    offset=(0.0, 0.0),
)


def write(path, data):
    """Write a finite JSON receipt."""
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


def _digest(path):
    return sha256(path.read_bytes()).hexdigest()


def _contract(lane, asset):
    return InteractionContract(
        action_space="continuous",
        control_period_s=_PERIOD,
        physical=PhysicalSystemContract(
            embodiment_id="sil-two-hinge-arm",
            calibration_id="zero-aligned-positive-z-v1",
            asset_source=lane,
            asset_id=f"sha256:{asset}",
        ),
        observations={
            key: TensorFieldContract(
                shape=(2,),
                dtype="float32",
                modality="vector",
                physical=PhysicalTensorContract(
                    units=("rad/s" if key == "joint_velocity" else "rad",) * 2,
                    frame="joint",
                    joints=_JOINTS,
                    sensor_id=key,
                ),
            )
            for key in _JOINT_FIELDS
        },
        actions={
            "action": TensorFieldContract(
                shape=(2,),
                dtype="float32",
                low=-1,
                high=1,
                rate_hz=1 / _PERIOD,
                physical=ACTION,
            )
        },
    )


class MujocoArm(gym.Env):
    """Run the matching arm using native MuJoCo position servos."""

    observation_space = gym.spaces.Dict(
        {key: gym.spaces.Box(-np.inf, np.inf, (2,)) for key in _JOINT_FIELDS}
    )
    action_space = gym.spaces.Box(-1.0, 1.0, (2,))

    def __init__(self):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(HERE / "sil_arm.xml"))
        self.data = mujoco.MjData(self.model)
        assert self.model.opt.timestep * 24 == _PERIOD
        for i, name in enumerate(_JOINTS):
            joint, drive = self.model.joint(name), self.model.actuator(name)
            assert joint.id == i == drive.id and drive.trnid[0] == joint.id
            np.testing.assert_array_equal(joint.axis, [0, 0, 1])
            np.testing.assert_array_equal(joint.range, [-1, 1])
            np.testing.assert_array_equal(drive.ctrlrange, [-1, 1])

    def _observe(self):
        return dict(
            joint_position=self.data.qpos.astype(np.float32),
            joint_velocity=self.data.qvel.astype(np.float32),
            target_position=np.array(_TARGET, dtype=np.float32),
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.mj.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.np_random.uniform(-0.04, 0.04, size=2)
        self.data.ctrl[:] = self.data.qpos
        self.steps = 0
        return self._observe(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        self.mj.mj_step(self.model, self.data, nstep=24)
        self.steps += 1
        obs = self._observe()
        error = float(np.linalg.norm(obs["joint_position"] - obs["target_position"]))
        success = error <= _TOLERANCE
        return obs, -error, success, self.steps == _HORIZON and not success, {}


gym.register("NexuML-SIL-MuJoCo-v0", entry_point=MujocoArm)


@environment("SILReferenceMuJoCo")
class MujocoDefinition(GymnasiumEnvironment):
    """Allow local fit only on the registered reference task."""

    def validate_local_fit(self):
        if self.env_name != "NexuML-SIL-MuJoCo-v0" or self.kwargs:
            raise ValueError("The reference admits only the pinned MuJoCo arm")


def _native_recipe(lane):
    if lane == "isaac":
        env = IsaacLabEnvironment(
            task="sil_arm_task:NexuML-SIL-Arm-v0",
            physics=None,
            contract=_contract(lane, _digest(HERE / "ros2_sil.py")),
            bindings=IsaacBindings(
                observations={key: ("policy", key) for key in _JOINT_FIELDS},
                direct_controls=FactorySpec(factory="sil_arm_task:controls"),
            ),
        )
    else:
        env = MujocoDefinition(
            env_name="NexuML-SIL-MuJoCo-v0",
            contract=_contract(lane, _digest(HERE / "sil_arm.xml")),
        )
    device = "cuda:0" if lane == "isaac" else "cpu"
    return ScenarioSpec(
        name="sil-arm",
        pipeline=continuous_policy_mlp(_JOINT_FIELDS, action_dim=2, hidden_dims=(16,)),
        interaction=InteractionSpec(environment=env),
        policy=PolicySpec(action_adapter=GaussianActionAdapter()),
        training=ReinforcementLearningSpec(
            total_frames=FRAMES,
            frames_per_batch=128,
            devices=1,
            accelerator="gpu" if lane == "isaac" else "cpu",
            algorithm=PPO(update_epochs=1, minibatch_size=64),
            collector=CollectorSpec(
                num_envs_per_collector=8, env_device=device, storing_device=device
            ),
            evaluation=RLEvaluationSpec(episodes=0),
        ),
    )


def _policy_for(recipe):
    return CompiledPolicy.from_pipeline(compile(recipe))


def _read_record(path, lane):
    data = json.loads(path.read_text())
    if lane == "ros2":
        known = (
            Ros2JointReachEnvironment(control_period_s=_PERIOD).describe().model_dump(mode="json")
        )
        assert data["provider"] == "isaac_sim_ros2_bridge" and data["contract"] == known
        assert data["position_unit"] == "rad" and data["velocity_unit"] == "rad/s"
        assert data["control_mode"] == "absolute_joint_position"
        assert tuple(data["joint_names"]) == _JOINTS and data["control_period_s"] == _PERIOD
        assert data["watchdog_clock_s"] > data["episodes"][-1]["clock_ns"] / 1e9
        for key in ("clock_ns", "joint_state_stamp_ns"):
            stamps = [row[key] for row in data["episodes"]]
            assert all(a < b for a, b in zip(stamps, stamps[1:]))
    else:
        assert data["provider"] == lane
        assert data["contract"] == _contract(lane, data["asset_sha256"]).model_dump(mode="json")
    assert data["simulated"] is True
    for episode in range(4):
        rows = [row for row in data["episodes"] if row["episode"] == episode]
        assert rows and rows[-1]["terminated"] and not rows[-1]["truncated"]
        assert not any(row["terminated"] or row["truncated"] for row in rows[:-1])
        assert [row["step"] for row in rows] == list(range(len(rows)))
        assert all(tuple(row["observation"]["target_position"]) == _TARGET for row in rows)
    assert {row["episode"] for row in data["episodes"]} == set(range(4))
    return data


class Windows(NexuDataset):
    """Use disjoint episode partitions for training, validation and test."""

    def __init__(self, path, lane, partition):
        import pandas as pd
        from nexuml.core.serialization import lower_model

        data = _read_record(path, lane)
        self.rows = [
            row for row in data["episodes"] if row["episode"] >= 2 or row["episode"] in partition
        ]
        self.info = {
            "provenance": {
                "interaction_contract": lower_model(_contract(lane, data["asset_sha256"]))
            }
        }
        meta = []
        origins = {}
        for row in self.rows:
            ep, stamp = row["episode"], row["clock_ns"]
            origins.setdefault(ep, stamp)
            meta.append(
                dict(
                    episode_index=ep,
                    frame_index=row["step"],
                    timestamp=(stamp - origins[ep]) / 1e9,
                    split="test" if ep == 3 else "val" if ep == 2 else "train",
                )
            )
        super().__init__(meta=pd.DataFrame(meta), label_names=[])

    def __getitem__(self, index):
        row = self.rows[index]
        x = TensorDict({key: torch.tensor(row["observation"][key]) for key in _JOINT_FIELDS}, [])
        return x, TensorDict({"action": torch.tensor(row["action"])}, [])


@data_source("SILReferenceWindows")
class Records(DataSourceDefinition):
    """Open a single approved local recording."""

    root: str
    lane: str
    partition: tuple[int, ...]

    def build(self):
        return Windows(Path(self.root) / f"{self.lane}.json", self.lane, self.partition)


@layer("SILBehaviorCloningLoss")
class BehaviorCloningLoss(LayerDefinition):
    """Train both location and scale against bounded demonstrated actions."""

    def build(self, context):
        return _BehaviorCloningLoss(**context.runtime_kwargs())


class _BehaviorCloningLoss(PipelineLayer):
    def forward(self, x, y=None):
        target = self.get_label(x, y)
        dist = BoundedNormal(x["action_loc"], x["action_scale"], unit_event_ndim=1)
        loss = -dist.log_prob(target) if target is not None else x["action_loc"][:, 0] * 0
        x["loss"] = loss
        return x, y


def _clone_recipe(root, lane, partition, actor):
    source = Records(root=str(root), lane=lane, partition=partition)
    recorded = _read_record(root / f"{lane}.json", lane)
    declared = _contract(lane, recorded["asset_sha256"])
    # Compile a data-only job. Its contract describes the recorded mapping, never live ROS.
    env = MujocoDefinition(env_name="NexuML-SIL-MuJoCo-v0", contract=declared)
    graph = actor.pipeline.model_copy(deep=True)
    graph.stages["loss"] = [
        LayerSpec(
            component=BehaviorCloningLoss(),
            keys_in=["action_loc", "action_scale"],
            keys_out=["loss"],
            label_key="action",
        )
    ]
    return actor.model_copy(
        update=dict(
            interaction=InteractionSpec(environment=env),
            # The supervised SDK checkpoints only the pipeline, not adapter buffers.
            policy=None,
            pipeline=graph,
            data=DataSpec(
                datasets=[DatasetSpec(source=source, split_type="keep")],
                input_shapes={key: [2] for key in _JOINT_FIELDS},
                input_contracts=declared.observations,
                targets=[TargetSpec(type="regression", key="action", num_outputs=2)],
            ),
            training=TrainingSpec(batch_size=4, lr=0.01, loss_keys={"loss": 1}),
        )
    )


def _load(path):
    return torch.load(path, map_location="cpu", weights_only=True)


def record_native(root, lane, env, device):
    """Probe signed commands and record native reach demonstrations."""
    runtime = env.build(EnvironmentBuildContext(purpose="eval", device=device, seed=23))
    rows = []
    try:
        for sign in (-1, 1):
            obs = runtime.reset(seed=23)
            assert obs["joint_position"].abs().max() <= 0.04
            action = torch.tensor([0.25, -0.25], device=device) * sign
            change = (
                runtime.step(TensorDict({"action": action}, [])).observation["joint_position"]
                - obs["joint_position"]
            )
            assert (change * action > 0).all()
        for episode in range(4):
            obs = runtime.reset(seed=23 + episode)
            for step in range(_HORIZON):
                action = obs["target_position"] * min(1, (step + 1) / (3 + episode))
                transition = runtime.step(TensorDict({"action": action}, []))
                rows.append(
                    dict(
                        episode=episode,
                        step=step,
                        clock_ns=step * 200_000_000,
                        observation={key: value.tolist() for key, value in obs.items()},
                        action=action.tolist(),
                        terminated=transition.terminated.item(),
                        truncated=transition.truncated.item(),
                    )
                )
                obs = transition.observation
                if rows[-1]["terminated"] or rows[-1]["truncated"]:
                    break
            assert rows[-1]["terminated"]
        declared = env.describe()
        write(
            root / f"{lane}.json",
            dict(
                provider=lane,
                simulated=True,
                contract=declared.model_dump(mode="json"),
                asset_sha256=declared.physical.asset_id.removeprefix("sha256:"),
                episodes=rows,
            ),
        )
    finally:
        runtime.close()


def native(root, lane, phase, candidate=None, client_index=0):
    """Own native resources for one recording, fit or evaluation process."""
    recipe = _native_recipe(lane)
    env = recipe.interaction.environment
    device = "cuda:0" if lane == "isaac" else "cpu"
    with env.job_scope():
        if phase == "record":
            record_native(root, lane, env, device)
            return
        if phase == "fit":
            base = _load(root / "base.pt")
            manifest = local_fit_manifest(recipe, base, global_version=VERSION)
            assert manifest == json.loads((root / "manifest.json").read_text())
            client = f"{lane}-{client_index}"
            assert client in CLIENTS
            update = local_fit(
                trusted_recipe=recipe,
                initial_tensors=base,
                model_manifest=manifest,
                budget={"frames": FRAMES},
                device=device,
                seed=31 + client_index,
                global_version=VERSION,
            )
            assert update.provenance["minibatch_updates"] == STEPS
            assert update.metrics["optimizer_steps"] == 2 * STEPS and update.episodes > 0
            torch.save(update.updated_tensors, root / f"ppo-{client}.pt")
            write(
                root / f"ppo-{client}.json",
                dict(
                    **update.provenance,
                    seed=31 + client_index,
                    versions={
                        key: version(key)
                        for key in (
                            ("torch", "torchrl", "isaaclab")
                            if lane == "isaac"
                            else ("torch", "torchrl", "mujoco")
                        )
                    },
                    device=torch.cuda.get_device_name(0) if lane == "isaac" else device,
                ),
            )
            return
        recipe.policy_evaluation = PolicyEvaluationSpec(
            episodes=8, seed=211, max_steps_per_episode=_HORIZON
        )
        policy = _policy_for(recipe).to(device)
        policy.load_state_dict(_load(root / "candidates" / f"{candidate}.pt"), strict=True)
        reports = []
        evaluate_policy(policy, recipe, episode_report_sink=reports.append)
        (report,) = reports
        rows = report["episodes"]
        assert len(rows) == 8 and all(
            row["native_terminated"] or row["native_truncated"] for row in rows
        )
        write(
            root / f"{lane}-evaluation-{candidate}.json",
            dict(
                mean_error=mean(-row["return"] / row["steps"] for row in rows),
                success=sum(row["native_terminated"] for row in rows),
                episodes=rows,
            ),
        )


def combine(root):
    """Validate and combine shared deltas, then score recorded test episodes."""
    actor = _native_recipe("mujoco")
    policy = _policy_for(actor)
    base = _load(root / "base.pt")
    manifest = json.loads((root / "manifest.json").read_text())
    assert local_fit_manifest(actor, base, global_version=VERSION) == manifest
    policy.load_state_dict(base, strict=True)
    fields = manifest["manifest"]["tensors"]
    names = tuple(
        sorted(key for key, field in fields.items() if field["role"] == "trainable_parameter")
    )
    assert names and all(key.startswith("pipeline.") for key in names)
    short = tuple(key.removeprefix("pipeline.") for key in names)
    pipeline_base = {key: base[f"pipeline.{key}"] for key in policy.pipeline.state_dict()}
    bcs, fits, tests = {}, {}, {}
    for client in CLIENTS:
        lane, index = client.split("-")
        part = (int(index),) if lane == "isaac" else (0, 1)
        spec = _clone_recipe(root, lane, part, actor)
        offline = local_fit_manifest(spec, pipeline_base, trainable_names=short)
        for key, field in offline["tensors"].items():
            expected = fields[f"pipeline.{key}"]
            assert field == {
                **expected,
                "role": "trainable" if f"pipeline.{key}" in names else "frozen",
            }
        compiled = _policy_for(spec.model_copy(update={"policy": actor.policy}))
        compiled.load_state_dict(base, strict=True)
        assert _state_hash(compiled.state_dict()) == manifest["base"]["tensor_sha256"]
        dataset = spec.data.datasets[0].source.build()
        x, _ = dataset[0]
        torch.testing.assert_close(compiled(x)["action"], policy(x)["action"], rtol=0, atol=0)
        update = local_fit(
            trusted_recipe=spec,
            initial_tensors=pipeline_base,
            model_manifest=offline,
            trainable_names=short,
            local_data=spec.data,
            device="cpu",
            budget={"steps": STEPS},
            seed=31 + int(index),
        )
        assert update.metrics["optimizer_steps"] == STEPS
        bcs[client] = {**base, **{f"pipeline.{k}": v for k, v in update.updated_tensors.items()}}
        fits[client] = dict(
            objective="bounded_gaussian_bc",
            unique_windows=update.train_windows,
            training_episodes=update.episodes,
            **update.metrics,
            provenance=update.provenance,
        )
        tests[lane] = [
            dataset[i] for i, row in enumerate(dataset.meta.itertuples()) if row.split == "test"
        ]

    mixed = dict(bcs)
    ppo_receipts = {}
    for client in CLIENTS[:-1]:
        receipt = json.loads((root / f"ppo-{client}.json").read_text())
        assert receipt["model_manifest"] == manifest and receipt["frames"] == FRAMES
        update = _load(root / f"ppo-{client}.pt")
        assert receipt["updated_tensor_sha256"] == _state_hash(update)
        mixed[client], ppo_receipts[client] = update, receipt
    candidates, conflicts = {"base": base}, {}
    for mode, updates in (("mixed", mixed), ("bc", bcs)):
        vectors = {}
        for client, update in updates.items():
            assert update.keys() == base.keys()
            assert all(torch.equal(update[k], base[k]) for k in base.keys() - set(names))
            vectors[client] = torch.cat([(update[k] - base[k]).flatten() for k in names]).double()
            assert torch.isfinite(vectors[client]).all() and vectors[client].norm() > 0
            candidates[f"{mode}_{client}"] = update
        conflicts[mode] = dict(
            l2={key: vec.norm().item() for key, vec in vectors.items()},
            cosine={
                f"{a}/{b}": torch.nn.functional.cosine_similarity(
                    vectors[a], vectors[b], dim=0
                ).item()
                for a, b in combinations(CLIENTS, 2)
            },
        )
        for kind, weights in WEIGHTS.items():
            candidates[f"{mode}_{kind}"] = {
                key: value
                + sum(
                    weight * (updates[client][key] - value)
                    for client, weight in zip(CLIENTS, weights, strict=True)
                )
                if key in names
                else value
                for key, value in base.items()
            }

    (root / "candidates").mkdir()
    mse = {}
    with torch.inference_mode():
        for name, state in candidates.items():
            path = root / "candidates" / f"{name}.pt"
            torch.save(state, path)
            reloaded = _load(path)
            assert _state_hash(reloaded) == _state_hash(state)
            policy.load_state_dict(reloaded, strict=True)
            errors = {
                lane: mean(
                    (policy(x)["action"] - y["action"]).square().mean().item() for x, y in rows
                )
                for lane, rows in tests.items()
            }
            mse[name] = {**errors, "worst": max(errors.values())}
    write(
        root / "comparison.json",
        dict(
            base=manifest,
            weights=WEIGHTS,
            clients=CLIENTS,
            conflicts=conflicts,
            bc=fits,
            ppo=ppo_receipts,
            recorded_action_mse=mse,
            sources={lane: _digest(root / f"{lane}.json") for lane in tests},
        ),
    )


def main():
    """Run one bounded phase in a fresh process."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("phase", choices=("prepare", "record", "fit", "combine", "evaluate"))
    parser.add_argument("--lane", choices=("isaac", "mujoco"))
    parser.add_argument("--client-index", choices=(0, 1), default=0, type=int)
    parser.add_argument(
        "--candidate",
        choices=(
            "base",
            *(f"{mode}_{name}" for mode in ("bc", "mixed") for name in (*CLIENTS, *WEIGHTS)),
        ),
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    root = args.root.resolve()
    if args.phase == "prepare":
        recipe = _native_recipe("mujoco")
        torch.manual_seed(17)
        base = dict(_policy_for(recipe).state_dict())
        torch.save(base, root / "base.pt")
        write(root / "manifest.json", local_fit_manifest(recipe, base, global_version=VERSION))
    elif args.phase == "combine":
        combine(root)
    else:
        if not args.lane:
            parser.error("--lane is required for a native phase")
        if args.phase == "evaluate" and not args.candidate:
            parser.error("--candidate is required for evaluation")
        native(root, args.lane, args.phase, args.candidate, args.client_index)


if __name__ == "__main__":
    main()
