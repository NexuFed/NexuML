"""Fixed native MARL references; critic state is derived from final actor observations."""

from nexuml.core.discovery import scenario
from nexuml.core.types import (
    AgentGroupContract,
    CollectorSpec,
    GroupPolicySpec,
    InteractionContract,
    InteractionSpec,
    LoggingSpec,
    PipelineSpec,
    PolicySpec,
    RLEvaluationSpec,
    ReinforcementLearningSpec,
    ScenarioSpec,
    TensorFieldContract,
    PhysicalTensorContract,
)
from nexuml_library.action_adapters.adapters import GaussianActionAdapter
from nexuml_library.environments.isaaclab import IsaacLabEnvironment
from nexuml_library.environments.isaaclab_config import IsaacBindings
from nexuml_library.reinforcement.algorithms.multiagent import IPPO, MAPPO
from nexuml_library.reinforcement.policies import continuous_policy_mlp


_HAND_MOTORS = (
    "rh_WRJ2",
    "rh_WRJ1",
    "rh_FFJ4",
    "rh_FFJ3",
    "rh_MFJ4",
    "rh_MFJ3",
    "rh_RFJ4",
    "rh_RFJ3",
    "rh_LFJ5",
    "rh_LFJ4",
    "rh_LFJ3",
    "rh_THJ5",
    "rh_THJ4",
    "rh_THJ3",
    "rh_THJ2",
    "rh_THJ1",
    "rh_FFJ0",
    "rh_MFJ0",
    "rh_RFJ0",
    "rh_LFJ0",
)


def _group(agents, width, action_width, observation, action, physical=None):
    return AgentGroupContract(
        agents=agents,
        action_space="continuous",
        observations={
            "observation": TensorFieldContract(
                shape=(width,), dtype="float32", modality="vector", description=observation
            )
        },
        actions={
            "action": TensorFieldContract(
                shape=(action_width,),
                dtype="float32",
                low=-1,
                high=1,
                rate_hz=60.0,
                description=action,
                physical=physical,
            )
        },
    )


def _native_marl(name, family, centralized, total_frames, frames_per_batch, num_envs):
    if family == "pendulum":
        task = "Isaac-Pendulum-MARL-Direct"
        groups = {
            "cart": _group(
                ("cart",),
                4,
                1,
                "Cart position m/velocity m/s; normalized pole angle rad and pole velocity rad/s.",
                "slider_to_cart effort = 100 * normalized action N.",
                PhysicalTensorContract(
                    units=("N",),
                    frame="joint",
                    joints=("slider_to_cart",),
                    control_mode="effort",
                    scale=(100.0,),
                    offset=(0.0,),
                ),
            ),
            "pendulum": _group(
                ("pendulum",),
                3,
                1,
                "Sum of normalized pole and pendulum angles rad; "
                "pendulum angle rad and velocity rad/s.",
                "pole_to_pendulum effort = 50 * normalized action Nm.",
                PhysicalTensorContract(
                    units=("Nm",),
                    frame="joint",
                    joints=("pole_to_pendulum",),
                    control_mode="effort",
                    scale=(50.0,),
                    offset=(0.0,),
                ),
            ),
        }
        ids = {"cart": ("cart_actor",), "pendulum": ("pendulum_actor",)}
    else:
        task = "Isaac-Shadow-Handover-Direct"
        groups = {
            "hands": _group(
                ("right_hand", "left_hand"),
                157,
                20,
                "Own hand: normalized 24 DOF positions, 0.2-scaled velocities; "
                "ordered five fingertip positions/rotations/velocities, 20 applied actions; "
                "shared 24D object/goal pose/velocity.",
                f"Own hand motor order {_HAND_MOTORS!r}; normalized absolute position targets. "
                "native limit unscale, joint target EMA factor 1, joint/tendon limit clamp. "
                "First 16 coordinates use native asset joint limits in rad; last four tendons "
                "use [0, 3.1415] rad. Asset: ShadowHandMultiPhysics_v0/right_hand/right_hand.usda. "
                "No independent privileged 290D state is used by this recipe.",
            )
        }
        ids = {"hands": ("hand_actor", "hand_actor")}
    contract = InteractionContract(
        groups=groups, action_space="structured", control_period_s=1 / 60, team_reward=True
    )
    definition = IsaacLabEnvironment(
        task=task,
        physics="newton_mjwarp",
        contract=contract,
        bindings=IsaacBindings(
            groups={group: IsaacBindings(observations={"observation": ()}) for group in groups}
        ),
    )
    policies = {
        group: GroupPolicySpec(
            pipeline=continuous_policy_mlp(
                action_dim=spec.actions["action"].shape[0], hidden_dims=(32, 32)
            ),
            action_adapter=GaussianActionAdapter(),
            actor_ids=ids[group],
        )
        for group, spec in groups.items()
    }
    return ScenarioSpec(
        name=name,
        pipeline=PipelineSpec(stages={}),
        interaction=InteractionSpec(environment=definition),
        policy=PolicySpec(groups=policies),
        logging=LoggingSpec(tensorboard=None),
        training=ReinforcementLearningSpec(
            algorithm=(MAPPO if centralized else IPPO)(update_epochs=2, minibatch_size=64),
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            accelerator="gpu",
            devices=1,
            seed=0,
            collector=CollectorSpec(
                backend="direct",
                num_collectors=1,
                num_envs_per_collector=num_envs,
                sync=True,
                env_device="cuda:0",
                storing_device="cuda:0",
            ),
            evaluation=RLEvaluationSpec(episodes=0, at_start=False, every_n_frames=None),
        ),
    )


@scenario("isaac-pendulum-ippo")
def isaac_pendulum_ippo(total_frames=50_000, frames_per_batch=1024, num_envs=32) -> ScenarioSpec:
    """Return heterogeneous local-policy/local-value Pendulum training."""
    return _native_marl(
        "isaac-pendulum-ippo", "pendulum", False, total_frames, frames_per_batch, num_envs
    )


@scenario("isaac-pendulum-mappo")
def isaac_pendulum_mappo(total_frames=50_000, frames_per_batch=1024, num_envs=32) -> ScenarioSpec:
    """Return heterogeneous local policies with a native cross-group critic."""
    return _native_marl(
        "isaac-pendulum-mappo", "pendulum", True, total_frames, frames_per_batch, num_envs
    )


@scenario("isaac-handover-ippo")
def isaac_handover_ippo(total_frames=50_000, frames_per_batch=1024, num_envs=32) -> ScenarioSpec:
    """Return compatible shared hand actors and independent local hand values."""
    return _native_marl(
        "isaac-handover-ippo", "handover", False, total_frames, frames_per_batch, num_envs
    )


@scenario("isaac-handover-mappo")
def isaac_handover_mappo(total_frames=50_000, frames_per_batch=1024, num_envs=32) -> ScenarioSpec:
    """Return shared hand actors with native centralized, distinct hand values."""
    return _native_marl(
        "isaac-handover-mappo", "handover", True, total_frames, frames_per_batch, num_envs
    )
