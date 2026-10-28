"""Native raw-image, final-scene and caller-history oracle; separate from learner proof."""

import shutil
import sys

import torch
from tensordict import TensorDict

from nexuml.core.compiler import compile
from nexuml.core.components import EnvironmentBuildContext
from nexuml.core.policy import CompiledPolicy
from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_mixed_cartpole_ppo


def _snapshot(batch):
    return {key: value.detach().cpu().numpy().copy() for key, value in batch.items()}


def _save_diagnostics(output, owner, definition, samples):
    import json
    from pathlib import Path

    import numpy as np
    from PIL import Image

    families = {"rgb", "depth", "segmentation", "state"}
    assert set(samples) == {
        f"{stage}_{kind}"
        for stage in ("initial", "final", "reset")
        for kind in ("raw", "processed")
    }
    assert all(set(batch) == families for batch in samples.values())
    arrays = {
        f"{stage}__{key}": value for stage, batch in samples.items() for key, value in batch.items()
    }
    assert all(value.dtype.kind in "buif" for value in arrays.values())
    array_bytes = sum(value.nbytes for value in arrays.values())
    assert array_bytes <= 2**20, "Sensor diagnostics exceed the 1 MiB array budget"
    metadata = {
        "scope": "controlled native sensor oracle; not policy behavior/learning/viewer proof",
        "owner": owner,
        "task": definition.task,
        "physics": definition.physics,
        "seed": 0,
        "num_envs": 2,
        "selected_reset_mask": [True, False],
        "source_size": [32, 32],
        "captured_size": [32, 32],
        "stage_labels": {
            "initial": "initial reset scene; repeat-first processed history",
            "final": (
                "ending scene before autoreset: env 0 cart intentionally offscreen, mostly "
                "background is not renderer-pass evidence; env 1 continuing"
            ),
            "reset": "cached selected reset: env 0 fresh scene/history; env 1 continuing history",
        },
        "observation_specs": {
            key: spec.model_dump(mode="json")
            for key, spec in definition.describe().observations.items()
        },
        "array_bytes": array_bytes,
        "arrays": {
            name: {"dtype": str(value.dtype), "shape": list(value.shape), "nbytes": value.nbytes}
            for name, value in arrays.items()
        },
    }
    encoded_metadata = json.dumps(metadata, allow_nan=False, indent=2) + "\n"
    output = Path(output)
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / "sensor-snapshots.npz", **arrays)
    (output / "sensor-metadata.json").write_text(encoded_metadata)
    for stage in ("initial", "final", "reset"):
        Image.fromarray(samples[f"{stage}_raw"]["rgb"][0]).save(output / f"{stage}-rgb-env0.png")


def main(owner, output):
    assert shutil.disk_usage(".").free > 4 * 2**30
    scenario = isaac_mixed_cartpole_ppo(processing_owner=owner)
    assert scenario.interaction is not None
    definition = scenario.interaction.environment
    policy = CompiledPolicy.from_pipeline(compile(scenario)).to("cuda:0")
    with definition.job_scope():
        runtime = definition.build(
            EnvironmentBuildContext(purpose="train", device="cuda:0", num_envs=2, seed=0)
        )
        try:
            initial = runtime.reset(seed=0)
            native = runtime.native
            raw = runtime._state(native.observation_manager.compute(update_history=False))
            processor = runtime._processor
            assert processor is not None
            state = processor.initialize_state((2,), device="cuda:0")
            expected, state = processor.step(raw, state)
            for key in initial.keys():
                torch.testing.assert_close(initial[key], expected[key], rtol=0, atol=0)
            if owner == "package":
                action, actor_state = policy.step(raw, policy.initialize_state((2,)))
                torch.testing.assert_close(
                    action["action"], policy(initial)["action"], rtol=0, atol=0
                )
                for key in state.keys(True, True):
                    torch.testing.assert_close(actor_state[key], state[key], rtol=0, atol=0)
            else:
                assert policy.processor is None
            assert raw["rgb"].shape == (2, 32, 32, 3) and raw["rgb"].dtype == torch.uint8
            assert raw["depth"].dtype == torch.float32
            assert raw["segmentation"].dtype == torch.int32
            assert torch.unique(raw["rgb"]).numel() > 4
            assert (raw["segmentation"] == 2).any()
            object_depth = raw["depth"][raw["segmentation"] == 2]
            assert object_depth.min() > 4 and object_depth.max() < 6
            assert (raw["depth"][raw["segmentation"] == 0] == 0).all()
            print("NATIVE_RGB_METRE_DEPTH_AND_EXACT_CLASS_LABELS_VERIFIED", flush=True)

            samples = {"initial_raw": _snapshot(raw), "initial_processed": _snapshot(initial)}
            robot = native.scene["robot"]
            joints, _ = robot.find_joints(["slider_to_cart"], preserve_order=True)
            ids = torch.tensor([0], dtype=torch.int32, device="cuda:0")
            robot.write_joint_position_to_sim_index(
                position=torch.tensor([[2.5]], device="cuda:0"), joint_ids=joints, env_ids=ids
            )
            robot.write_joint_velocity_to_sim_index(
                velocity=torch.zeros(1, 1, device="cuda:0"), joint_ids=joints, env_ids=ids
            )
            native.episode_length_buf[0] = native.max_episode_length - 1
            transition = runtime.step(
                TensorDict({"action": torch.zeros(2, 1, device="cuda:0")}, [2])
            )
            assert (transition.terminated | transition.truncated).flatten().tolist() == [
                True,
                False,
            ]
            final_raw = runtime._state(native.extras["final_obs"])
            live_raw = runtime._state(native.observation_manager.compute(update_history=False))
            assert (final_raw["segmentation"][0] == 0).all()
            assert (final_raw["depth"][0] == 0).all()
            assert (live_raw["segmentation"][0] == 2).any()
            assert final_raw["state"][0, 0] > 2.4 and live_raw["state"][0, 0].abs() < 1
            assert not torch.equal(final_raw["rgb"][0], live_raw["rgb"][0])
            for key in raw.keys():
                assert torch.equal(final_raw[key][1], live_raw[key][1])
            samples["final_raw"] = _snapshot(final_raw)
            samples["reset_raw"] = _snapshot(live_raw)
            final, final_state = processor.step(final_raw, state)
            for key in final.keys():
                torch.testing.assert_close(transition.observation[key], final[key], rtol=0, atol=0)
            samples["final_processed"] = _snapshot(transition.observation)
            before_ack = runtime._processing_state.clone()
            reset = runtime.reset(mask=torch.tensor([True, False], device="cuda:0"))
            for key in before_ack.keys(True, True):
                assert torch.equal(runtime._processing_state[key], before_ack[key])
            fresh = processor.reset_state(final_state, torch.tensor([True, False], device="cuda:0"))
            restarted, restarted_state = processor.step(live_raw[:1], fresh[:1])
            for key in reset.keys():
                assert torch.equal(reset[key][0], restarted[key][0])
                assert torch.equal(reset[key][1], final[key][1])
            for key in restarted_state.keys(True, True):
                assert torch.equal(runtime._processing_state[key][0], restarted_state[key][0])
                assert torch.equal(runtime._processing_state[key][1], final_state[key][1])
            samples["reset_processed"] = _snapshot(reset)
            print(
                "NATIVE_FINAL_SCENE_HISTORY_AND_CACHED_SELECTED_RESET_VERIFIED", owner, flush=True
            )
            # Kit-backed scope exit can terminate Python before further evidence writes.
            _save_diagnostics(output, owner, definition, samples)
        finally:
            runtime.close()
    assert shutil.disk_usage(".").free > 4 * 2**30


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
