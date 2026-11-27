"""Software guards for the visual pressure lane, without a native runtime or GPU."""

import pytest
import torch

pytestmark = pytest.mark.requires_optional("torchrl")


def _forbidden(*args, **kwargs):
    pytest.fail("configuration must be rejected before learner or simulator allocation")


@pytest.mark.parametrize(
    "flags,message",
    [
        (["--learner", "cpu"], "explicit visible CUDA index"),
        (["--inference", "cuda"], "explicit visible CUDA index"),
        (["--inference", "cuda:2"], "explicit visible CUDA index"),
        (["--channels", "0"], "positive frame counts and channels"),
        (["--learner", "cuda:1", "--frames", "2048"], "cuda:1 estimated working set"),
        (["--channels", "4096"], "cuda:0 estimated working set"),
    ],
)
def test_pressure_preflight(flags, message, monkeypatch, capsys):
    import visual_rollout_pressure as pressure

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbidden)
    monkeypatch.setattr(
        torch.cuda, "mem_get_info", lambda d: (2**30 if d.index == 0 else 2**26, 2**30)
    )
    monkeypatch.setattr(pressure, "_run", _forbidden)
    with pytest.raises(SystemExit) as error:
        pressure._main(flags)
    assert error.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize("inference", ["cuda:0", "cuda:1"])
def test_pressure_passes_explicit_split_roles(inference, monkeypatch):
    import visual_rollout_pressure as pressure

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbidden)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda d: (2**30, 2**30))

    def select_roles(frames, side, batch, channels, sim, actor, learner):
        assert (sim, actor, learner) == tuple(map(torch.device, ("cuda:0", inference, "cuda:1")))

    monkeypatch.setattr(pressure, "_run", select_roles)
    pressure._main(["--inference", inference, "--learner", "cuda:1"])


def test_rgb_terminal_gae_cpu_reference():
    from visual_rollout_pressure import _check_gae

    _check_gae(torch.device("cpu"), torch.device("cpu"))


@pytest.mark.parametrize("configuration", ["split-learner", "split-inference", "fsdp", "ddp-mixed"])
def test_native_rgb_rejects_unqualified_configuration(configuration, monkeypatch):
    from nexuml.execution.semantics import ExecutionError, ensure_distributed_semantics
    from nexuml.training.reinforcement import NexuRLLightningModule
    from nexuml_library.environments.isaaclab import IsaacLabEnvironment
    from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_rgb_cartpole_ppo

    monkeypatch.setattr(IsaacLabEnvironment, "build", _forbidden)
    monkeypatch.setattr(torch.cuda, "_lazy_init", _forbidden)
    monkeypatch.setattr("nexuml.training.reinforcement.compile", _forbidden)
    scenario = isaac_rgb_cartpole_ppo()
    training = scenario.training
    if configuration == "ddp-mixed":
        training.precision = "16-mixed"
        training.collector.env_device = training.collector.storing_device = "cuda"
        with pytest.raises(ExecutionError, match="DDP with 32-true precision"):
            ensure_distributed_semantics(scenario, world_size=2, accelerator="cuda")
        return
    if configuration == "fsdp":
        training.strategy = "fsdp"
        match = "training.strategy='auto' or 'ddp'"
    else:
        training.devices = "1,"
        if configuration == "split-inference":
            training.collector.policy_device = "cuda:1"
        match = "split placement is limited to single-agent PhysX"
    with pytest.raises(ValueError, match=match):
        NexuRLLightningModule(scenario)
