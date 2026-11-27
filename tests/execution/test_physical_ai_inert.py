"""The complete driver dry-run reports inputs without launching any workload."""

import json
from pathlib import Path
import runpy
import sys
from types import ModuleType

import pytest


@pytest.mark.parametrize("data_root", [None, "/worker/approved-location-still-unqualified"])
def test_main_dry_run_reports_missing_inputs_without_runtime_work(
    tmp_path, monkeypatch, capsys, data_root
):
    import nexuml.core.compiler as compiler
    import nexuml.training.lightning as lightning
    import torch

    script = Path(__file__).resolve().parents[2] / "scripts/physical_ai_acceptance.py"
    main = runpy.run_path(str(script))["main"]
    ca = tmp_path / "ca.pem"
    ca.write_text("inert test: not used for a connection")
    output = tmp_path / "no-submission"

    def forbidden(*args, **kwargs):
        pytest.fail("dry-run constructed a runtime, submitted, installed or initialized CUDA")

    sdk = ModuleType("ray.job_submission")
    monkeypatch.setattr(sdk, "JobSubmissionClient", forbidden, raising=False)
    monkeypatch.setitem(sys.modules, "ray.job_submission", sdk)
    monkeypatch.setattr(compiler, "compile", forbidden)
    monkeypatch.setattr(lightning, "NexuSession", forbidden)
    monkeypatch.setattr(torch.cuda, "init", forbidden)
    monkeypatch.setattr(main.__globals__["subprocess"], "run", forbidden)
    monkeypatch.setattr(
        main.__globals__["subprocess"],
        "check_output",
        lambda args, **kwargs: "a" * 40 if args == ["git", "rev-parse", "HEAD"] else "",
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    argv = [str(script), "--case", "preflight", "--ca", str(ca), "--output", str(output)]
    if data_root is not None:
        argv.extend(["--data-root", data_root])
    monkeypatch.setattr(sys, "argv", argv)
    main()
    record = json.loads(capsys.readouterr().out)
    inputs = record["deployment_inputs"]
    assert inputs["address"] == "https://ray.ika.rub.de"
    assert inputs["ca_path"] == str(ca)
    assert inputs["durable_root"] == "/mnt/data/nexuml-qualification"
    assert record["pool"] == "pool_gpu_workers" and record["concurrency"] == 1
    assert inputs["gpu_reservation_per_job"] == 1
    assert inputs["initial_qualification_concurrency"] == 1
    assert inputs["maximum_in_flight"] == 4 and "UNVERIFIED" in inputs["parallel_capacity"]
    prerequisites = inputs["worker_prerequisites"]
    assert set(prerequisites) == {
        "native_assets_licences",
        "graphics",
        "display_private_route",
        "browser_private_route",
        "ros_world_controllers_reset_clock",
        "private_data",
    }
    assert prerequisites["private_data"].startswith(
        "MISSING" if data_root is None else "SUPPLIED_UNVERIFIED"
    )
    assert all(
        value.startswith("UNVERIFIED")
        for name, value in prerequisites.items()
        if name != "private_data"
    )
    assert "local application/GPU fallback" in inputs["boundaries"]
    assert not output.exists()
    assert record["attempts"][0]["options"]["budgets"]["disk_floor_bytes"] == 4 * 1024**3


@pytest.mark.parametrize(
    "address",
    [
        "https://name:secret@ray.example",
        "https://ray.example?token=secret",
        "https://ray.example#secret",
    ],
)
def test_dry_run_does_not_disclose_credentials_from_endpoint(
    address, tmp_path, monkeypatch, capsys
):
    script = Path(__file__).resolve().parents[2] / "scripts/physical_ai_acceptance.py"
    main = runpy.run_path(str(script))["main"]
    ca = tmp_path / "ca.pem"
    ca.touch()
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(script),
            "--case",
            "preflight",
            "--ca",
            str(ca),
            "--address",
            address,
            "--output",
            str(tmp_path / "output"),
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "must not contain credentials" in captured.err
    assert "secret" not in captured.out + captured.err
