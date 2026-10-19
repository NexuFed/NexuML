"""CPU proofs that Kit's accepted fast exit cannot preempt job finalization."""

from pathlib import Path
import asyncio
import signal
import subprocess
import sys
from types import SimpleNamespace

import pytest

from nexuml_library.environments import _isaaclab_runtime as runtime


def test_process_close_follows_job_finalization_and_resets_scope(monkeypatch):
    events: list[object] = []
    launcher = SimpleNamespace(__exit__=lambda *args: events.append(("launcher.exit", args)))
    with runtime.isaaclab_process():
        monkeypatch.setattr(runtime, "_process_launcher", launcher)
        try:
            events.append("train.export")
        finally:
            events.append("logger.finalize")
        assert events == ["train.export", "logger.finalize"]
    assert events[-1] == ("launcher.exit", (None, None, None))
    assert not runtime._process_active and runtime._process_launcher is None


@pytest.mark.parametrize(
    "primary,code,message",
    [
        (RuntimeError("training failed"), 1, "training failed"),
        (KeyboardInterrupt(), 130, "KeyboardInterrupt"),
        (SystemExit(7), 7, ""),
        (SystemExit(0), 0, ""),
        (SystemExit(None), 0, ""),
        (SystemExit("bad job"), 1, "bad job"),
    ],
)
def test_process_shutdown_preserves_failures(monkeypatch, capsys, primary, code, message):
    events = []
    launcher = SimpleNamespace(__exit__=lambda *args: events.append(args))
    with pytest.raises(type(primary)) as caught, runtime.isaaclab_process():
        monkeypatch.setattr(runtime, "_process_launcher", launcher)
        raise primary
    assert caught.value is primary
    assert len(events) == 1 and events[0][:2] == (type(primary), primary)
    assert message in capsys.readouterr().err
    assert not runtime._process_active and runtime._process_launcher is None


def test_process_cleanup_failure_preserves_primary(monkeypatch):
    def fail_close(error_type, error, tb):
        assert error is primary and error_type is ValueError
        raise RuntimeError("Kit close failed")

    primary = ValueError("export failed")
    with pytest.raises(ValueError) as caught, runtime.isaaclab_process():
        monkeypatch.setattr(runtime, "_process_launcher", SimpleNamespace(__exit__=fail_close))
        raise primary
    assert caught.value is primary
    assert "Kit close failed" in "\n".join(primary.__notes__)


@pytest.mark.parametrize("primary", [asyncio.CancelledError(), BaseException("cancelled")])
def test_base_exception_cancellation_gets_nonzero_native_close_status(monkeypatch, primary):
    exits = []
    launcher = SimpleNamespace(__exit__=lambda kind, error, tb: exits.append((kind, error)))
    with pytest.raises(type(primary)) as caught, runtime.isaaclab_process():
        monkeypatch.setattr(runtime, "_process_launcher", launcher)
        raise primary
    assert caught.value is primary
    assert len(exits) == 1 and exits[0][0] is SystemExit and exits[0][1].code == 1


def test_no_app_is_allocated_for_a_job_that_never_builds_a_runtime():
    previous = signal.getsignal(signal.SIGTERM)
    with runtime.isaaclab_process():
        assert runtime._process_active and runtime._process_launcher is None
        assert signal.getsignal(signal.SIGTERM) is runtime._terminate_job
    assert signal.getsignal(signal.SIGTERM) is previous


def test_nested_process_scope_does_not_close_the_outer_app():
    with runtime.isaaclab_process():
        with pytest.raises(RuntimeError, match="cannot be nested"):
            with runtime.isaaclab_process():
                pytest.fail("nested job executed")
        assert runtime._process_active


def test_native_allocation_requires_process_boundary_before_cuda_probe():
    with pytest.raises(RuntimeError, match="scripts/isaaclab/run.py"):
        runtime._create_environment(None, None, None)


def test_sigterm_unwinds_environment_and_logger_before_app_close(tmp_path):
    script = tmp_path / "sigterm.py"
    script.write_text(
        "from types import SimpleNamespace\nimport sys\n"
        "from nexuml_library.environments import _isaaclab_runtime as runtime\n"
        "with runtime.isaaclab_process():\n"
        "    runtime._process_launcher = SimpleNamespace(\n"
        "        __exit__=lambda kind, error, tb: print('APP_CLOSE', error.code, flush=True))\n"
        "    try:\n"
        "        print('READY', flush=True)\n"
        "        sys.stdin.readline()\n"
        "    finally:\n"
        "        print('ENV_CLOSE', flush=True)\n"
        "        print('LOGGER_FAILED', flush=True)\n"
    )
    process = subprocess.Popen(
        [sys.executable, str(script)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "READY"
        process.send_signal(signal.SIGTERM)
        stdout, stderr = process.communicate(timeout=15)
        assert process.returncode == 143, stderr
        assert stdout.splitlines() == ["ENV_CLOSE", "LOGGER_FAILED", "APP_CLOSE 143"]
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()


@pytest.mark.parametrize("module", [False, True])
@pytest.mark.parametrize("shell_entry", [False, True])
def test_python_entrypoint_preserves_arguments_imports_and_exit_status(
    tmp_path, module, shell_entry, monkeypatch
):
    (tmp_path / "sibling.py").write_text("value = 17\n")
    script = tmp_path / "job.py"
    script.write_text(
        "import sys\nfrom sibling import value\n"
        "from nexuml_library.environments._isaaclab_runtime import _process_active\n"
        "assert _process_active and value == 17\n"
        "assert sys.argv[1:] == ['argument']\n"
        "print('JOB_FINALIZED', flush=True)\nraise SystemExit(7)\n"
    )
    entry = Path(__file__).resolve().parents[2] / "scripts/isaaclab/run.py"
    monkeypatch.setenv("NEXUML_ISAAC_PYTHON", sys.executable)
    command = (
        ["bash", str(entry.with_suffix(".sh"))] if shell_entry else [sys.executable, str(entry)]
    )
    target = ["-m", "job"] if module else [str(script)]
    result = subprocess.run(
        [*command, *target, "argument"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 7, result.stdout + result.stderr
    assert "JOB_FINALIZED" in result.stdout


def test_shell_entrypoint_does_not_install_a_missing_interpreter(tmp_path, monkeypatch):
    monkeypatch.setenv("NEXUML_ISAAC_PYTHON", str(tmp_path / "missing-python"))
    entry = Path(__file__).resolve().parents[2] / "scripts/isaaclab/run.sh"
    result = subprocess.run(["bash", str(entry)], capture_output=True, text=True, timeout=10)
    assert result.returncode == 1 and "no installer is run" in result.stderr
