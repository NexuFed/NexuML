# Task 6.2 — live CARLA reference

**Result:** Native tick, RGB/ego synchronization, collision receipt, and owned cleanup
passed on the isolated CARLA 0.9.16 configuration below. After fixing the
client-present export failure, the existing live smoke passed through export and
reload. The live-exported policy also ran in a fresh process without CARLA.
This is one reference configuration, not a provider-wide qualification.

## Binary provenance and invocation

- [CARLA 0.9.16 Docker instructions](https://carla.readthedocs.io/en/0.9.16/build_docker/)
  identify `docker.io/carlasim/carla:0.9.16`. Its Linux/amd64 manifest resolved to
  `sha256:aaf1df22702780ece072069e23d03c4879b002ae028c79744b09c4c7ddbae953`;
  the pinned pull and loaded image succeeded. The [upstream 0.9.16 release](https://github.com/carla-simulator/carla/releases/tag/0.9.16)
  points to source commit `294096eb1c38eabf246e4f3a9cdab704e33a7f4c`.
- [PyPI 0.9.16 metadata](https://pypi.org/pypi/carla/0.9.16/json) lists
  `carla-0.9.16-cp312-cp312-manylinux_2_31_x86_64.whl` from the CARLA team.
  Download verified against the advertised SHA-256
  `c323a7b1f8ad3ac81cfac79f5d96c9cac8afa6d0a768fa2fffa46bb336945c4f`.
  Installed only the wheel in an isolated venv referencing the project's existing
  site-packages, without changing the project venv. Client and server both reported
  `0.9.16`. No source build or Harbor image.
- CPython 3.12.16, PyTorch 2.11.0+cu130, TorchRL 0.14.0, Linux x86-64
  (glibc 2.39), Docker Engine 28.0.4 with NVIDIA runtime, RTX 4080 SUPER
  (driver 610.43.03), UE 4.26.2, `Carla/Maps/Town10HD_Opt`.
- The launcher held `flock /tmp/opencode/nexuml-decouple-native-gpu.lock` through
  teardown. Server ran `bash CarlaUE4.sh -RenderOffScreen -nosound -quality-level=Low`
  with separate `-carla-rpc-port` and `-carla-streaming-port` matching their
  loopback-only Docker TCP mappings (`40163/57729` and `50935/60629`).
  No Traffic Manager, host networking, external service or physical robot.

From the repository root on this host, with the scratch venv and pinned image:

```bash
d=/tmp/opencode/nexuml-carla-O6kEUc
bash "$d/run-live.sh"            # existing pytest smoke + native lifecycle probe
bash "$d/run-live.sh" collision  # native collision probe on another owned server
```

The pre-fix run invoked `NEXUML_RUN_SYSTEM_TESTS=1 NEXUML_CARLA_HOST=127.0.0.1
NEXUML_CARLA_PORT=40163 "$d/venv/bin/python" -m pytest
tests/core/test_carla_environment.py -m requires_system --basetemp "$d/pytest-live"
-o cache_dir="$d/pytest-cache" -q -rs`. It reached the post-training assertions
for a returned policy, nonempty `interaction_results` and positive `global_step`,
then failed at `tests/core/test_carla_environment.py:261` → `src/nexuml/core/export.py:752`:

```text
torch.package.package_exporter.PackagingError:
* Module did not match against any action pattern. Extern, mock, or intern it.
    carla
```

The same error reproduced without a server in
`test_carla_environment.py::test_contract_round_trip_and_export_without_carla`
using the wheel venv. Using the original venv without CARLA, the same test plus
the rest of `-m 'not requires_system'` returned **7 passed, 1 deselected**.

## Native receipts

The separate probe used the unchanged runtime. It logged native camera callbacks
and took an independent settings/actor readback. Selected raw lines:

```text
reset start world=2693 host=183567.293
camera frame=2694 sim_time=13.833850 host=183567.308 size=32x32
reset observation 2694 13.833850388095016 (3, 32, 32) [0.4899999797344208, 0.0, 0.0]
camera frame=2695 sim_time=13.883850 host=183567.319 size=32x32
step observation 2695 13.883850388840074 [0.9800000786781311, 0.0, 0.0]
vehicle control VehicleControl(throttle=0.500000, steer=0.250000, brake=0.000000, hand_brake=False, reverse=False, manual_gear_shift=False, gear=0)
step timeout CARLA camera timed out waiting for frame 2711
injected in-process cancellation
foreign tick 2741
step rejected interference CARLA world frame advanced outside the owned tick
```

Exactly one 50 ms tick advanced each successful reset/step; the native snapshot
contained the ego and the camera frame matched. Stopping the actual camera
stream induced the timeout. Cancellation injected a `BaseException` at queue
read and exercised the caller's `finally` with live actors; it was not a
process kill or `NexuSession` cancellation test. A second client intentionally
advanced the private world once for the interference case; the runtime rejected
the next step without advancing again. CARLA does not enforce exclusive authority
server-side. Each path, and the failing smoke, restored all 11 readable world
settings and left the original 23 actor IDs/types unchanged. The probe verified
the vehicle, camera and collision sensor were absent after idempotent `close()`.

On the separate collision run a probe-owned parked vehicle was placed 6 m ahead:

```text
impact setup frame=373 ego=24 blocker=27 Location(x=109.543678, y=47.293293, z=0.585300)
world=391 sim_time=4.201722022 collision=391 event_time=4.201722022 other=27 terminated=True
runtime destroyed {24, 25, 26}, kept blocker 27
collision cleanup left baseline settings and actors unchanged
```

One positive event arrived in time for its matching transition; this does not
prove per-frame negative collision acknowledgment or a worst-case callback bound.
Both pre-fix containers were removed. Raw launch/pytest/probe receipts and baseline
JSON are in `/tmp/opencode/nexuml-carla-O6kEUc/` (`run-live.log`,
`pytest-live.log`, `adapter.log`, `run-collision.log`, `collision.log`,
`pytest-export-isolated.log`, `before.json`, and `after-*.json`).

## Client-present export fix and retest

`PackageExporter.get_rdeps()` traced all three pickle entries (`model.pipeline`,
`nexuml_export.policy`, `nexuml_export.artifact`) through
`nexuml_library.environments.carla` to `carla` with `PackagingErrorReason.NO_ACTION`.
Torch Package scans function-local `import` statements; its resolver includes
the wheel only when importable. CARLA is an environment-build dependency, not a
policy dependency. `build()` now uses the same runtime import mechanism as ROS
transport (`importlib.import_module`); no exporter policy change or package mock.
The regression's synthetic importable client reproduced the error before the
fix in the simulator-free venv. The same case uses the real client when installed.

From the repository root (`d=/tmp/opencode/nexuml-carla-O6kEUc`):

```bash
bash "$d/run-live.sh" export-fix
"$d/venv/bin/python" -m pytest tests/core/test_carla_environment.py -m 'not requires_system' -q
.venv/bin/python -m pytest tests/core/test_carla_environment.py tests/core/test_export.py \
  tests/core/test_export_dependency_boundary.py -m 'not requires_system' -q
```

- Live launcher: same pinned binary/wheel and GPU lock, new container `aab619db83cd`,
  RPC `60541`, stream `38411`. Opt-in `-m requires_system` smoke: **1 passed,
  8 deselected**; lifecycle probe: exit 0 (matched frames `2407/2408`, timeout
  `2423`, rejected foreign tick `2453`, injected cancellation). The independent
  before/after JSON files match byte-for-byte for settings and actor IDs/types.
  The container was removed and independently confirmed absent.
- Wheel venv, no live tests: **8 passed, 1 deselected**. Regular venv (client absent),
  CARLA plus export suites: **51 passed, 1 skipped, 1 deselected**.
- Fresh processes loaded the same live-trained artifact using `load_package`
  with the wheel and `load_inference_package` without it. The second process
  rejected imports of CARLA, ROS, Gymnasium, TorchRL and Ray. Both produced
  `[-0.6084176898, 0.6309055090, 0.3413906693]` on one fixed synthetic RGB/ego
  input. CARLA was absent from archive externs, metadata and requirements.
  `pipeline.package` SHA-256:
  `1a0efa40790db36ce37429b2559083c4087397482c95088893870dc58c177798`.

Post-fix receipts in the same scratch directory: `export-graph-before.log`,
`regression-before.log`, `run-live-export-fix.log`, `pytest-live-export-fix.log`,
`adapter-export-fix.log`, `*export-fix.json`, `pytest-cp312-wheel.log`,
`pytest-export-regular.log`, and `reload-live-policy.log`. No evaluator,
task-list or acceptance-matrix edits.
