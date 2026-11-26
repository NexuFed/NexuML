# Local policy playback — partial 4a qualification

The local `nexuml play` and `nexuml.interaction.playback.play_policy` paths
use the complete actor and the same semantic contract check, inference RNG
isolation and native job scope as evaluation. They do not create a trainer
or produce benchmark metrics. Providers must opt in to simulated playback.
Unsupported viewers, benchmark-only environments, unsupported remote
profiles and incompatible contracts fail closed. A runtime returning native
autoreset state acknowledges exactly the ended rows; a runtime requiring
explicit resets uses the same selected reset boundary. Neither path uses
terminal observations as the next action input on a finished row.

## Pinned launcher audit

Isaac Lab revision `70ab24fec3eabf51222817db0cdb5a4e7133bd7e` normalizes
`visualizer` and `livestream` in `isaaclab/app/sim_launcher.py`. If
`livestream` is omitted or `-1`, it inherits the `LIVESTREAM` environment
variable. This can implicitly select Kit, including for Viser. NexuML
passes `livestream=0` and rejects a conflicting override before allocation.
Offscreen sensors still resolve from the native sensor/renderer config:
`test_native_multimodal_values_terminal_scene_and_history[package]` passed
alongside the state-only and Viser guards (`3 passed in 41.06 s`).

On this pinned version, local viewing needs the native `kit` visualizer
and non-headless display; `HEADLESS` can independently force headless.
`isaaclab_physx/app/kit_launcher.py` treats livestream mode `1` as public
advertisement (uses `PUBLIC_IP`) and mode `2` as private. Both enable the
Kit stream extension and set TCP 49100 signaling / UDP 47998 media.
Neither mode is enabled by this change: the current profile has no
prepared local, X11 or TigerVNC display, and no approved WebRTC client
lane. The CI job consumes an already-prepared `NEXUML_ISAAC_IMAGE`; the
existing image recipe is not an automatic publication path. Do not treat
Kit startup, port binding or a prior editor session as playback evidence.

## Executed local proof

Runtime matches [custom-direct-task.md](custom-direct-task.md).

- `nexuml play PATH --device cuda:0 --num-envs 8 --steps 60 --seed 23`:
  **60 ticks, 480 transitions, 24 episodes**. The native integration test
  rebuilt its policy and passed both scalar evaluation and vector play.
  Logs: `/tmp/opencode/nexuml-decouple-playback-final-5XHR05/`.
- `nexuml play --trainer-checkpoint PATH --device cuda:0 --steps 4 --seed 23`:
  **4 ticks, 4 transitions** in a fresh N=1 job, no optimizer or collection.
  The supported trainer-checkpoint path reuses evaluation's policy export
  helper. Logs: `/tmp/opencode/nexuml-decouple-headless-play-x0NQgH/`.
- Native mixed RGB/depth/segmentation/state PPO integration:
  **6 ticks, 24 transitions, 4 episodes** through an exported package in
  headless playback with four environments. The same test passed checkpoint
  continuation, torch-package inference without simulator or learner modules,
  and fresh evaluation. Logs: `/tmp/opencode/nexuml-decouple-playback-mixed-M3NxDv/`.
- The existing trained native Pendulum IPPO export ran 10 ticks with N=1
  and N=3, yielding 10/30 transitions and 2/6 completed episodes.
  The native job explicitly loaded the trusted `isaaclab_marl_config`
  module from `tests/fixtures`; the export does not provide native task
  code. Logs: `/tmp/opencode/nexuml-decouple-marl-qualified-nOD0Gt/`.
- The browser connected to the actual public CLI (`--viewer viser`,
  `--num-envs 8`, `--real-time`), with a play-only Viser configuration and
  visible rows 1 and 3. Chromium rendered changing native output over
  loopback on TCP 35585, with no simulation mutators on the spectator.
  SIGTERM exited with 143, and the exact port could be rebound immediately.
  Logs and rendered frames: `/tmp/opencode/nexuml-decouple-viser-play-dsKeJO/`.
  This does not qualify a tunnel, control mutation, or a remote worker.
- Hermetic playback, existing evaluation, native config/runtime, Arena
  protocol, simulated ROS and CARLA coverage: **493 passed, 2 live tests
  deselected**. Covered independent per-row reset, fixed actor tensors,
  seeded stochastic action parity, caller RNG restoration, clean SDK stop,
  unsupported playback preflight and unchanged trainer/evaluator options.
  Scoped lint/type and strict OpenSpec checks passed. `npm --prefix docs run
  check` passed with zero diagnostics and rendered links/assets checked.

## Opt-in Viser controls

The browser callback only queues bounded playback requests. The regular
playback loop owns simulation and applies requests between control ticks.
Allowed speeds are 0.25×, 0.5×, 1× and 2× wall-clock targets, never changes
to the physics timestep. Selected explicit reset is available only on a
native manager task; direct tasks retain reset-all. The train/eval path
does not receive an inbox even when a trusted config opts in to play
controls for the same Viser definition.

Real Chromium drove a trained Newton Cartpole export with three simulated
instances and two initially visible. Pause held tick **101**. Single step
reached **102** and no more. At tick 102, reset-row-1 changed episode
counters `[36, 47, 37] → [36, 0, 37]`; reset-all changed them to
`[0, 0, 0]` with no physics advance. All resets and steps ran on the owner
thread. The browser selected 2× pacing, resumed and disconnected; ticks
continued and a second browser reconnected. SIGTERM exited 143 and the
loopback port was rebound. Native, browser, and per-reset receipts:
`/tmp/opencode/nexuml-decouple-interactive-play-DKA409/`.

The post-control scoped suite passed **503 tests**, with two live system
tests deselected. It verifies bounded/invalid control admission and
independence of train, eval and play construction. Scoped lint/type and
strict OpenSpec checks passed. **4 native read-only spectator tests passed
in 44.48 s**, including identical noisy/nonnoisy trajectories and cleanup
after partial startup. Logs:
`/tmp/opencode/nexuml-decouple-controls-spectator-QwaKU1/`.

## Bounded headless recording

Playback uses the same RGB validation, sizing and GIF/MP4 encoders as
evaluation but publishes a playback manifest, not evaluation metrics.
The CLI requires a separate frame budget and a new output directory.
At the limit it finalizes the video without ending playback. The manifest
separates completed native control ticks from monotonic capture timestamps
and identifies the policy/checkpoint bytes and configured environment.

The headless N=1 Kit PhysX manager reference recorded both formats, with
**three captured native ticks** in a run that completed **eight ticks**.
The policy digest matched, capture times increased, and the GIF and
three-frame MP4 decoded. Receipts:
`/tmp/opencode/nexuml-decouple-native-record-LLX2De/`. Hermetic tests proved
that playback continues after publishing, that existing media targets are
not overwritten, and that cancellation finalizes its captured prefix.
In the native continuous run, SIGTERM at tick 3 exited **143**; the
manifest recorded `cancelled` and the MP4 decoded exactly the two
previous frames. Receipts:
`/tmp/opencode/nexuml-decouple-record-cancel-NXd3Ec/`.
The focused capture/evaluation/playback/native-validation suite passed
**443**, with 14 opt-in cases skipped.

## Installed playback boundary

Offline-built NexuML and library wheels passed all **11 playback tests**
from an isolated installation target using the prepared dependency
environment. The subprocess played an exported CPU actor for 12 native
transitions while forbidding and asserting zero attempts to import
Lightning, NexuML training, TorchRL, Ray, Isaac, Warp/Newton and ROS.
Discovery no longer loads the dataset exporter or trainer callbacks on
behalf of unrelated declarations. The related dataset export, callback,
native API and CLI regression passed **111 tests**, with one opt-in case
skipped. Scoped lint, types, docs and strict OpenSpec checks passed.

Wheel and test receipts (not published):
`/tmp/opencode/nexuml-decouple-clean-playback-wheels-HqKeIe/`.

## Policy-owned playback state (4a.2)

Playback uses the existing `CompiledPolicy` state API per session. The same
mask clears history, recurrent state and chunk cursors for ended or selected
rows before acting on the reset observation. A viewer reset-all starts new
policy state without stepping the simulator. Isaac forwards raw package-owned
sensor inputs during play so that only the package advances that history.
Externally processed observations stay runtime-owned. The training and
scientific evaluation inputs are unchanged.

The three-lane tests check different termination/truncation ticks, selected
and full viewer resets, package history, and a GRU and two-command chunk
fixture without changing actor weights. Native tests passed for the exported
mixed-sensor actor (four lanes, six ticks) and the camera/final-observation
oracle (**2 passed in 58.98 s**). Logs:
`/tmp/opencode/nexuml-decouple-stateful-play-6zSE75/`. The broader focused
suite passed **378 tests**. A direct baseline comparison for schema-1
dataset local fit matched two fixtures exactly in manifest, counts, metrics,
and updated tensors; the ordinary PPO numerical baseline check still passes.

The externally processed mixed-sensor policy also trained, exported, and
replayed through the unchanged runtime-owned history path. The paired
Isaac/CPU MuJoCo regression ran in the same check (**2 passed in 149.26 s**).
Logs: `/tmp/opencode/nexuml-decouple-stateful-parity-uMwfbL/`.

## Bounded task commands (4a.3)

Only an explicitly controlled Viser play session resolves a task's optional
native reset binding. The browser sends a declared name and choice through
the existing bounded queue. The owner checks the choice again, applies it at
a full reset boundary, clears policy state, and counts the intervention
separately from completed episodes. There are no free-form actuator commands.

Chromium used a trained SO-101 export on three native PhysX lanes with two
visible. **Positive** and **Negative** reset all lanes at paused tick 74
without advancing physics. The new policy inputs exactly matched the native
target tensors, then single step reached 75. Resume and browser reconnection
worked; SIGTERM exited 143 and TCP 34133 could be rebound. This run exposed
the Viser USD import before Kit startup; deferring Viser construction until
after the launcher enters fixed it without changing the backend. Both Newton
spectator trajectory comparisons and native SO-101 held outcomes passed
(**3 tests in 38.91 s**). Browser and native logs:
`/tmp/opencode/nexuml-decouple-task-commands-native-FtQchz/`.
The existing real-browser read-only gate also passed after the startup-order
fix (**1 test in 213.66 s**). It compared native training state and rollout
trajectories with headless runs and verified failure/cancellation cleanup.
Logs: `/tmp/opencode/nexuml-decouple-spectator-readonly-final-1KbSf4/`.

The broader regression passed **776 tests**, with 14 opt-in cases skipped
and two live system tests deselected. Viser controls and headless GIF/MP4
capture are qualified separately; viewer recording, Local Kit and WebRTC,
and remote Ray Jobs playback are not enabled by this proof.

## Ray Jobs playback transport (4a.5, partial)

The public CLI and `submit_ray_job(..., operation="play")` submit an explicit
resolved scenario through the existing Jobs SDK. Packages, checkpoints and
recordings use absolute worker paths, not submitter paths. The driver does
not load the artifact or start a simulator. The worker runs normal local CLI
playback in a separate module process. The submission result is a job ID,
not proof of successful playback or viewer readiness.

The existing Ray job runner previously embedded `app(argv)` for training.
A real GPU Cartpole producer using that path reached the worker, then
failed with a Kit crash (exit 139). The ordinary module subprocess was
already necessary for evaluation; it now executes train, export, evaluate
and play alike. Ray signals the owned job process tree, and the bootstrap
waits for the CLI process and cleans its temporary resolved configuration.
Tests verified literal argv, discharge of owned resources on local SIGTERM,
and unchanged CLI/SDK job semantics: **273 passed**. Live producer and
remote playback receipts are separate from this transport change.

An actual local process launched from the generated worker entrypoint
recorded two frames from continuous Pendulum playback. Sending SIGTERM to
the owned process group returned 143, removed the temporary worker config,
and left no process group. The transport submission was mocked, so this
does not establish Ray service cancellation. Receipt:
`/tmp/opencode/nexuml-job-play-cancel-LqfTjf/`.

The next bounded existing-pool GPU job with the ordinary module bootstrap
was not admitted in its admission window. A stop request was sent for that
job only. It had not yet reached a worker when last checked, so it did not
qualify the new bootstrap on a remote Isaac worker or produce a policy for
playback. Do not retry the same allocation or count the earlier Kit crash
as a playback attempt. Receipts: `/tmp/opencode/nexuml-remote-play-ZtrXQj/`.

## Native Kit display probes (4a.1 and 4a.4, partial)

On an existing X11/RDP display, Lab's default non-headless Kit experience
reached at least **60** control ticks on an actual exported two-lane
Cartpole policy before the 140 s bound cancelled the requested 240 ticks.
That experience loads the native stage, script editor and prim-manipulator
extensions. It does not prove a bounded playback-only display. Selecting
the upstream headless-rendering experience while requesting a non-headless
window did not reach the viewport or control-loop witness. No X11/RDP Kit
viewer, native local desktop, TigerVNC or WebRTC support is inferred.
Probe logs: `/tmp/opencode/nexuml-kit-local-native-WNPXME/` and
`/tmp/opencode/nexuml-kit-minimal-dpOhMT/`. No network route or display was
provisioned and no fallback was enabled.
