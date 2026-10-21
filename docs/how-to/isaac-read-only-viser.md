# Inspect a native job with read-only Viser

Viser is an explicitly enabled spectator, not a simulation controller. Omit
`viewer` to keep headless jobs independent of Viser. Static configuration and
actor-only inference never start a server.

The prepared-stack Newton-Warp Cartpole recipes have native browser training and
separate replay qualification, including mesh/frame markers and cached plots.
Other native task/backend combinations require their own qualification; this is
not universal graphics or camera support.

## Enable training and separately enable replay

Use the existing environment declaration and whole-job scope:

```python
from nexuml.core.serialization import lower_model
from nexuml_library.environments.isaaclab_config import ViserSpec
from nexuml_library.scenarios.reinforcement.isaaclab_sensors import isaac_mixed_cartpole_ppo

scenario = isaac_mixed_cartpole_ppo(total_frames=256, frames_per_batch=64, num_envs=2)
scenario.interaction.environment = scenario.interaction.environment.model_copy(
    update={"viewer": ViserSpec(port=8080, max_visible_envs=2, lookat=(0.0, 0.0, 2.0))}
)
portable = lower_model(scenario)  # Inert data; no native runtime or server.
```

Train through ordinary `nexuml train` or the
[whole-job Python scope](isaac-native-configuration.md#normal-cli-and-whole-job-python-ownership).
The configured purpose defaults to `("train",)`. A saved training spectator does
**not** enable viewing when the policy is evaluated in a fresh process.

For replay, explicitly replace the selected evaluation environment's viewer:

```python
from nexuml.evaluation.policy import evaluate_policy
from nexuml.core.types import PolicyEvaluationSpec

definition = scenario.interaction.environment.model_copy(
    update={"viewer": ViserSpec(purposes=("eval",), port=8080)}
)
scenario.interaction.evaluation_environment = definition
scenario.policy_evaluation = PolicyEvaluationSpec(episodes=10, max_steps_per_episode=120)
policy = policy.to("cuda:0")
with definition.job_scope():
    evaluation = evaluate_policy(policy, scenario)  # Existing evaluation owner.
```

`policy` is the complete loaded policy, not just its neural pipeline. Native replay
uses the trusted installed task/configuration package; clean actor inference does
not reconstruct that task. Selecting replay viewing never changes actor inputs,
control semantics, environment count or budgets.
Short examples may finish before a browser connects; choose the job budget
deliberately rather than using display pause to prolong simulation.

## Private access

The default endpoint is `http://127.0.0.1:8080`. Both `127.0.0.1` and `localhost`
declarations bind the actual IPv4 loopback listener. Sharing is disabled.

For a remote prepared host, forward TCP instead of opening a public listener:

```bash
ssh -N -L 8080:127.0.0.1:8080 user@prepared-host
```

Then open `http://127.0.0.1:8080` locally. VS Code's Ports panel can forward the
same remote port privately. Do not change the native bind address to `0.0.0.0`.
An occupied port, silent port reassignment, unsupported source identity or startup
failure rejects the viewer-requested job; there is no silent headless fallback.
TCP `TIME_WAIT` from a closed connection does not constitute a live listener.

## Display controls and boundaries

- Move/zoom the browser camera or use **Reset Camera**. This never moves a policy
  camera, edits USD/physics transforms or requests a sensor update.
- **Pause Display** toggles rendering only. Training and control continue.
- **Visible Environment** selects a displayed environment; **Configured** restores
  the declared selection/cap. Selection is deterministic and consumes no task RNG.
- **Markers** controls native marker overlays, including mesh and frame markers.
- **Live Plots** shows supported cached manager terms. Expand a term to see its
  chart. Only native action, observation, reward, termination and curriculum cache
  readers are admitted; observations are skipped until a cache exists. Custom
  manager collectors, command getters and arbitrary scalar callbacks are not
  qualified spectator reads.

No simulation pause/reset, task/configuration edits, action/force injection or
timing/budget controls are registered. Stale/forged controls cannot cross the
native pause/reset/force handoffs. Browser absence or disconnect does not terminate,
pause or reset the job; reconnecting restores the current display.

The adapter reuses native Viser/Newton scene rendering with a separately owned
native model/state snapshot. Transform publication copies into spectator-owned
storage rather than borrowing the solver's mutable transform array. No vendor
files, global constructors or policy sensor paths are patched.

Sensor streaming, native recording/static playback, picking and notebook recording
helpers remain unsupported. A browser's software WebGL rendering is not evidence
of server-side RTX/Vulkan sensor support. Only explicitly exercised native
task/backend/sensor combinations may be advertised; see
[sensor qualification](isaac-sensors-and-policy-state.md).

## Ownership and verification

Completion, failure and cancellation close the private viewer and callback threads
before native application termination. Python callers must retain the whole-job
scope through export/checkpoint/logger finalization. Even a failure inside native
environment construction must release its pending viewer.

Constructor/HTTP/WebSocket denial, connected storage isolation, noisy observation
cache/RNG, real browser training, separately enabled replay and lifecycle checks
are distinct opt-in gates. Screenshots alone establish neither sensor semantics
nor learning correctness. No installation or graphics provisioning is authorized
by enabling a spectator.

For the opt-in browser regression, select **already-installed** Playwright and
Chromium, then run the existing native test harness:

```bash
export NEXUML_PLAYWRIGHT_MODULE=/approved/node_modules/playwright
export NEXUML_CHROMIUM_EXECUTABLE=/approved/chromium
NEXUML_RUN_SIMULATOR_TESTS=1 NEXUML_RUN_SYSTEM_TESTS=1 \
  .venv/bin/python -m pytest tests/core/test_isaaclab_integration.py \
  -k native_spectator_real_browser_train_replay_and_lifecycle
```

`NEXUML_BROWSER_SWIFTSHADER_ICD`, when explicitly supplied, selects an existing
Chromium-bundled software ICD **only for the browser child**. It does not alter the
native job's graphics or Python environment. Missing tools skip this opt-in
regression and leave the browser gate unverified; they never authorize an install.
