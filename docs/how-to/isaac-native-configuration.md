# Author native Isaac task configuration

Use `IsaacLabEnvironment` for a trusted installed native task, portable I/O
meaning and typed configuration. The current environment component version is
**2**, pinned to Isaac Lab `70ab24fec3eabf51222817db0cdb5a4e7133bd7e`
(Lab 3.0 / Sim 6.2 / Python 3.12). Restore current-schema definitions directly;
there is no legacy native-config reader. Historical EA results do not qualify
this runtime.

Static scenario discovery, `describe()`, resolve and YAML restoration do not
launch Isaac, execute configuration functions or allocate CUDA. Native execution
still requires the prepared, operator-approved stack, assets/licences and one
available CUDA device. Do not replace an active environment or install another
stack to run these examples. Native runs require at least **40 GiB free disk**.

## Typed functions, not override strings

Put customization in an importable operator-owned module, not a script's
`__main__`, lambda, closure or bound method. Keep native annotation imports out of
module initialization:

```python
# my_tasks/configuration.py — trusted, installed local code
from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnvCfg


def set_horizon(cfg: ManagerBasedRLEnvCfg, *, seconds: float) -> None:
    if seconds <= 0:
        raise ValueError("seconds must be positive")
    cfg.episode_length_s = seconds


def replace_horizon(
    cfg: ManagerBasedRLEnvCfg, *, seconds: float
) -> ManagerBasedRLEnvCfg:
    replacement = deepcopy(cfg)
    set_horizon(replacement, seconds=seconds)
    return replacement
```

```python
from my_tasks.configuration import set_horizon
from nexuml_library.environments.isaaclab import (
    IsaacLabEnvironment, IsaacTaskConfigSpec, isaac_task_config,
)

definition = IsaacLabEnvironment(
    task="Isaac-Cartpole",
    physics="isaacsim_physx",
    task_config=isaac_task_config(set_horizon, seconds=2.0),
)
# Equivalent inert representation, also usable during static restoration:
customization = IsaacTaskConfigSpec(
    factory="my_tasks.configuration:set_horizon", kwargs={"seconds": 2.0},
)
```

The synchronous function receives the **resolved native config**, not an empty
config or live environment. Annotate its first positional config parameter, every
named option and its return. Options must be finite JSON-safe values; positional
option lists, variadic signatures and native objects in YAML are rejected.
Annotations are not evaluated or used to coerce arguments. Returning `None`
retains the mutated input; returning a config selects that replacement.

The native job resolves the registered task and presets, applies runtime-owned
count/device/seed/final-observation settings, invokes customization, then resolves
and validates the result before stepping. Do not allocate an application or
environment in a customization function or at module import. Count comes from
`training.collector.num_envs_per_collector`, not the environment declaration.
Customization cannot change protected count, device, seed, final-observation or
selected physics/runtime ownership.

`physics` selects a named native backend, for example `isaacsim_physx` or
`newton_mjwarp`; `presets=("installed_preset_name",)` selects ordered native preset
names declared by that task. `physics=None` leaves the task's native selection.
Unknown presets and incompatible selections fail; there is no automatic backend
fallback or universal task/backend support claim. Do not pass strings such as
`"sim.dt=..."` as presets. Solver, material, reward, reset, randomization and
curriculum changes belong in the typed function and must retain a matching
physical/task contract.

Noise/history are not disabled globally. The maintained manager-based state
recipes explicitly select `enable_corruption=False` and `history_length=0` through
their configuration function. Custom tasks retain native settings unless their
function changes them. Replay never silently calls `play_mode()`; use a separate
compatible evaluation definition for deliberately different semantics.

## Contracts, bindings and a registered custom task

An arbitrary registered task needs both an explicit `InteractionContract` and
`IsaacBindings`. Field shapes exclude the environment batch; declare dtype,
physical meaning, control period and action decoding, not just matching widths.
The contract's `num_envs` stays 1; the collector owns the live count.

Bindings map declared observation fields to tuples of **literal mapping keys**:
`{"observation": ("policy",)}` reads native `observations["policy"]`.
`("sensors", "camera.rgb")` means two keys, not a dotted attribute expression.
Every declared observation must be mapped, and `action` names the single declared
action field. Bindings are not configuration overrides or automatic contract
inference. A binding declaration alone does not establish sensor support.

The installed `isaac-custom-cartpole-ppo` scenario is a real Gym-registered
`NexuML-Custom-Cartpole` manager task, not a renamed built-in. Its registration
starts with a 3-second horizon and spacing 2; the scenario customizes the resolved
config to **2 seconds** and **spacing 3** with `newton_mjwarp`. It explicitly reuses
the maintained Cartpole contract (float32 state `[4]`, continuous action `[1]`,
100-N effort scale, 60-Hz control) and binds `observation` to `("policy",)`:

```python
from nexuml_library.scenarios.reinforcement.isaaclab_custom import (
    isaac_custom_cartpole_ppo,
)

scenario = isaac_custom_cartpole_ppo(
    total_frames=32, frames_per_batch=16, num_envs=2,
)
definition = scenario.interaction.environment
assert definition.task == "NexuML-Custom-Cartpole"
assert definition.task_config.kwargs == {
    "episode_length_s": 2.0, "env_spacing": 3.0,
}
```

For external tasks, install the trusted registration package locally and use
Gym's `"my_tasks:Registered-Task"` form to import its registrations at execution.
Static restoration does not run that package. A missing registration or wrong
native workflow, spaces, bindings, rates or protected settings is an error, not
permission to infer a replacement contract.

## Normal CLI and whole-job Python ownership

Use the ordinary CLI in the prepared environment; no manual application-launch
prefix or alternate trainer is needed:

```bash
export PYTHONPATH="$PWD/src:$PWD/library/src"
export NEXUML_LOGS_ROOT=/tmp/opencode/isaac-custom-job
.venv/bin/python -m nexuml resolve isaac-custom-cartpole-ppo \
  --output /tmp/opencode/isaac-custom.yaml
# Native execution: only with the approved stack/device/assets already available.
.venv/bin/python -m nexuml train --config /tmp/opencode/isaac-custom.yaml \
  -O training.total_frames=32 -O training.frames_per_batch=16 \
  -O training.collector.num_envs_per_collector=2 \
  -O training.algorithm.update_epochs=1 -O training.algorithm.minibatch_size=8
.venv/bin/python -m nexuml train --trainer-checkpoint /approved/run/intermediate.ckpt
# Separately exported complete policy; replay runs in a fresh process.
.venv/bin/python -m nexuml evaluate /approved/run/policy --device cuda:0 \
  -O training.evaluation.episodes=2 -O training.evaluation.max_steps_per_episode=120
```

These are bounded execution instructions, not evidence of native acceptance.
Training requires one synchronous direct collector, one GPU and matching
simulation/storage/policy/learner devices; `frames_per_batch` must be divisible by
the collector count. Automatic start/periodic/final in-fit evaluation and video
remain disabled for these recipes; use separate-process evaluation.

`train` scopes the effective `session.scenario`, including checkpoint-restored
definitions, through final outputs and failure cleanup. `evaluate` scopes the
selected `interaction.evaluation_environment` when present, otherwise the training
definition, through evaluation artifacts/logger cleanup. It does not always own
the training environment. A completed-budget resume allocates no simulator.

Python callers must keep the same scope around **all** work, not just `run()`:

```python
from nexuml.core.export import export_package
from nexuml.training.lightning import NexuSession

session = NexuSession(scenario=scenario)  # also resolves a supplied trainer checkpoint
definition = session.scenario.interaction.environment
with definition.job_scope():
    try:
        result = session.run()
        export_package(result.policy, "/approved/new-output/policy")
        # Save any additional checkpoints/provenance before leaving this scope.
    except BaseException as primary:
        try:
            session.finalize_loggers("failed")
        except BaseException as cleanup:
            primary.add_note(f"Logger cleanup also failed: {cleanup!r}")
        raise
    else:
        session.finalize_loggers("success")
```

For Python replay, scope the selected evaluation definition returned by
`nexuml.evaluation.policy.resolve_policy_evaluation(scenario)` around
`evaluate_policy(...)` and any artifact/logger cleanup. Same-job nested scopes
join the owner; conflicting definitions/runtimes reject. Entering a scope is lazy.
Native launcher state determines whether Kit owns process-terminating fast-close;
finish outputs and explicit cleanup before scope exit, not in `atexit`.

## Trust and unqualified paths

Importable does not mean sandboxed: configuration functions, task registrations
and policy packages are trusted executable local code. Remote requests must not
choose factories, recipes, processors, archives or private paths. Actor-only
inference does not reconstruct a simulator or execute saved configuration functions.

`ViserSpec` is an inert, loopback-only declaration; omitted means no viewer.
Runtime activation for a requested purpose currently **fails closed** pending job
integration and Milestone C qualification. The isolated
[constructor boundary](../explanation/isaac-spectator-boundary.md) is separate from
actual training/replay browser acceptance. Do not advertise working
viewing, policy sensors or MARL acceptance from these declarations or state-only
examples. Native/backend, sensor and browser gates require their own real checks.
See [SO-101 local fit](so101-local-fit.md) for its separate capture/qualification
and unchanged quality protocol.
