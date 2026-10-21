# Train state-only Isaac Cartpole

Use the prepared robotics environment. This example needs one CUDA GPU and the
pinned Isaac stack; it does not install packages, start a public server or use a
different trainer. It learns continuous cart effort from four native state values,
not camera pixels or Gymnasium's categorical CartPole actions.

## Train and export

From the repository root:

```bash
source .venv/bin/activate
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  nexuml train --scenario-file scripts/isaaclab/cartpole.py
```

The file only declares a `scenario()`. The CLI owns training, native lifetime,
logger cleanup and export. On completion it prints a unique complete policy path:
`rl-runs/cartpole-<id>/policy`. Keep that path for evaluation; reruns do not reuse it.

The private read-only spectator is at **http://127.0.0.1:8080**, with at most four
visible environments. An occupied port fails rather than replacing another server.
For a remote machine, use a private SSH tunnel/port forward, not a public bind.
The viewer closes when the job finishes. Its controls do not actuate the task.

This is the separate `isaac-cartpole-learning-ppo` reference:

- Native `Isaac-Cartpole`, **Newton mjwarp**, unchanged native reward/reset rules
  and five-second horizon (300 control steps at 60 Hz).
- 512 environments, 8,192 frames/batch: **16 steps per environment**, not 8,192
  sequential steps on one cart.
- 2,097,152 total frames, training seed 0; 32×32 actor and critic.
- PPO: five update epochs, four 2,048-sample minibatches, actor/critic learning
  rate 0.001, gamma 0.99, GAE lambda 0.95 and entropy coefficient 0.005.

These are scenario-local settings. The ordinary PPO defaults and existing
state/sensor smoke recipes are unchanged. The tiny four-step sensor example is
not a replacement for this learning reference.

## Evaluate the exported policy

In a fresh process, replace the placeholder with the path printed by training:

```bash
nexuml evaluate rl-runs/cartpole-<id>/policy --device cuda:0 \
  -O training.evaluation.episodes=32 \
  -O training.evaluation.seed=10000
```

This uses the same native task, complete exported policy and held-out reset seeds
10000–10031. Do **not** add `max_steps_per_episode`: let native termination or the
full five-second horizon end each episode. In-fit evaluation is disabled so it
does not compete with the training runtime.

Look at `eval/rl/return_mean`, `length_mean`, `terminated` and `truncated`.
The native reward is integrated over simulation time: near-ideal five-second
returns approach **5**, not 300. A zero-weight `success_rate` reward plot remains
zero and is **not** a learning score. Mean per-transition training reward is also
not the same as an episode return.

Training viewing does not automatically enable viewing during evaluation. To
watch a bounded replay explicitly:

```bash
nexuml evaluate rl-runs/cartpole-<id>/policy --device cuda:0 \
  --scenario-file scripts/isaaclab/cartpole.py \
  -O training.evaluation.episodes=128 -O training.evaluation.seed=20000 \
  -O 'interaction.evaluation_environment.viewer.purposes=["eval"]'
```

Simulation runs faster than real time; 128 episodes leave time to open the browser.
The server still closes at the end of this finite replay.

See [private Viser ownership](isaac-read-only-viser.md) and
[native configuration](isaac-native-configuration.md) for customization. Keep the
operator's monitored **4 GiB free-disk stop** during native jobs.

This reference must be assessed against an untrained policy on identical held-out
seeds. A single training seed does not establish robustness across seeds, backend
equivalence, or learning from mixed sensor inputs.
