# SO-101 learning follow-up: predeclared protocol

The [original comparison](baseline.md#matched-so-101-result-and-scope-decision)
remains unchanged. This is a diagnostic experiment on the same joint-tracking
task, not a new task or an authorization to expand placement.

- **Controls:** Same native PhysX task, home, goals, 30 Hz control, rewards,
  tolerances, networks, optimizer settings, rollout width, and seeds (17, 43,
  97) as the original. Direct goal-to-actuator commands must solve every
  evaluated lane before treating any learned outcome as valid.
- **Training:** For each learner and seed, compare the original exploration
  initialization against a lower one. RSL-RL starts at `0.7` or `0.2` Gaussian
  standard deviation. NexuML retains its original, observation-dependent
  scale or shifts only the scale-head biases by
  `softplus⁻¹(0.2) − softplus⁻¹(0.7)`. Means, other parameters, entropy,
  optimizer, and learnable scale dynamics are unchanged. These are
  within-learner interventions, not identical action distributions.
- **Budgets:** One uninterrupted fit per learner, seed, and setting. Export
  inference actors at 0, 131,072, 524,288, and 1,048,576 physical
  transitions (0, 128, 512, and 1,024 completed PPO rollouts).
  Use seed-blocked serial execution, with condition order shuffled within
  each block by `random.Random(261126)`. Don't stop a fit based on interim
  scores or select a better budget after looking.
- **Evaluation:** In a process independent of training, use deterministic
  actions from each export on the same 32 native lanes reset with seed
  10000. All three built-in goals must appear. Use only the first episode
  per lane and use the pre-reset terminal observation. Log per-goal
  native success, best worst-joint error, longest in-tolerance streak,
  and fraction of joint commands at or beyond `0.99` normalized magnitude.
  Exclude repeated lanes as independent evidence: the independent training
  units are the three seed pairs, and the three poses are not unseen goal
  configurations. Report the equal-goal average for each seed and paired
  differences by learner at each checkpoint. A shorter no-learning control
  or two-rollout smoke run is validation, never an experiment result.
- **Interpretation:** Longer exposure is supported if final-budget
  equal-goal success exceeds the 131,072-transition checkpoint in at least
  two seeds for a setting. Lower initialization is supported if it improves native
  success at a matched budget in at least two of three pairs without
  reducing it in the third; best error/hold/saturation are diagnostic if
  success remains zero. Report seed-level disagreements. This does not
  prove generalization or alter the failed original expansion gate.

Run each stage in its own process in the prepared Isaac environment. The
same driver can check export and metrics with `--smoke`, which uses two
1,024-transition rollouts and is excluded from the comparison.

```bash
.venv/bin/python scripts/isaaclab/so101_learning_followup.py control "$OUTPUT/control"
.venv/bin/python scripts/isaaclab/so101_learning_followup.py train "$OUTPUT/rsl-rl-17-low" --learner rsl-rl --seed 17 --low
.venv/bin/python scripts/isaaclab/so101_learning_followup.py evaluate "$OUTPUT/rsl-rl-17-low"
```
