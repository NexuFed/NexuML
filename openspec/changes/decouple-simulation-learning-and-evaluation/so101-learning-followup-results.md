# SO-101 learning follow-up: results

**Nine of 12 trained policies solved all 32 replay episodes by 524,288
transitions and again at 1,048,576.** All 12 fits and 48 fresh-process
checkpoint evaluations completed. The scripted positive control independently
passed 32/32 in 8–11 control ticks.

## Success by learner and budget

Entries are the percentages of successes averaged equally across the three
built-in goals, reported for seeds **17 / 43 / 97**. Each actor was replayed
in 32 lanes (7 home, 11 positive, 14 negative). In this experiment each actor
at each checkpoint either solved all 32 episodes or none. The three training
seeds are the replicates, not the 32 lanes.

| Learner | Initial exploration | 0 transitions | 131,072 | 524,288 | 1,048,576 |
| --- | --- | --- | --- | --- | --- |
| RSL-RL PPO | Original | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 |
| RSL-RL PPO | Lower | 0 / 0 / 0 | 0 / 0 / 0 | 100 / 100 / 100 | 100 / 100 / 100 |
| NexuML PPO | Original | 0 / 0 / 0 | 0 / 0 / 0 | 100 / 100 / 100 | 100 / 100 / 100 |
| NexuML PPO | Lower | 0 / 0 / 0 | 0 / 0 / 0 | 100 / 100 / 100 | 100 / 100 / 100 |

- **Budget:** The [predeclared criterion](so101-learning-followup.md) was met
  for both NexuML settings and lower-exploration RSL-RL (all three seeds
  improved from 131,072 to the final budget), but not original RSL-RL (none
  improved). The first passing checkpoint was 524,288; the exact first
  passing transition is unknown.
- **Exploration:** Paired differences (lower minus original) were 0 percentage
  points for all seeds at 0 and 131,072 transitions. At both later
  checkpoints, RSL-RL gained 100 points in each pair; NexuML remained tied in
  every pair. Lower initialization met the success criterion for RSL-RL,
  not NexuML. The two learners do not use identical action distributions.

## Final-checkpoint diagnostics by goal

Each cell shows the mean **best worst-joint error (rad) / longest
in-tolerance streak (ticks) / saturated joint commands (%)** for that goal.
The error is the closest each episode came, not its final error. Commands at
or beyond `0.99` normalized magnitude count as saturated. Goal-level means
by seed at every checkpoint, native success flags, and first-episode rows are
in the receipts.

| Learner / exploration | Home | Positive | Negative |
| --- | ---: | ---: | ---: |
| RSL-RL / original | 0.0782 / 0.33 / 65.71 | 0.0524 / 0.67 / 53.41 | 0.0342 / 0.67 / 55.52 |
| RSL-RL / lower | 0.0096 / 8 / 0 | 0.0086 / 8 / 1.57 | 0.0079 / 8 / 1.50 |
| NexuML / original | 0.0118 / 8 / 0 | 0.0256 / 8 / 0 | 0.0205 / 8 / 0 |
| NexuML / lower | 0.0087 / 8 / 0 | 0.0149 / 8 / 0 | 0.0146 / 8 / 0 |

Original-exploration RSL-RL got closer to the goals with more training, but
every final episode timed out. Some lanes entered the 0.05 rad tolerance
band briefly; none held it for eight ticks. The other three conditions
finished in 8–10 ticks at the final checkpoint.

## Evidence and scope

- [Receipts](so101-learning-followup-receipts.tar.gz) include the run order,
  source and artifact SHA-256 hashes, versions, rollout timings and losses,
  and raw evaluation data. Archive SHA-256:
  `c37f4ff4348fb0c22d483f9115634e5b9460647e0866fe3c4bcf47fad51850ce`.
  Local model exports and process logs are in
  `/tmp/opencode/nexuml-so101-budget-full-uayBXC/`.
- Base revision `fcb4a81a150b0eb272be6708315ea3fa64e00813`, uncommitted candidate,
  RTX 4080 SUPER, Isaac Lab 3.0.0 and Isaac Sim 6.2.0.0. The verification
  recomputed all 48 summaries from 1,536 raw first-episode rows and checked
  source and artifact hashes, finite losses, and 8,192 minibatch updates per
  fit. Within each learner and seed, initial deterministic commands were
  identical between the exploration settings.
- This establishes learned joint tracking for the three training poses in
  one PhysX simulation. It does not show unseen-goal transfer, physical arm
  performance, or object manipulation. The [original comparison](baseline.md)
  and its expansion decision remain unchanged.
