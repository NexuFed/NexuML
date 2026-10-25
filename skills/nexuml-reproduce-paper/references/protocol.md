# Reproduction checklist

Use the user's documentation format; this table is a starting point, not a new
database requirement. Keep exact source locations and executable checks together.

| Area | Facts to extract | Evidence/check |
| --- | --- | --- |
| Source | Paper revision, supplement, official code revision, target table/claim | Page/section or file/config and revision |
| Data | Release, access/license, manifest, exclusions, sample/group IDs, labels | Counts, checksums and deterministic identity checks |
| Partitions | Official splits, subject/device/time groups, fit/adaptation cohorts | Disjoint IDs/groups; no implicit random replacement |
| Preprocessing | Order, sample rate/resolution, padding/windowing, normalization, augmentation | Small reference tensors and justified tolerances |
| Algorithm | Equations, dimensions, activations, initialization, objectives | Numerical/gradient checks and key/shape contracts |
| Training | Batch/accumulation, optimizer/schedule, epochs/steps, checkpoint/early stopping | Resolved configuration and actual command/overrides |
| Fitting/scoring | Statistic fit scope, post-train fitting, score formula/direction | Known scores and train-only state fitting |
| Evaluation | Metric implementation, cohorts, grouping, weights, thresholds | Raw outputs plus independent aggregation check |
| Repetitions | Seeds, partitions per seed, confidence/variability, hardware precision | Per-run evidence; no invented multi-seed estimate |
| Claims | Parity tolerance, omitted details, deviations, blocked stages | Explicit comparison and supported claim level |

For every consequential ambiguity record **known / assumed / unresolved**.
Missing access or compute can legitimately block full reproduction; do not
replace the dataset/evaluator and claim the paper was recreated.

Suggested report sections:

1. Target and verified sources.
2. Protocol-to-library/config map.
3. Assumptions, conflicts and deviations.
4. Checks and experiment evidence (including failures).
5. Reference comparison and supported conclusion.
6. Separately scoped improvement candidates, if requested.
