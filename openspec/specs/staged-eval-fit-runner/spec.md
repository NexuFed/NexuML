## Purpose

**DEPRECATED**: The staged evaluation fit runner has been removed. Score-producing components are now `PostTrainFitLayer` pipeline layers that run during the standard `predict` + `test` lifecycle via `NexuSession._fit_post_train_layers()`. There is no materialized cache and no separate eval runner.

**Migration**: Move `group_distance_estimator`, `score_calibrator`, `score_reducer`, and `decision_rule` from `evaluation.algorithms` to `pipeline.layers` in scenario configuration. Remove any use of `materialize_eval_cache` or `EvalContainer`.

## Requirements

_No active requirements. All requirements have been removed and migrated to `post-train-fit-orchestration` and `post-train-fit-pipeline-layers` specs._
