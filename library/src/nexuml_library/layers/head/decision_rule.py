"""DecisionRulePipelineLayer — post-training fit layer for binary decision thresholding."""

from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDict

from nexuml.core.discovery import layer
from nexuml.core.post_train_layer import PostTrainFitLayer
from nexuml_library.evaluation.anomalous_sound_detection.decision import (
    DecisionRule,
    create_decision_rule,
)
from nexuml_library.layers.head._td_utils import get_fit_mask_from_td


@layer("decision_rule_pipeline_layer")
class DecisionRulePipelineLayer(PostTrainFitLayer):
    """Fits a decision threshold on train scores; emits binary decisions at inference.

    Replaces DecisionRuleAlgorithm as a proper pipeline layer.
    """

    def __init__(
        self,
        score_key: str = "anomaly_score",
        decision_key: str = "decision",
        rule_type: str = "percentile",
        rule_params: dict | None = None,
        fit_mask_key: str | None = None,
        fit_label_key: str | None = None,
        normal_label_value: int | float = 0,
        input_sizes: dict | None = None,
        keys_in: list | None = None,
        keys_out: list | None = None,
        **kwargs,
    ) -> None:
        super().__init__(
            input_sizes=input_sizes or {},
            keys_in=keys_in or [score_key],
            keys_out=keys_out or [decision_key],
            **kwargs,
        )
        self.score_key = score_key
        self.decision_key = decision_key
        self.rule_type = rule_type
        self.rule_params = dict(rule_params or {})
        self.fit_mask_key = fit_mask_key
        self.fit_label_key = fit_label_key
        self.normal_label_value = normal_label_value

        self._rule: DecisionRule | None = None
        self._acc_scores: list[torch.Tensor] = []

    def collect_batch(self, x: TensorDict, y: TensorDict | None) -> None:
        scores = x[self.score_key].float().cpu().flatten()
        n = scores.shape[0]
        mask = get_fit_mask_from_td(
            x, y, self.fit_mask_key, self.fit_label_key, self.normal_label_value, n
        )
        self._acc_scores.append(scores[mask])

    def finalize_fit(self) -> None:
        self._rule = create_decision_rule(self.rule_type, **self.rule_params)
        all_scores = torch.cat(self._acc_scores, dim=0) if self._acc_scores else torch.zeros(0)
        self._rule.fit(all_scores)
        self._acc_scores = []

    def _transform_forward(
        self, x: TensorDict, y: TensorDict | None
    ) -> tuple[TensorDict, TensorDict | None]:
        assert self._rule is not None
        scores = x[self.score_key].float().cpu().flatten()
        decisions = self._rule(scores)
        x[self.decision_key] = decisions.to(x.device if hasattr(x, "device") else "cpu")
        return x, y

    def _get_fit_state(self) -> dict[str, Any]:
        return {"rule": self._rule}

    def _set_fit_state(self, state: dict[str, Any]) -> None:
        self._rule = state.get("rule")
