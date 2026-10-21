"""Typed bounded native MARL qualification settings; never imported for inference."""

from typing import Any


def short_horizon(cfg: Any) -> None:
    cfg.episode_length_s = 0.1
