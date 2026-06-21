"""Configuration serialization for full scenario documents."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from ruamel.yaml import YAML

from nexuml.core.types import PipelineSpec, ScenarioSpec


class ResolvedConfig(ScenarioSpec):
    """Backward-compatible alias for a fully resolved scenario document."""

    name: str = ""
    pipeline: PipelineSpec = Field(default_factory=PipelineSpec)

    @classmethod
    def from_yaml(cls, yaml_str: str) -> "ResolvedConfig":
        yaml = YAML()
        data = yaml.load(yaml_str) or {}
        return cls.model_validate(data)

    def to_yaml(self) -> str:
        yaml = YAML()
        yaml.default_flow_style = False

        import io

        stream = io.StringIO()
        yaml.dump(self.model_dump(mode="json"), stream)
        return stream.getvalue()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.to_yaml())

    @classmethod
    def load(cls, path: Path) -> "ResolvedConfig":
        return cls.from_yaml(path.read_text())

    @classmethod
    def from_scenario(cls, scenario: ScenarioSpec) -> "ResolvedConfig":
        return cls.model_validate(scenario.model_dump())

    def to_scenario(self) -> ScenarioSpec:
        return ScenarioSpec.model_validate(self.model_dump())
