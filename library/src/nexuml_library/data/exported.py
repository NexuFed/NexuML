"""Typed definition for loading NexuML exported datasets."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from nexuml.core.components import DataSourceDefinition
from nexuml.core.discovery import data_source

if TYPE_CHECKING:
    from nexuml.data.dataset import NexuDataset


@data_source("ExportedDataset")
class ExportedDataset(DataSourceDefinition):
    """Load a local or S3 dataset previously written by ``export_data_module``."""

    root: str | Path
    split: str | list[str] | None = None
    feature_keys: list[str] | None = None
    label_keys: list[str] | None = None
    label_prefix: str = "label__"
    s3_endpoint_url: str | None = None
    s3_region: str | None = None
    s3_profile: str | None = None

    def build(self) -> NexuDataset:
        from nexuml.data.exported import ExportedDataset as Runtime

        return Runtime(**self.model_dump())


__all__ = ["ExportedDataset"]
