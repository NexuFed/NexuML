"""Dataset composition must respect source-owned splits and canonical sample pairs."""

from typing import ClassVar

import pandas as pd
import pytest
import torch
from tensordict import TensorDict

from nexuml.core.components import DataSourceDefinition
from nexuml.core.types import DataSpec, DatasetSpec
from nexuml.data.creator import NexuDataCreator
from nexuml.data.dataset import NexuDataset
from nexuml.data.super_dataset import SuperDataset


class _FrameDataset(NexuDataset):
    def __init__(self):
        super().__init__(
            meta=pd.DataFrame(
                {
                    "frame": [10, 11, 12],
                    "split": ["train", "val", "test"],
                }
            ),
            label_names=[],
        )

    def validate_split_type(self, split_type):
        if split_type != "keep":
            raise ValueError("source requires keep")

    def __getitem__(self, index):
        assert self.meta is not None
        value = torch.tensor(int(self.meta.iloc[index]["frame"]))
        return TensorDict({"state": value}, []), TensorDict({"action": value + 1}, [])


class _FrameSource(DataSourceDefinition):
    component_name: ClassVar[str] = "FrameSource"

    def build(self):
        return _FrameDataset()


@pytest.mark.parametrize("position", [0, 1])
@pytest.mark.parametrize("split_type", ["fit", "all"])
def test_creator_validates_each_source_before_merged_splits(position, split_type):
    sources = [DatasetSpec(source=_FrameSource(), split_type="keep") for _ in range(2)]
    sources[position] = DatasetSpec(source=_FrameSource(), split_type=split_type)
    with pytest.raises(ValueError, match="source requires keep"):
        NexuDataCreator().build_dataset(DataSpec(datasets=sources))


def test_composition_preserves_source_sample_pairs_and_nested_views():
    original = _FrameDataset().take([2, 0])
    combined = SuperDataset({"frames": original}, label_names=[])
    reordered = combined.take([1, 0]).get_split("train")
    x, y = reordered[0]
    torch.testing.assert_close(x["state"], torch.tensor(10))
    torch.testing.assert_close(x["sample_index"], torch.tensor(0))
    assert y is not None
    torch.testing.assert_close(y["action"], torch.tensor(11))


def test_metadata_label_composition_retains_its_existing_labels():
    source = NexuDataset(
        meta=pd.DataFrame({"target": [7.0, 8.0], "split": ["train", "val"]}),
        data=[torch.tensor([1.0]), torch.tensor([2.0])],
        label_names=["target"],
    )
    combined = SuperDataset({"source": source}, label_names=["target"])
    x, y = combined[1]
    torch.testing.assert_close(x["features"], torch.tensor([2.0]))
    assert y is not None
    torch.testing.assert_close(y["target"], torch.tensor(8.0))


def test_frame_pairs_survive_composition_with_a_metadata_target_source():
    frames = _FrameDataset()
    labelled = NexuDataset(
        meta=pd.DataFrame({"action": [7.0], "split": ["train"]}),
        data=[torch.tensor([1.0])],
        label_names=["action"],
    )
    combined = SuperDataset({"frames": frames, "labelled": labelled}, label_names=["action"])
    x, y = combined[0]
    torch.testing.assert_close(x["state"], torch.tensor(10))
    assert y is not None
    torch.testing.assert_close(y["action"], torch.tensor(11))
    _, other_target = combined[3]
    assert other_target is not None
    torch.testing.assert_close(other_target["action"], torch.tensor(7.0))


def test_explicit_metadata_label_merging_is_not_bypassed():
    source = NexuDataset(
        meta=pd.DataFrame({"target": [7.0, 8.0], "split": ["train", "val"]}),
        data=[torch.tensor([1.0]), torch.tensor([2.0])],
        label_names=["target"],
    )
    combined = SuperDataset(
        {"source": source},
        label_names=["target", "merged"],
        merge_labels={"merged": {"columns": ["target"]}},
    )
    _, y = combined[1]
    assert y is not None
    torch.testing.assert_close(y["merged"], torch.tensor(1.0))
    torch.testing.assert_close(y["target"], torch.tensor(8.0))


def test_categorical_metadata_labels_are_merged_before_tensor_conversion():
    source = NexuDataset(
        meta=pd.DataFrame({"target": ["cat", "dog"], "split": ["train", "val"]}),
        data=[torch.tensor([1.0]), torch.tensor([2.0])],
        label_names=["target"],
    )
    combined = SuperDataset(
        {"source": source},
        label_names=["merged"],
        merge_labels={"merged": {"columns": ["target"]}},
    )
    x, y = combined[1]
    torch.testing.assert_close(x["features"], torch.tensor([2.0]))
    assert y is not None and set(y.keys()) == {"merged"}
    torch.testing.assert_close(y["merged"], torch.tensor(1.0))
