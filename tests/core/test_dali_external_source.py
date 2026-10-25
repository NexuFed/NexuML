"""Array layout contracts at the DALI Python input boundary."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("nvidia.dali")

from nvidia.dali.tensors import TensorCPU

from nexuml.data.loaders.dali_multimodal import ExternalSampleIterator, _to_numpy


@pytest.mark.parametrize("as_tensor", [False, True])
@pytest.mark.parametrize("layout", ["fortran", "sliced", "contiguous", "scalar"])
def test_dali_numpy_conversion_preserves_values_shape_and_dtype(as_tensor, layout):
    array = np.arange(64 * 101, dtype=np.float32).reshape(1, 64, 101)
    if layout == "fortran":
        array = np.asfortranarray(array)
    elif layout == "sliced":
        array = array[:, :, ::2]
    elif layout == "scalar":
        array = np.asarray(3, dtype=np.int64)
    value = torch.from_numpy(array) if as_tensor else array

    actual = _to_numpy(value)

    assert actual.shape == array.shape
    assert actual.dtype == array.dtype
    assert actual.flags.c_contiguous
    np.testing.assert_array_equal(actual, array)
    np.testing.assert_array_equal(np.asarray(TensorCPU(actual)), array)
    if not as_tensor and array.flags.c_contiguous:
        assert actual is array


def test_dali_external_source_accepts_transposed_features_and_scalar_labels():
    features = torch.arange(64 * 101, dtype=torch.float32).reshape(101, 64).T.unsqueeze(0)
    dataset = [
        (
            TensorDict({"features": features}, batch_size=[]),
            TensorDict({"class": torch.tensor(3)}, batch_size=[]),
        )
    ]
    iterator = ExternalSampleIterator(
        dataset,
        x_keys=["features"],
        y_keys=["class"],
        num_shards=1,
        shard_id=0,
        shuffle=False,
    )

    feature, label, index = next(iter(iterator))

    assert feature.flags.c_contiguous
    np.testing.assert_array_equal(np.asarray(TensorCPU(feature)), features.numpy())
    assert label.shape == index.shape == ()
    assert label.dtype == index.dtype == np.int64
    assert label.item() == 3
    assert index.item() == 0
