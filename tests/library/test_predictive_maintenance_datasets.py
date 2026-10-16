"""Tests for NEX-233 bearing sources without external dataset downloads."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest
import torch

from nexuml.data.super_dataset import SuperDataset
from nexuml_library.data.maintenance.bearings import (
    IMSBearingDataset,
    PRONOSTIABearingDataset,
    PaderbornBearingDataset,
    XJTUSYBearingDataset,
    _safe_extract,
)


def _save_csv(path: Path, *, columns: int, header: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.arange(24 * columns, dtype=np.float32).reshape(24, columns)
    np.savetxt(
        path, data, delimiter=",", fmt="%.2f",
        header=",".join(f"channel_{i}" for i in range(columns)) if header else "",
        comments="",
    )


def _assert_tensor_contract(dataset, channels: int, length: int = 16) -> None:
    x, y = dataset[0]
    assert y is not None
    assert x["features"].shape == (channels, length)
    assert x["features"].dtype == torch.float32
    assert x["sample_index"].dtype == torch.long
    assert set(y.keys()) == {"health", "rul", "rul_valid"}
    assert torch.isfinite(x["features"]).all()
    assert torch.isfinite(y["rul"])


def _assert_group_isolation(dataset) -> None:
    assert dataset.meta is not None
    meta = dataset.meta
    assert (meta.groupby("group_id")["split"].nunique() == 1).all()
    assert set(meta["split"].unique()) <= {"train", "val", "test"}


def test_paderborn_endpoints_have_no_fake_rul(tmp_path: Path) -> None:
    scipy_io = pytest.importorskip("scipy.io")
    for bearing in ("K001", "KA01"):
        file = tmp_path / bearing / f"N15_M07_F10_{bearing}_1.mat"
        file.parent.mkdir(parents=True)
        scipy_io.savemat(
            file, {"bearing": {"vibration": {"y_values": np.arange(32, dtype=np.float32)}}}
        )
    definition = PaderbornBearingDataset(root=tmp_path, window_size=16)
    runtime = definition.build()

    assert len(runtime) == 2
    assert runtime.meta is not None
    assert set(runtime.meta["health"]) == {0, 1}
    assert not runtime.meta["rul_valid"].any()
    assert (runtime.meta["split"] == "train").all()
    _assert_tensor_contract(runtime, 1)


def test_pronostia_full_and_truncated_runs(tmp_path: Path) -> None:
    for split, bearing in (("Learning_set", "Bearing1_1"), ("Test_set", "Bearing1_3")):
        for i in range(1, 4):
            _save_csv(tmp_path / split / bearing / f"acc_{i:05d}.csv", columns=6)

    runtime = PRONOSTIABearingDataset(root=tmp_path, window_size=16).build()
    assert runtime.meta is not None
    complete = runtime.meta[runtime.meta["bearing_id"] == "Bearing1_1"]
    truncated = runtime.meta[runtime.meta["bearing_id"] == "Bearing1_3"]

    assert complete["rul"].tolist() == [20.0, 10.0, 0.0]
    assert complete["rul_valid"].tolist() == [1, 1, 1]
    assert complete["health"].tolist() == [0, -1, 1]
    assert truncated["rul_valid"].tolist() == [0, 0, 0]
    assert truncated["health"].tolist() == [-1, -1, -1]
    assert (truncated["split"] == "test").all()
    _assert_group_isolation(runtime)
    _assert_tensor_contract(runtime, 2)


def test_xjtu_splits_are_group_disjoint_and_deterministic(tmp_path: Path) -> None:
    for run in ("Bearing1_1", "Bearing1_2", "Bearing1_3", "Bearing1_4", "Bearing1_5"):
        for index in (1, 2, 3):
            _save_csv(tmp_path / "35Hz12kN" / run / f"{index}.csv", columns=2, header=True)

    source = XJTUSYBearingDataset(root=tmp_path, window_size=16, seed=17)
    a = source.build()
    b = source.build()
    assert a.meta is not None and b.meta is not None
    _assert_group_isolation(a)
    assert a.meta["split"].tolist() == b.meta["split"].tolist()
    assert {"train", "val", "test"} <= set(a.meta["split"])
    assert a.meta.groupby("run_id")["rul"].apply(list).iloc[0] == [120.0, 60.0, 0.0]
    assert (a.meta["rul_valid"] == 1).all()
    _assert_tensor_contract(a, 2)


def test_xjtu_split_overrides_validate_keys(tmp_path: Path) -> None:
    _save_csv(tmp_path / "35Hz12kN" / "Bearing1_1" / "1.csv", columns=2)
    source = XJTUSYBearingDataset(
        root=tmp_path,
        split_map={"xjtu_sy:Bearing1_1": "val"},
    )
    runtime = source.build()
    assert runtime.meta is not None
    assert set(runtime.meta["split"]) == {"val"}
    with pytest.raises(ValueError, match="Unknown group_id"):
        XJTUSYBearingDataset(root=tmp_path, split_map={"xjtu_sy:unknown": "test"}).build()


def test_ims_preserves_full_experiment_groups_and_failure_identity(tmp_path: Path) -> None:
    snapshots = ("2003.10.22.12.00.00", "2003.10.22.12.10.00")
    for name, channels in (("1st_test", 8), ("2nd_test", 4), ("3rd_test", 4)):
        directory = tmp_path / name
        directory.mkdir()
        for snapshot in snapshots:
            np.savetxt(directory / snapshot, np.ones((24, channels)))

    runtime = IMSBearingDataset(root=tmp_path, window_size=16).build()
    assert runtime.meta is not None
    _assert_group_isolation(runtime)
    assert runtime.meta["group_id"].nunique() == 3
    assert set(runtime.meta["split"]) == {"train", "val", "test"}
    first_test = runtime.meta[runtime.meta["operating_condition"] == "1st_test"]
    assert (first_test[first_test["bearing_id"] == "1st_test:bearing1"]["rul_valid"] == 0).all()
    assert (first_test[first_test["bearing_id"] == "1st_test:bearing3"]["rul_valid"] == 1).all()
    assert first_test[first_test["bearing_id"] == "1st_test:bearing3"]["rul"].tolist() == [
        600.0, 0.0
    ]
    _assert_tensor_contract(runtime, 1)


def test_multisource_meta_keeps_validity_and_provenance(tmp_path: Path) -> None:
    _save_csv(tmp_path / "35Hz12kN" / "Bearing1_1" / "1.csv", columns=2)
    _save_csv(tmp_path / "35Hz12kN" / "Bearing1_1" / "2.csv", columns=2)
    labeled = XJTUSYBearingDataset(root=tmp_path).build()
    # A second source using a different run, without endpoint RUL.
    truncated_root = tmp_path / "partial"
    _save_csv(truncated_root / "Test_set" / "Bearing1_3" / "acc_00001.csv", columns=6)
    unlabeled = PRONOSTIABearingDataset(root=truncated_root).build()
    merged = SuperDataset(
        meta_data_list={"xjtu": labeled, "pronostia": unlabeled},
        label_names=["health", "rul", "rul_valid"],
    )
    assert merged.meta is not None
    assert {"source_dataset", "source_index", "group_id", "rul_valid"} <= set(
        merged.meta.columns
    )
    assert int(merged.meta["rul_valid"].sum()) == 2
    assert (merged.meta[merged.meta["rul_valid"] == 0]["rul"] == 0).all()


def test_unavailable_download_is_explicit_and_offline(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Official source"):
        PaderbornBearingDataset(root=tmp_path / "missing", download=True).build()
    with pytest.raises(FileNotFoundError, match="Official source"):
        PRONOSTIABearingDataset(root=tmp_path / "missing", download=False).build()
    with pytest.raises(ValueError, match="direct HTTPS"):
        XJTUSYBearingDataset(
            root=tmp_path / "missing",
            archive_url="http://example.org/bearings.zip",
        ).build()


def test_archives_reject_path_traversal(tmp_path: Path) -> None:
    archive = tmp_path / "bad.zip"
    with ZipFile(archive, "w") as output:
        output.writestr("../escape.txt", "no")
    with pytest.raises(ValueError, match="Unsafe ZIP"):
        _safe_extract(archive, tmp_path / "extract")
    assert not (tmp_path / "escape.txt").exists()
