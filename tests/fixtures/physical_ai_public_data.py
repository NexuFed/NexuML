"""Fixed public S3 snapshot -> worker scratch -> existing local LeRobot reader."""

from __future__ import annotations

import hashlib
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from typing import Literal


PREFIX = "data/so101/armnetbench-block-stack/5eb6bde28a86e6bd713587bf006b4754c128b027/"
SOURCE_URI = "s3://rglitza/" + PREFIX
MANIFEST_SHA256 = "929c0fa480905f06e94cc8ae6653407b3f5cf929764a20ab00c60174f5a5ccb4"
FLOOR = 4 * 1024**3


def manifest(payload):
    if hashlib.sha256(payload).hexdigest() != MANIFEST_SHA256:
        raise ValueError("public S3 manifest differs from the pinned snapshot")
    value = json.loads(payload)
    files = value["files"]
    if len(files) != 15 or sum(entry["bytes"] for entry in files) != 624607954:
        raise ValueError("public snapshot inventory mismatch")
    names = [entry["path"] for entry in files]
    if len(set(names)) != len(names):
        raise ValueError("duplicate public snapshot path")
    for entry in files:
        path = PurePosixPath(entry["path"])
        if path.is_absolute() or ".." in path.parts or "\\" in entry["path"]:
            raise ValueError("unsafe public snapshot path")
    return value


def download(client, value, destination):
    """Read only pinned keys with byte/hash checks and exclusive local files."""
    for entry in value["files"]:
        if shutil.disk_usage(destination).free - entry["bytes"] <= FLOOR:
            raise RuntimeError("public download would breach the physical disk floor")
        target = destination / entry["path"]
        target.parent.mkdir(parents=True, exist_ok=True)
        response = client.get_object(Bucket="rglitza", Key=PREFIX + entry["path"])
        count, digest = 0, hashlib.sha256()
        with response["Body"] as source, target.open("xb") as output:
            if response["ContentLength"] != entry["bytes"]:
                raise ValueError("public S3 object size mismatch")
            while chunk := source.read(min(1024**2, entry["bytes"] - count + 1)):
                count += len(chunk)
                if count > entry["bytes"]:
                    raise ValueError("public S3 object exceeds the pinned size")
                output.write(chunk)
                digest.update(chunk)
        if count != entry["bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError("public S3 object content mismatch")


@contextmanager
def staged_dataset():
    import boto3
    from botocore import UNSIGNED
    from botocore.config import Config
    from nexuml.storage.s3 import S3Path

    # The existing outer guard owns setup/workload time, descendants and /tmp/ray disk.
    with tempfile.TemporaryDirectory(prefix="nexuml-public-so101-", dir="/tmp/ray") as directory:
        scratch = Path(directory)
        ca = scratch / "ca.pem"
        ca.write_text(os.environ["NEXUML_PUBLIC_DATA_CA_PEM"])
        client = boto3.client(
            "s3",
            endpoint_url="https://s3.ika.rub.de",
            verify=str(ca),
            region_name="us-east-1",
            config=Config(
                signature_version=UNSIGNED,
                connect_timeout=10,
                read_timeout=60,
                retries={"total_max_attempts": 1},
                s3={"addressing_style": "path"},
            ),
        )
        source = S3Path.parse(SOURCE_URI + "source-manifest.json")
        response = client.get_object(Bucket=source.bucket, Key=source.key)
        with response["Body"] as body:
            payload = body.read(65537)
        if len(payload) > 65536:
            raise ValueError("public S3 manifest exceeds its metadata bound")
        value = manifest(payload)
        root = scratch / "dataset"
        root.mkdir()
        download(client, value, root)
        yield root, value


def main(output):
    import torch

    from nexuml_library.data.lerobot import LeRobotDataSource

    with staged_dataset() as (root, value):
        splits: dict[Literal["train", "val", "test"], tuple[int, ...]] = {
            "train": tuple(range(40)),
            "val": tuple(range(40, 45)),
            "test": tuple(range(45, 50)),
        }
        dataset = LeRobotDataSource(
            root=str(root), repo_id=value["repo"], split_episodes=splits
        ).build()
        if (
            len(dataset) != 12975
            or dataset.meta is None
            or dataset.meta["episode_index"].nunique() != 50
        ):
            raise AssertionError("public LeRobot reader count mismatch")
        for index in (0, len(dataset) - 1):
            inputs, targets = dataset[index]
            if targets is None:
                raise AssertionError("public LeRobot reader targets absent")
            for tensor in (inputs["observation.state"], targets["action"]):
                if (
                    not isinstance(tensor, torch.Tensor)
                    or tuple(tensor.shape) != (6,)
                    or not torch.isfinite(tensor).all()
                ):
                    raise AssertionError("public LeRobot reader sample mismatch")
        result = {
            "source_uri": SOURCE_URI,
            "manifest_sha256": MANIFEST_SHA256,
            "files_sha256_verified": 15,
            "bytes_verified": value["total_bytes"],
            "episodes": 50,
            "frames": len(dataset),
            "split_episode_counts": {key: len(ids) for key, ids in splits.items()},
            "reader_samples": [0, len(dataset) - 1],
            "scope": (
                "S3 staging and actual LeRobotDataSource reading only; "
                "no training or frozen L2/native proof"
            ),
        }
        del dataset
    output.write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
