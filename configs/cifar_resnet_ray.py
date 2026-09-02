# Install:
# uv sync --python 3.12 --extra dali --extra s3 --extra ray
#
# Configure:
# export NEXUML_DATA_ROOT="/mnt/local"
# export NEXUML_CIFAR_S3_URI="s3://rglitza/datasets/cifar10"
# export NEXUML_RAY_STORAGE_PATH="s3://rglitza/nexuml/runs"
# export NEXUML_RAY_ADDRESS="ray://ray.ika.rub.de:10001"
# export AWS_ENDPOINT_URL="https://s3.ika.rub.de"
# export AWS_ACCESS_KEY_ID="anonymous"
# export AWS_SECRET_ACCESS_KEY="anonymous"
# export AWS_DEFAULT_REGION="us-east-1"

# HTTPS not working
# kubectl -n seaweedfs port-forward svc/seaweedfs-cluster-s3 8333:8333
# export AWS_ENDPOINT_URL="http://127.0.0.1:8333"

# SeaweedFS currently has no IAM identities, so arbitrary signing credentials work.
# Replace the two anonymous values when IAM is configured.


# uv run nexuml export-dataset cifar-resnet \
#   --output "$NEXUML_CIFAR_S3_URI" \
#   --backend webdataset \
#   --split train --split val --split test \
#   --samples-per-shard 2048 \
#   --s3-endpoint-url "$AWS_ENDPOINT_URL" \
#   --s3-region "$AWS_DEFAULT_REGION"

# Train on Ray:
# export AWS_ENDPOINT_URL="http://seaweedfs-cluster-s3.seaweedfs.svc.cluster.local:8333"
# uv run --python 3.12 nexuml train --scenario-file configs/cifar_resnet_ray.py
#
# NexuML forwards the AWS endpoint, credentials, and region to the Ray workers.

"""CIFAR-10 ResNet training on Ray using an exported S3 WebDataset."""

from __future__ import annotations

import os

from nexuml.core.types import DatasetSpec, RayClusterTarget, RayExecutionSpec, ScenarioSpec
from nexuml_library.scenarios.vision.cifar_resnet import cifar_resnet


def scenario() -> ScenarioSpec:
    """Build the CIFAR ResNet scenario with S3 data and Ray execution.

    Returns:
        Configured CIFAR ResNet scenario.
    """
    base = cifar_resnet(download=False)
    region = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-east-1"))
    data = base.data.model_copy(
        update={
            "source_type": "exported",
            "datasets": [
                DatasetSpec(
                    type_key="ExportedDataset",
                    params={
                        "root": os.environ["NEXUML_CIFAR_S3_URI"],
                        "s3_endpoint_url": os.environ["AWS_ENDPOINT_URL"],
                        "s3_region": region,
                    },
                    modality="image",
                    split_type="keep",
                )
            ],
        }
    )
    execution = RayExecutionSpec(
        target=RayClusterTarget(
            address=os.environ["NEXUML_RAY_ADDRESS"],
            working_dir=".",
            py_executable=(
                "uv run --python 3.12 --locked --extra ray --extra s3 --extra dali python"
            ),
        ),
        workers=int(os.getenv("NEXUML_RAY_WORKERS", "4")),
        resources_per_worker={
            "CPU": float(os.getenv("NEXUML_RAY_CPUS_PER_WORKER", "4")),
            "GPU": float(os.getenv("NEXUML_RAY_GPUS_PER_WORKER", "1")),
        },
        storage_path=os.getenv("NEXUML_RAY_STORAGE_PATH"),
    )
    return base.model_copy(
        update={
            "name": "cifar_resnet_ray",
            "data": data,
            "execution": execution,
        }
    )
