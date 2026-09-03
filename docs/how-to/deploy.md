# Deployment boundary

NexuML intentionally does not prescribe one organization's container registry, Kubernetes cluster, shared filesystem, credentials, or scheduling policy.

For deployment, separate two concerns:

1. **NexuML artifact/runtime** — package the trained pipeline and install the runtime dependencies recorded by the export.
2. **Infrastructure** — choose the container image, registry, secrets, volumes/object storage, node placement, ingress, and scheduler appropriate for the target environment.

## Package the model

See [Export and reload](export.md) for the train-package contract. A package contains the pipeline artifact, state dict, resolved config, metadata, and dependency snapshot needed to reconstruct or directly load the model.

## Containers

Build the consuming application image from its own locked dependency environment. Do not embed site-specific registry credentials or cluster addresses in reusable NexuML scenarios/documentation.

## Distributed training

For training on an existing Ray cluster, use [Ray execution](training-backends/ray.md). KubeRay/RayJob cluster manifests remain infrastructure-owned; their entrypoint can invoke the same `nexuml train ...` command.

This page intentionally contains no organization-specific Kubernetes manifest because those details are deployment policy rather than part of the NexuML public API.
