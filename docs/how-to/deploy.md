# Docker & Kubernetes

## Docker

Build and push the container image:

```bash
NAME=nexuml
IMAGE=harbor.ika.rub.de/library/${NAME}:latest

docker build -f Dockerfile -t ${IMAGE} .
docker login harbor.ika.rub.de -u student -p Student1234
docker push ${IMAGE}
```

Sync code to shared storage (e.g. JuiceFS):

```bash
rsync -a --info=progress2 --no-perms --no-owner --no-group --delete \
    ./ /mnt/juicefs/code/nexuml/
```

## Kubernetes

Schedule a PyTorchJob on the cluster:

```bash
kubectl config use-context local
kubectl delete -f .k8s/pytorchjob.yaml
kubectl apply -f .k8s/pytorchjob.yaml
```

The job spec at `.k8s/pytorchjob.yaml` defines the worker configuration, resource requests, and environment variables (`NEXUML_DATA_ROOT`, `NEXUML_LOGS_ROOT`).
