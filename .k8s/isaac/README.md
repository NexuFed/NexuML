# Isaac workstation on Kubernetes

Use the [combined Isaac image](../../containers/isaac/README.md) from a
**network-restricted Harbor registry**, pinned by manifest digest. `pod.yaml` starts the
editor and bundled ROS 2 bridge; `job.yaml` runs separate headless training
using the same image. Use Kubernetes and Harbor only. No node login, image
preload, host networking, public Service, or node firewall change is needed.

## Registry and storage

- In this cluster, use the internal Harbor `library/nexuml-isaac` repository.
  The `library` project permits anonymous pulls **within the restricted
  network**; do not mirror it to a public registry. Both manifests reference the
  existing `rglitza/regcred` Harbor pull Secret. Do not commit credentials.
- Both workloads use one NVIDIA GPU and the `nvidia` RuntimeClass. The template
  targets compatible AMD64 RTX nodes; it may remain Pending until one is free.
- Both mount the `seaweedfs` claim. The Isaac process runs as `1234:1234` and
  needs write access to `/mnt/persistent/nexuml/isaac-workstation`. Do not
  recursively change ownership on an existing shared volume.

## Enroll a pod-local NetBird peer

The installed operator supports `netbird.io/setup-key` and `NBSetupKey`.
Use its **sidecar** path, not an HTTP ingress or a route to a public node.
The operator gives the sidecar `NET_ADMIN` for the Pod network namespace;
the Isaac process stays non-root with all capabilities dropped.

1. [Install NetBird on the laptop](https://docs.netbird.io/get-started/install)
   and sign in to the same NetBird account that will enroll the Pod.
2. In [Access Control](https://docs.netbird.io/manage/access-control), create
   a group dedicated to the Isaac peer and allow only the trusted laptop
   group → Isaac peer group on **TCP 49100** and **UDP 47998**. Check broader
   policies (especially `All → All`) before exposing Isaac. The stream has
   no built-in authentication. Use a scoped, short-lived setup key that
   auto-assigns the Isaac group. Do not share the key in chat or commit it.
3. Provision the Secret `isaac-netbird-enrollment` with key `setup-key` in
   `rglitza` through the normal secret management path. Apply `netbird.yaml`:
   it references the Secret and denies direct cluster ingress to the editor.
   NetBird may fall back to a relay. Check actual UDP media before relying on
   this policy in production.

```bash
kubectl apply -f .k8s/isaac/netbird.yaml
kubectl -n rglitza wait nbsetupkey/isaac-workstation \
  --for=condition=Ready --timeout=30s
```

## Start and connect

`pod.yaml` is pinned to the verified Harbor digest. If updating the image later,
change both workload manifests to the new verified digest. Check admission:

```bash
kubectl apply --dry-run=server -f .k8s/isaac/pod.yaml \
  -o jsonpath='{.spec.containers[*].name}{"\n"}'
```

Confirm the output contains **`isaac netbird`** before applying. The webhook
can fail open; an accepted manifest without the sidecar is not a working
setup. The Isaac launcher waits at most 90 seconds for a pod-local address
on `wt0` and advertises that address for WebRTC. It does not use a node IP
or a ClusterIP.

```bash
kubectl apply -f .k8s/isaac/pod.yaml
kubectl -n rglitza logs -f isaac-workstation -c isaac
```

After `Isaac Sim Full Streaming App is loaded.`, get the Pod's VPN address:

```bash
kubectl -n rglitza exec isaac-workstation -c netbird -- netbird status --ipv4
```

Enter that IP in the [NVIDIA WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/6.2.0/installation/download.html).
Confirm video and keyboard/mouse input. A TCP connection alone does not prove
UDP media works; `kubectl port-forward` does not forward UDP.

Use Kubernetes attach for remote editing if needed. Save USD scenes under
`/mnt/persistent/nexuml/isaac-workstation/assets`. The editor exits when the
Pod ends; saved files on the PVC remain. Use a new enrollment key if recreating
a Pod that used a single-use key.

## Train

Copy `job.yaml` and choose a new Job name. Keep the same Harbor digest as the editor.
Place your trusted `train.py` with `scenario()` under the PVC's `tasks/`
directory. Leave the scenario's execution backend local. The Job saves logs
under `runs/<job-name>` and has no NetBird sidecar or streaming ports.
Use explicit durable checkpoint and export paths. Scenarios referencing
NVIDIA online assets need outbound HTTPS or a prepared local asset root.

This is a deployment template. Verify image pulling, sidecar enrollment and
actual WebRTC before calling it a working workstation.
